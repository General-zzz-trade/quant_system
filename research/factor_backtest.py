"""Bridge from AlphaFactor to backtest engine.

Converts any AlphaFactor into a DecisionModule and provides convenience
functions for one-call backtesting and walk-forward validation.
"""

from __future__ import annotations

import csv
import logging
import math
import os
import tempfile
import uuid
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

from event.header import EventHeader
from event.types import EventType, IntentEvent, OrderEvent
from research.alpha_factor import AlphaFactor
from runner.backtest.csv_io import OhlcvBar, iter_ohlcv_csv
from runner.backtest.metrics import _build_trades_from_fills, _build_summary


@dataclass
class FactorStrategyConfig:
    """Configuration for factor-based trading strategy."""

    symbol: str = "BTCUSDT"
    long_threshold: float = 0.5
    short_threshold: float = -0.5
    position_pct: float = 0.5
    atr_window: int = 14
    atr_stop_multiple: float = 3.0
    zscore_window: int = 100
    cooldown_bars: int = 1


class FactorDecisionModule:
    """DecisionModule that trades based on a factor's z-score.

    Implements the decide(snapshot) protocol from engine/decision_bridge.py.
    Each bar: extract OHLCV → compute factor → z-score → threshold → trade.
    """

    def __init__(
        self,
        factor: AlphaFactor,
        config: Optional[FactorStrategyConfig] = None,
    ) -> None:
        self._factor = factor
        self._cfg = config or FactorStrategyConfig()
        self._bars: List[OhlcvBar] = []
        self._factor_history: List[float] = []
        self._position: int = 0  # 1=long, -1=short, 0=flat
        self._cooldown: int = 0
        self._atr_history: List[float] = []
        self._entry_price: float = 0.0

    def decide(self, snapshot: Any) -> Iterable[Any]:
        """Produce IntentEvents based on factor signal."""
        market, positions, event_id = _snapshot_views(snapshot)
        close = float(getattr(market, "close", 0) or getattr(market, "last_price", 0) or 0)
        high = float(getattr(market, "high", close) or close)
        low = float(getattr(market, "low", close) or close)
        open_ = float(getattr(market, "open", close) or close)
        volume = float(getattr(market, "volume", 0) or 0)
        ts = getattr(market, "last_ts", None) or getattr(market, "ts", None)

        if close == 0:
            return []

        from datetime import datetime, timezone
        bar = OhlcvBar(
            ts=ts or datetime.now(timezone.utc),
            o=Decimal(str(open_)),
            h=Decimal(str(high)),
            l=Decimal(str(low)),
            c=Decimal(str(close)),
            v=Decimal(str(volume)),
        )
        self._bars.append(bar)

        # Compute ATR incrementally
        if len(self._bars) >= 2:
            prev = self._bars[-2]
            tr = max(
                high - low,
                abs(high - float(prev.c)),
                abs(low - float(prev.c)),
            )
        else:
            tr = high - low
        self._atr_history.append(tr)

        # Compute factor on full bar history
        vals = self._factor.compute_fn(self._bars)
        latest = vals[-1] if vals else None
        if latest is None:
            return []
        self._factor_history.append(latest)

        # Z-score
        window = min(self._cfg.zscore_window, len(self._factor_history))
        if window < 10:
            return []
        recent = self._factor_history[-window:]
        mean = sum(recent) / len(recent)
        var = sum((v - mean) ** 2 for v in recent) / (len(recent) - 1)
        std = math.sqrt(var) if var > 0 else 0.0
        if std < 1e-12:
            return []
        zscore = (latest - mean) / std

        # ATR for stop-loss
        atr_win = min(self._cfg.atr_window, len(self._atr_history))
        atr = sum(self._atr_history[-atr_win:]) / atr_win if atr_win > 0 else 0.0

        events: List[Any] = []
        symbol = self._cfg.symbol

        # Cooldown
        if self._cooldown > 0:
            self._cooldown -= 1

        # Stop-loss check
        if self._position != 0 and atr > 0:
            stop_dist = atr * self._cfg.atr_stop_multiple
            if self._position == 1 and close < self._entry_price - stop_dist:
                events.extend(self._make_intent_and_order(symbol, "sell", self._cfg.position_pct, "risk"))
                self._position = 0
                self._cooldown = self._cfg.cooldown_bars
                return events
            if self._position == -1 and close > self._entry_price + stop_dist:
                events.extend(self._make_intent_and_order(symbol, "buy", self._cfg.position_pct, "risk"))
                self._position = 0
                self._cooldown = self._cfg.cooldown_bars
                return events

        if self._cooldown > 0:
            return []

        # Signal generation
        if zscore > self._cfg.long_threshold and self._position <= 0:
            if self._position == -1:
                events.extend(self._make_intent_and_order(symbol, "buy", self._cfg.position_pct, "signal"))
            events.extend(self._make_intent_and_order(symbol, "buy", self._cfg.position_pct, "signal"))
            self._position = 1
            self._entry_price = close
            self._cooldown = self._cfg.cooldown_bars
        elif zscore < self._cfg.short_threshold and self._position >= 0:
            if self._position == 1:
                events.extend(self._make_intent_and_order(symbol, "sell", self._cfg.position_pct, "signal"))
            events.extend(self._make_intent_and_order(symbol, "sell", self._cfg.position_pct, "signal"))
            self._position = -1
            self._entry_price = close
            self._cooldown = self._cfg.cooldown_bars

        return events

    def _make_intent_and_order(self, symbol: str, side: str, pct: float, reason: str) -> tuple:
        intent_id = f"fct_{uuid.uuid4().hex[:16]}"
        order_id = f"fco_{uuid.uuid4().hex[:16]}"
        source = f"factor:{self._factor.name}"

        intent_h = EventHeader.new_root(
            event_type=EventType.INTENT, version=1, source=source,
        )
        order_h = EventHeader.from_parent(
            parent=intent_h, event_type=EventType.ORDER, version=1, source=source,
        )
        intent = IntentEvent(
            header=intent_h,
            intent_id=intent_id,
            symbol=symbol,
            side=side,
            target_qty=Decimal(str(pct)),
            reason_code=reason,
            origin=source,
        )
        order = OrderEvent(
            header=order_h,
            order_id=order_id,
            intent_id=intent_id,
            symbol=symbol,
            side=side,
            qty=Decimal(str(pct)),
            price=None,
        )
        return intent, order


def _snapshot_views(snapshot: Any):
    """Extract market, positions, event_id from snapshot."""
    if hasattr(snapshot, "market") and hasattr(snapshot, "positions"):
        return getattr(snapshot, "market"), getattr(snapshot, "positions"), getattr(snapshot, "event_id", None)
    if isinstance(snapshot, dict):
        market = snapshot.get("market")
        if market is None:
            markets = snapshot.get("markets") or {}
            market = next(iter(markets.values()), None) if markets else None
        positions = snapshot.get("positions") or {}
        event_id = snapshot.get("event_id")
        if market is None:
            raise RuntimeError("snapshot missing market/markets")
        return market, positions, event_id
    raise RuntimeError(f"unsupported snapshot type: {type(snapshot).__name__}")


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def backtest_factor(
    factor: AlphaFactor,
    csv_path: Path,
    *,
    symbol: str = "BTCUSDT",
    starting_balance: Decimal = Decimal("10000"),
    fee_bps: Decimal = Decimal("4"),
    slippage_bps: Decimal = Decimal("0"),
    config: Optional[FactorStrategyConfig] = None,
    out_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """One-call factor backtest → summary dict."""
    from runner.backtest_runner import run_backtest

    cfg = config or FactorStrategyConfig(symbol=symbol)
    module = FactorDecisionModule(factor, cfg)

    equity, fills = run_backtest(
        csv_path=csv_path,
        symbol=symbol,
        starting_balance=starting_balance,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        decision_modules=[module],
        out_dir=out_dir,
    )

    trades = _build_trades_from_fills(fills)
    summary = _build_summary(equity=equity, trades=trades, csv_path=csv_path, symbol=symbol)
    summary["factor_name"] = factor.name
    return summary


def walk_forward_factor(
    factor: AlphaFactor,
    csv_path: Path,
    *,
    symbol: str = "BTCUSDT",
    starting_balance: Decimal = Decimal("10000"),
    fee_bps: Decimal = Decimal("4"),
    slippage_bps: Decimal = Decimal("0"),
    config: Optional[FactorStrategyConfig] = None,
    train_size: int = 500,
    test_size: int = 100,
    out_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """One-call walk-forward validation → list of fold summaries."""
    all_bars = list(iter_ohlcv_csv(csv_path))
    if len(all_bars) < train_size + test_size:
        raise ValueError(
            f"Not enough bars for walk-forward: need {train_size + test_size}, got {len(all_bars)}"
        )

    cfg = config or FactorStrategyConfig(symbol=symbol)
    results: List[Dict[str, Any]] = []
    window_idx = 0
    start = 0

    while start + train_size + test_size <= len(all_bars):
        test_start = start + train_size
        test_end = test_start + test_size
        window_bars = all_bars[start:test_end]

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, newline="",
        ) as tmp:
            writer = csv.writer(tmp)
            writer.writerow(["ts", "open", "high", "low", "close", "volume"])
            for bar in window_bars:
                writer.writerow([
                    bar.ts.isoformat(),
                    str(bar.o), str(bar.h), str(bar.l), str(bar.c),
                    str(bar.v) if bar.v is not None else "0",
                ])
            tmp_path = tmp.name

        try:
            from runner.backtest_runner import run_backtest

            module = FactorDecisionModule(factor, cfg)
            equity, fills = run_backtest(
                csv_path=Path(tmp_path),
                symbol=symbol,
                starting_balance=starting_balance,
                fee_bps=fee_bps,
                slippage_bps=slippage_bps,
                decision_modules=[module],
            )

            test_equity = equity[train_size:] if len(equity) > train_size else equity
            trades = _build_trades_from_fills(fills)
            summary = _build_summary(
                equity=test_equity, trades=trades, csv_path=csv_path, symbol=symbol,
            )
            summary["window_idx"] = window_idx
            summary["train_bars"] = train_size
            summary["test_bars"] = test_size
            summary["factor_name"] = factor.name
            results.append(summary)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError as e:
                logger.debug("Failed to clean up temp file %s: %s", tmp_path, e)

        start += test_size
        window_idx += 1

    return results
