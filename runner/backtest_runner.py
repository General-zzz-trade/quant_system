"""BacktestRunner -- offline historical backtesting entry point.

Uses runner/backtest/ subpackage for execution simulation, CSV I/O, and metrics.
For production trading, see runner/live_runner.py.
"""
from __future__ import annotations

import json
import csv
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


def _ensure_utc(ts):
    """Convert a datetime to UTC, treating naive datetimes as UTC."""
    if ts is None:
        return None
    if not isinstance(ts, datetime):
        raise TypeError(f"ts must be datetime, got {type(ts).__name__}")
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)

from engine.coordinator import CoordinatorConfig, EngineCoordinator  # noqa: E402
from engine.decision_bridge import DecisionBridge  # noqa: E402
from engine.execution_bridge import ExecutionBridge  # noqa: E402
from event.header import EventHeader  # noqa: E402
from event.types import EventType, MarketEvent, IntentEvent, OrderEvent  # noqa: E402
from _quant_hotpath import RustPositionState as PositionState  # noqa: E402
from state import PortfolioState, RiskState, RiskLimits  # noqa: E402,F401 — state type aliases for backtest snapshot typing

# Re-export from subpackage for backward compatibility
from runner.backtest.csv_io import iter_ohlcv_csv  # noqa: E402
from runner.backtest.adapter import BacktestExecutionAdapter, _make_id  # noqa: E402
from runner.backtest.metrics import (  # noqa: E402
    EquityPoint,
    _max_drawdown,  # noqa: F401 -- re-exported for backtest_cli.py
    _build_trades_from_fills,
    _build_summary,
    _json_safe,
)


# ============================================================
# Decision module
# ============================================================


class MovingAverageCrossModule:
    def __init__(self, *, symbol: str, window: int, order_qty: Decimal, origin: str = "ma_cross") -> None:
        self.symbol = symbol.upper()
        self.window = int(window)
        self.order_qty = Decimal(str(order_qty))
        self.origin = origin
        self._closes: List[Decimal] = []

    def decide(self, snapshot: Any) -> Iterable[Any]:
        market, positions, event_id = _snapshot_views(snapshot)
        close = getattr(market, "close", None) or getattr(market, "last_price", None)
        if close is None:
            return ()

        close_d = Decimal(str(close))
        self._closes.append(close_d)
        if len(self._closes) > self.window:
            self._closes.pop(0)
        if len(self._closes) < self.window:
            return ()

        ma = sum(self._closes) / Decimal(str(self.window))

        pos = positions.get(self.symbol) or PositionState.empty(self.symbol)
        qty = getattr(pos, "qty", Decimal("0"))

        want_long = close_d > ma

        events: List[Any] = []
        if qty == 0 and want_long:
            events.extend(self._open_long(event_id=event_id))
        elif qty > 0 and (not want_long):
            events.extend(self._close_long(qty=qty, event_id=event_id))

        return events

    def _open_long(self, *, event_id: Optional[str]) -> Sequence[Any]:
        intent_id = _make_id("intent")
        order_id = _make_id("order")

        intent_h = EventHeader.new_root(
            event_type=EventType.INTENT,
            version=1,
            source=f"decision:{self.origin}",
            correlation_id=str(event_id) if event_id else None,
        )
        order_h = EventHeader.from_parent(
            parent=intent_h,
            event_type=EventType.ORDER,
            version=1,
            source=f"decision:{self.origin}",
        )

        return (
            IntentEvent(
                header=intent_h,
                intent_id=intent_id,
                symbol=self.symbol,
                side="buy",
                target_qty=self.order_qty,
                reason_code="ma_cross_long",
                origin=self.origin,
            ),
            OrderEvent(
                header=order_h,
                order_id=order_id,
                intent_id=intent_id,
                symbol=self.symbol,
                side="buy",
                qty=self.order_qty,
                price=None,
            ),
        )

    def _close_long(self, *, qty: Decimal, event_id: Optional[str]) -> Sequence[Any]:
        intent_id = _make_id("intent")
        order_id = _make_id("order")

        intent_h = EventHeader.new_root(
            event_type=EventType.INTENT,
            version=1,
            source=f"decision:{self.origin}",
            correlation_id=str(event_id) if event_id else None,
        )
        order_h = EventHeader.from_parent(
            parent=intent_h,
            event_type=EventType.ORDER,
            version=1,
            source=f"decision:{self.origin}",
        )

        q = abs(qty)
        return (
            IntentEvent(
                header=intent_h,
                intent_id=intent_id,
                symbol=self.symbol,
                side="sell",
                target_qty=q,
                reason_code="ma_cross_exit",
                origin=self.origin,
            ),
            OrderEvent(
                header=order_h,
                order_id=order_id,
                intent_id=intent_id,
                symbol=self.symbol,
                side="sell",
                qty=q,
                price=None,
            ),
        )


def _snapshot_views(snapshot: Any) -> Tuple[Any, Mapping[str, Any], Optional[str]]:
    if hasattr(snapshot, "market") and hasattr(snapshot, "positions"):
        market = getattr(snapshot, "market")
        positions = getattr(snapshot, "positions")
        event_id = getattr(snapshot, "event_id", None)
        return market, positions, event_id

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


# ============================================================
# Runner
# ============================================================


def run_backtest(
    *,
    csv_path: Path,
    symbol: str,
    starting_balance: Decimal,
    ma_window: int = 20,
    order_qty: Decimal = Decimal("0.01"),
    fee_bps: Decimal,
    slippage_bps: Decimal = Decimal("0"),
    out_dir: Optional[Path] = None,
    embargo_bars: int = 1,
    funding_csv: Optional[str] = None,
    decision_modules: Optional[List[Any]] = None,
    feature_computer: Any = None,
    feature_hook: Any = None,
    alpha_models: Optional[List[Any]] = None,
    enable_attribution: bool = False,
    enable_regime_gate: bool = False,
) -> Tuple[List[EquityPoint], List[Dict[str, Any]]]:
    symbol_u = symbol.upper()

    equity: List[EquityPoint] = []
    fills: List[Dict[str, Any]] = []

    def _on_fill(ev: Any) -> None:
        ts_val = ""
        ev_ts = getattr(ev, "ts", None)
        if isinstance(ev_ts, datetime):
            ts_val = ev_ts.isoformat()
        else:
            ts_ns = getattr(getattr(ev, "header", None), "ts_ns", None)
            ts_val = str(ts_ns) if ts_ns is not None else ""

        fills.append(
            {
                "ts": ts_val,
                "symbol": getattr(ev, "symbol", None),
                "side": getattr(ev, "side", None),
                "qty": getattr(ev, "qty", None),
                "price": getattr(ev, "price", None),
                "fee": getattr(ev, "fee", None),
                "realized_pnl": getattr(ev, "realized_pnl", None),
                "event_id": getattr(getattr(ev, "header", None), "event_id", None),
                "root_event_id": getattr(getattr(ev, "header", None), "root_event_id", None),
            }
        )

    def _on_pipeline(out: Any) -> None:
        m = out.market
        a = out.account
        positions = out.positions
        ts = getattr(m, "last_ts", None)
        if ts is None:
            return
        # Rust types return last_ts as ISO string; convert to datetime
        if isinstance(ts, str):
            raw = ts.strip()
            if raw.endswith("Z"):
                raw = raw[:-1] + "+00:00"
            ts = _ensure_utc(datetime.fromisoformat(raw))

        # Use float accessors if available (Rust types), else Decimal
        close_f = getattr(m, "close_f", None)
        if close_f is not None:
            close_d = Decimal(str(close_f))
        else:
            close = getattr(m, "close", None) or getattr(m, "last_price", None)
            if close is None:
                return
            close_d = Decimal(str(close))

        pos = positions.get(symbol_u)
        if pos is None:
            qty = Decimal("0")
            avg = None
        else:
            qf = getattr(pos, "qty_f", None)
            qty = Decimal(str(qf)) if qf is not None else Decimal(str(getattr(pos, "qty", 0)))
            af = getattr(pos, "avg_price_f", None)
            avg = Decimal(str(af)) if af is not None else getattr(pos, "avg_price", None)

        unreal = Decimal("0")
        if qty != 0 and avg is not None:
            unreal = (close_d - avg) * qty

        bf = getattr(a, "balance_f", None)
        bal = Decimal(str(bf)) if bf is not None else getattr(a, "balance", Decimal("0"))
        rf = getattr(a, "realized_pnl_f", None)
        realized = Decimal(str(rf)) if rf is not None else getattr(a, "realized_pnl", Decimal("0"))
        eq = bal + unreal

        assert isinstance(ts, datetime)
        equity.append(
            EquityPoint(
                ts=ts,
                close=close_d,
                position_qty=qty,
                avg_price=avg,
                balance=bal,
                realized=realized,
                unrealized=unreal,
                equity=eq,
            )
        )

    # Feature compute hook (same as live runner)
    # feature_hook takes precedence (pre-built hook, e.g. PrecomputedFeatureHook)
    feat_hook = feature_hook
    if feat_hook is None and feature_computer is not None:
        from engine.feature_hook import FeatureComputeHook
        inference_bridge = None
        if alpha_models:
            from alpha.inference.bridge import LiveInferenceBridge
            inference_bridge = LiveInferenceBridge(models=list(alpha_models))
        feat_hook = FeatureComputeHook(
            computer=feature_computer, inference_bridge=inference_bridge,
        )

    # Attribution tracker (same as live runner)
    attribution_tracker = None
    if enable_attribution:
        from attribution.tracker import AttributionTracker
        attribution_tracker = AttributionTracker()

    coordinator = EngineCoordinator(
        cfg=CoordinatorConfig(
            symbol_default=symbol_u,
            currency="USDT",
            starting_balance=float(starting_balance),
            on_pipeline_output=_on_pipeline,
            feature_hook=feat_hook,
        )
    )

    def _emit(ev: Any) -> None:
        if attribution_tracker is not None:
            attribution_tracker.on_event(ev)
        coordinator.emit(ev, actor="backtest")

    def _price(sym: str) -> Optional[Decimal]:
        view = coordinator.get_state_view()
        m = view.get("market")
        if m is None:
            return None
        px = getattr(m, "close", None) or getattr(m, "last_price", None)
        return px

    def _ts() -> Optional[datetime]:
        view = coordinator.get_state_view()
        m = view.get("market")
        return getattr(m, "last_ts", None) if m is not None else None

    from execution.sim.embargo import EmbargoExecutionAdapter

    base_adapter = BacktestExecutionAdapter(
        price_source=_price,
        ts_source=_ts,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        source="paper",
        on_fill=_on_fill,
    )

    embargo_adapter = EmbargoExecutionAdapter(inner=base_adapter, embargo_bars=embargo_bars)
    exec_bridge = ExecutionBridge(adapter=embargo_adapter, dispatcher_emit=_emit)

    # Pluggable decision modules — fallback to MovingAverageCross for backward compat
    if decision_modules is not None:
        modules = decision_modules
    else:
        modules = [MovingAverageCrossModule(symbol=symbol_u, window=ma_window, order_qty=order_qty)]

    if enable_regime_gate and modules:
        from decision.regime_bridge import RegimeAwareDecisionModule
        from decision.regime_policy import RegimePolicy
        modules = [
            RegimeAwareDecisionModule(inner=m, policy=RegimePolicy())
            for m in modules
        ]

    decision_bridge = DecisionBridge(dispatcher_emit=_emit, modules=modules)

    coordinator.attach_execution_bridge(exec_bridge)
    coordinator.attach_decision_bridge(decision_bridge)

    # Build funding schedule if funding CSV provided
    funding_schedule: Dict[datetime, Any] = {}
    if funding_csv is not None:
        from data.loaders.funding_rate import load_funding_csv, funding_schedule_for_bars
        funding_records = load_funding_csv(funding_csv, symbol=symbol_u)
        bar_list = list(iter_ohlcv_csv(csv_path))
        bar_timestamps = [b.ts for b in bar_list]
        funding_schedule = funding_schedule_for_bars(bar_timestamps, funding_records)
    else:
        bar_list = list(iter_ohlcv_csv(csv_path))

    coordinator.start()

    for i, bar in enumerate(bar_list):
        if embargo_bars > 0:
            # Fill embargoed orders at this bar's OPEN price — the first
            # tradeable price after the embargo window, not the stale
            # previous bar's close sitting in the coordinator state.
            for fill_ev in embargo_adapter.on_bar(i, open_price=bar.o):
                coordinator.emit(fill_ev, actor="backtest")
        embargo_adapter.set_bar(i)

        h = EventHeader.new_root(event_type=EventType.MARKET, version=MarketEvent.VERSION, source="csv")
        ev = MarketEvent(
            header=h,
            ts=bar.ts,
            symbol=symbol_u,
            open=bar.o,
            high=bar.h,
            low=bar.l,
            close=bar.c,
            volume=bar.v if bar.v is not None else Decimal("0"),
        )
        coordinator.emit(ev, actor="replay")

        if bar.ts in funding_schedule:
            fr = funding_schedule[bar.ts]
            pos_qty = base_adapter._pos_qty.get(symbol_u, Decimal("0"))
            if pos_qty != 0 and fr.funding_rate != 0:
                # Funding payment: rate × position notional
                # Positive rate + long position = you pay; negative = you receive
                close_px = bar.c
                funding_cost = fr.funding_rate * pos_qty * close_px
                # Record as a fill with zero qty (funding settlement)
                fh = EventHeader.new_root(event_type=EventType.FILL, version=1, source="funding")
                funding_fill = SimpleNamespace(
                    header=fh,
                    event_type=EventType.FILL,
                    ts=bar.ts,
                    symbol=symbol_u,
                    side="buy" if pos_qty > 0 else "sell",
                    qty=Decimal("0"),
                    price=close_px,
                    fee=abs(funding_cost),
                    realized_pnl=-funding_cost,  # Cost reduces PnL
                    cash_delta=float(-funding_cost),
                    margin_change=0.0,
                )
                if _on_fill is not None:
                    _on_fill(funding_fill)

    if embargo_bars > 0 and bar_list:
        # End-of-data flush: no next bar available, use last bar's close
        # as best-effort fill price (conservative: no better data exists).
        for fill_ev in embargo_adapter.on_bar(len(bar_list), open_price=bar_list[-1].c):
            coordinator.emit(fill_ev, actor="backtest")

    coordinator.stop()

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

        eq_path = out_dir / "equity_curve.csv"
        with eq_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["ts", "close", "qty", "avg_price", "balance", "realized", "unrealized", "equity"])
            for r in equity:
                w.writerow(
                    [
                        r.ts.isoformat(),
                        str(r.close),
                        str(r.position_qty),
                        str(r.avg_price) if r.avg_price is not None else "",
                        str(r.balance),
                        str(r.realized),
                        str(r.unrealized),
                        str(r.equity),
                    ]
                )

        fills_path = out_dir / "fills.csv"
        with fills_path.open("w", newline="") as f:
            dw = csv.DictWriter(
                f,
                fieldnames=["ts", "symbol", "side", "qty", "price", "fee", "realized_pnl", "event_id", "root_event_id"],
            )
            dw.writeheader()
            for row in fills:
                dw.writerow(row)

        trades = _build_trades_from_fills(fills)

        trades_path = out_dir / "trades.csv"
        with trades_path.open("w", newline="") as f:
            fieldnames = [
                "trade_id",
                "symbol",
                "side",
                "entry_ts",
                "exit_ts",
                "qty",
                "entry_price",
                "exit_price",
                "gross_pnl",
                "fees",
                "net_pnl",
                "return",
                "duration_sec",
            ]
            dw2 = csv.DictWriter(f, fieldnames=fieldnames)
            dw2.writeheader()
            for t in trades:
                dw2.writerow(t)

        summary = _build_summary(equity=equity, trades=trades, csv_path=csv_path, symbol=symbol_u)
        summary_path = out_dir / "summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(_json_safe(summary), f, ensure_ascii=False, indent=2)

    return equity, fills


# ============================================================
# Walk-Forward Validation
# ============================================================


@dataclass(frozen=True, slots=True)
class WalkForwardWindow:
    """One window of a walk-forward test."""
    window_idx: int
    train_bars: int
    test_bars: int
    test_summary: Dict[str, Any]


def run_walk_forward(
    *,
    csv_path: Path,
    symbol: str,
    starting_balance: Decimal,
    ma_window: int,
    order_qty: Decimal,
    fee_bps: Decimal,
    slippage_bps: Decimal = Decimal("0"),
    train_size: int = 500,
    test_size: int = 100,
    out_dir: Optional[Path] = None,
) -> List[WalkForwardWindow]:
    all_bars = list(iter_ohlcv_csv(csv_path))
    if len(all_bars) < train_size + test_size:
        raise ValueError(
            f"Not enough bars for walk-forward: need {train_size + test_size}, got {len(all_bars)}"
        )

    results: List[WalkForwardWindow] = []
    window_idx = 0
    start = 0

    while start + train_size + test_size <= len(all_bars):
        test_start = start + train_size
        test_end = test_start + test_size
        test_window_bars = all_bars[start:test_end]
        test_only_bars = all_bars[test_start:test_end]

        import tempfile
        import os

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, newline=""
        ) as tmp:
            writer = csv.writer(tmp)
            writer.writerow(["ts", "open", "high", "low", "close", "volume"])
            for bar in test_window_bars:
                writer.writerow([
                    bar.ts.isoformat(),
                    str(bar.o),
                    str(bar.h),
                    str(bar.l),
                    str(bar.c),
                    str(bar.v) if bar.v is not None else "0",
                ])
            tmp_path = tmp.name

        try:
            window_out = (out_dir / f"window_{window_idx:03d}") if out_dir else None
            equity, fills = run_backtest(
                csv_path=Path(tmp_path),
                symbol=symbol,
                starting_balance=starting_balance,
                ma_window=ma_window,
                order_qty=order_qty,
                fee_bps=fee_bps,
                slippage_bps=slippage_bps,
                out_dir=window_out,
            )

            test_equity = equity[train_size:] if len(equity) > train_size else equity
            test_fills_raw = fills

            trades = _build_trades_from_fills(test_fills_raw)
            summary = _build_summary(
                equity=test_equity,
                trades=trades,
                csv_path=csv_path,
                symbol=symbol,
            )
            summary["window_idx"] = window_idx
            summary["train_bars"] = train_size
            summary["test_bars"] = len(test_only_bars)
            summary["test_start_ts"] = test_only_bars[0].ts.isoformat() if test_only_bars else ""
            summary["test_end_ts"] = test_only_bars[-1].ts.isoformat() if test_only_bars else ""

            results.append(WalkForwardWindow(
                window_idx=window_idx,
                train_bars=train_size,
                test_bars=len(test_only_bars),
                test_summary=summary,
            ))
        finally:
            try:
                os.unlink(tmp_path)
            except OSError as e:
                logger.debug("Failed to clean up temp file %s: %s", tmp_path, e)

        start += test_size
        window_idx += 1

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        wf_path = out_dir / "walk_forward_summary.json"
        with wf_path.open("w", encoding="utf-8") as f:
            json.dump(
                [_json_safe(w.test_summary) for w in results],
                f,
                ensure_ascii=False,
                indent=2,
            )

    return results


# ============================================================
# Multi-asset backtest
# ============================================================


def run_multi_backtest(
    *,
    csv_paths: Dict[str, Path],
    starting_balance: Decimal,
    fee_bps: Decimal,
    slippage_bps: Decimal = Decimal("0"),
    out_dir: Optional[Path] = None,
    embargo_bars: int = 1,
    decision_modules: Optional[List[Any]] = None,
) -> Tuple[List[EquityPoint], List[Dict[str, Any]]]:
    symbols = sorted(csv_paths.keys())
    first_symbol = symbols[0]

    # Load and tag all bars with symbol
    tagged_bars: List[Tuple[str, Any]] = []
    for sym, path in csv_paths.items():
        for bar in iter_ohlcv_csv(path):
            tagged_bars.append((sym.upper(), bar))

    tagged_bars.sort(key=lambda x: x[1].ts)

    equity: List[EquityPoint] = []
    fills: List[Dict[str, Any]] = []

    def _on_fill(ev: Any) -> None:
        ts_val = ""
        ev_ts = getattr(ev, "ts", None)
        if isinstance(ev_ts, datetime):
            ts_val = ev_ts.isoformat()
        else:
            ts_ns = getattr(getattr(ev, "header", None), "ts_ns", None)
            ts_val = str(ts_ns) if ts_ns is not None else ""

        fills.append(
            {
                "ts": ts_val,
                "symbol": getattr(ev, "symbol", None),
                "side": getattr(ev, "side", None),
                "qty": getattr(ev, "qty", None),
                "price": getattr(ev, "price", None),
                "fee": getattr(ev, "fee", None),
                "realized_pnl": getattr(ev, "realized_pnl", None),
                "event_id": getattr(getattr(ev, "header", None), "event_id", None),
                "root_event_id": getattr(getattr(ev, "header", None), "root_event_id", None),
            }
        )

    def _on_pipeline(out: Any) -> None:
        m = out.market
        a = out.account
        positions = out.positions
        ts = getattr(m, "last_ts", None)
        if ts is None:
            return
        if isinstance(ts, str):
            raw = ts.strip()
            if raw.endswith("Z"):
                raw = raw[:-1] + "+00:00"
            ts = _ensure_utc(datetime.fromisoformat(raw))

        close_f = getattr(m, "close_f", None)
        if close_f is not None:
            close_d = Decimal(str(close_f))
        else:
            close = getattr(m, "close", None) or getattr(m, "last_price", None)
            if close is None:
                return
            close_d = Decimal(str(close))

        # Sum unrealized across all positions
        total_unreal = Decimal("0")
        for sym in symbols:
            pos = positions.get(sym.upper())
            if pos is None:
                continue
            qf = getattr(pos, "qty_f", None)
            qty = Decimal(str(qf)) if qf is not None else Decimal(str(getattr(pos, "qty", 0)))
            af = getattr(pos, "avg_price_f", None)
            avg = Decimal(str(af)) if af is not None else getattr(pos, "avg_price", None)
            if qty != 0 and avg is not None:
                total_unreal += (close_d - avg) * qty

        bf = getattr(a, "balance_f", None)
        bal = Decimal(str(bf)) if bf is not None else getattr(a, "balance", Decimal("0"))
        rf = getattr(a, "realized_pnl_f", None)
        realized = Decimal(str(rf)) if rf is not None else getattr(a, "realized_pnl", Decimal("0"))
        eq = bal + total_unreal

        # Use first symbol's position for the equity point
        pos = positions.get(first_symbol.upper())
        if pos is None:
            qty = Decimal("0")
            avg = None
        else:
            qf = getattr(pos, "qty_f", None)
            qty = Decimal(str(qf)) if qf is not None else Decimal(str(getattr(pos, "qty", 0)))
            af = getattr(pos, "avg_price_f", None)
            avg = Decimal(str(af)) if af is not None else getattr(pos, "avg_price", None)

        assert isinstance(ts, datetime)
        equity.append(
            EquityPoint(
                ts=ts,
                close=close_d,
                position_qty=qty,
                avg_price=avg,
                balance=bal,
                realized=realized,
                unrealized=total_unreal,
                equity=eq,
            )
        )

    coordinator = EngineCoordinator(
        cfg=CoordinatorConfig(
            symbol_default=first_symbol.upper(),
            symbols=tuple(s.upper() for s in symbols),
            currency="USDT",
            starting_balance=float(starting_balance),
            on_pipeline_output=_on_pipeline,
        )
    )

    def _emit(ev: Any) -> None:
        coordinator.emit(ev, actor="backtest")

    # Price source returns latest close for any symbol
    _latest_prices: Dict[str, Decimal] = {}

    def _price(sym: str) -> Optional[Decimal]:
        return _latest_prices.get(sym.upper())

    def _ts() -> Optional[datetime]:
        view = coordinator.get_state_view()
        m = view.get("market")
        return getattr(m, "last_ts", None) if m is not None else None

    from execution.sim.embargo import EmbargoExecutionAdapter

    base_adapter = BacktestExecutionAdapter(
        price_source=_price,
        ts_source=_ts,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        source="paper",
        on_fill=_on_fill,
    )

    embargo_adapter = EmbargoExecutionAdapter(inner=base_adapter, embargo_bars=embargo_bars)
    exec_bridge = ExecutionBridge(adapter=embargo_adapter, dispatcher_emit=_emit)

    if decision_modules is not None:
        modules = decision_modules
    else:
        modules = []

    decision_bridge = DecisionBridge(dispatcher_emit=_emit, modules=modules)

    coordinator.attach_execution_bridge(exec_bridge)
    coordinator.attach_decision_bridge(decision_bridge)

    coordinator.start()

    for i, (sym, bar) in enumerate(tagged_bars):
        if embargo_bars > 0:
            # Fill embargoed orders at this bar's OPEN price.
            for fill_ev in embargo_adapter.on_bar(i, open_price=bar.o):
                coordinator.emit(fill_ev, actor="backtest")
        embargo_adapter.set_bar(i)

        _latest_prices[sym.upper()] = bar.c

        h = EventHeader.new_root(event_type=EventType.MARKET, version=MarketEvent.VERSION, source="csv")
        ev = MarketEvent(
            header=h,
            ts=bar.ts,
            symbol=sym.upper(),
            open=bar.o,
            high=bar.h,
            low=bar.l,
            close=bar.c,
            volume=bar.v if bar.v is not None else Decimal("0"),
        )
        coordinator.emit(ev, actor="replay")

    if embargo_bars > 0 and tagged_bars:
        # End-of-data flush for multi-symbol path.
        last_bar = tagged_bars[-1][1]
        for fill_ev in embargo_adapter.on_bar(len(tagged_bars), open_price=last_bar.c):
            coordinator.emit(fill_ev, actor="backtest")

    coordinator.stop()

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

        eq_path = out_dir / "equity_curve.csv"
        with eq_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["ts", "close", "qty", "avg_price", "balance", "realized", "unrealized", "equity"])
            for r in equity:
                w.writerow(
                    [
                        r.ts.isoformat(),
                        str(r.close),
                        str(r.position_qty),
                        str(r.avg_price) if r.avg_price is not None else "",
                        str(r.balance),
                        str(r.realized),
                        str(r.unrealized),
                        str(r.equity),
                    ]
                )

        fills_path = out_dir / "fills.csv"
        with fills_path.open("w", newline="") as f:
            dw = csv.DictWriter(
                f,
                fieldnames=["ts", "symbol", "side", "qty", "price", "fee", "realized_pnl", "event_id", "root_event_id"],
            )
            dw.writeheader()
            for row in fills:
                dw.writerow(row)

        trades = _build_trades_from_fills(fills)

        trades_path = out_dir / "trades.csv"
        with trades_path.open("w", newline="") as f:
            fieldnames = [
                "trade_id", "symbol", "side", "entry_ts", "exit_ts", "qty",
                "entry_price", "exit_price", "gross_pnl", "fees", "net_pnl",
                "return", "duration_sec",
            ]
            dw2 = csv.DictWriter(f, fieldnames=fieldnames)
            dw2.writeheader()
            for t in trades:
                dw2.writerow(t)

        summary = _build_summary(equity=equity, trades=trades, csv_path=list(csv_paths.values())[0],
            symbol=",".join(symbols))
        summary_path = out_dir / "summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(_json_safe(summary), f, ensure_ascii=False, indent=2)

    return equity, fills


# Re-export CLI functions for backward compatibility
from runner.backtest_cli import build_arg_parser, parse_args, main  # noqa: F401, E402


if __name__ == "__main__":
    main()
