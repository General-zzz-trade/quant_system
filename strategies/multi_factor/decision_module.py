from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Tuple

from event.header import EventHeader
from event.types import EventType, IntentEvent, OrderEvent
from state.position import PositionState
from runner.backtest.adapter import _make_id

from strategies.multi_factor.feature_computer import MultiFactorFeatureComputer, MultiFactorFeatures
from strategies.multi_factor.regime import Regime, classify_regime
from strategies.multi_factor.signal_combiner import CombinedSignal, combine_signals


def _snapshot_views(snapshot: Any) -> Tuple[Any, Mapping[str, Any], Optional[str], Any]:
    """Extract market, positions, event_id, account from snapshot."""
    if hasattr(snapshot, "market") and hasattr(snapshot, "positions"):
        return (
            getattr(snapshot, "market"),
            getattr(snapshot, "positions"),
            getattr(snapshot, "event_id", None),
            getattr(snapshot, "account", None),
        )
    if isinstance(snapshot, dict):
        market = snapshot.get("market")
        if market is None:
            markets = snapshot.get("markets") or {}
            market = next(iter(markets.values()), None)
        return (
            market,
            snapshot.get("positions") or {},
            snapshot.get("event_id"),
            snapshot.get("account"),
        )
    raise RuntimeError(f"unsupported snapshot: {type(snapshot).__name__}")


@dataclass
class MultiFactorConfig:
    symbol: str = "BTCUSDT"
    risk_per_trade: float = 0.02
    atr_stop_multiple: float = 3.0
    trailing_atr_multiple: float = 6.0  # wide trailing lets winners run
    max_position_pct: float = 0.80
    cooldown_bars: int = 12
    max_consecutive_losses: int = 3
    loss_reduction_factor: float = 0.5
    long_only_above_trend: bool = True  # SMA(200) trend filter + exit
    range_long_only: bool = True  # ranging: only long (oversold bounce)
    # Feature computer params
    sma_fast_window: int = 20
    sma_slow_window: int = 50
    sma_trend_window: int = 200
    rsi_window: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_window: int = 20
    bb_std: float = 2.0
    atr_window: int = 14
    atr_pct_window: int = 100
    ma_slope_window: int = 10
    # Signal thresholds
    trend_threshold: float = 0.30
    range_threshold: float = 0.99  # effectively disable ranging
    # Regime params
    atr_extreme_pct: float = 85.0
    slope_threshold: float = 0.001


class MultiFactorDecisionModule:
    """Multi-factor trend/mean-reversion hybrid strategy.

    Implements the DecisionModule protocol: decide(snapshot) -> Iterable[Event]
    """

    def __init__(self, config: Optional[MultiFactorConfig] = None) -> None:
        self.cfg = config or MultiFactorConfig()
        self.symbol = self.cfg.symbol.upper()
        self.origin = "multi_factor"

        self._fc = MultiFactorFeatureComputer(
            sma_fast_window=self.cfg.sma_fast_window,
            sma_slow_window=self.cfg.sma_slow_window,
            sma_trend_window=self.cfg.sma_trend_window,
            rsi_window=self.cfg.rsi_window,
            macd_fast=self.cfg.macd_fast,
            macd_slow=self.cfg.macd_slow,
            macd_signal=self.cfg.macd_signal,
            bb_window=self.cfg.bb_window,
            bb_std=self.cfg.bb_std,
            atr_window=self.cfg.atr_window,
            atr_pct_window=self.cfg.atr_pct_window,
            ma_slope_window=self.cfg.ma_slope_window,
        )

        # Position tracking state
        self._entry_atr: Optional[float] = None
        self._entry_regime: Optional[Regime] = None
        self._entry_price: Optional[float] = None
        self._trailing_peak: Optional[float] = None  # best price since entry
        self._cooldown: int = 0
        self._consecutive_losses: int = 0
        self._bar_count: int = 0

    def decide(self, snapshot: Any) -> Iterable[Any]:
        market, positions, event_id, account = _snapshot_views(snapshot)

        close = getattr(market, "close", None) or getattr(market, "last_price", None)
        if close is None:
            return ()

        o = float(getattr(market, "open", close) or close)
        h = float(getattr(market, "high", close) or close)
        l_ = float(getattr(market, "low", close) or close)
        c = float(close)
        vol = float(getattr(market, "volume", 0) or 0)

        features = self._fc.on_bar(open=o, high=h, low=l_, close=c, volume=vol)
        self._bar_count += 1

        # Cooldown countdown
        if self._cooldown > 0:
            self._cooldown -= 1

        # Need warmup: ATR + percentile + slope
        regime = classify_regime(
            features,
            atr_extreme_pct=self.cfg.atr_extreme_pct,
            slope_threshold=self.cfg.slope_threshold,
        )
        if regime is None:
            return ()

        pos = positions.get(self.symbol) or PositionState.empty(self.symbol)
        qty = getattr(pos, "qty", Decimal("0"))
        avg_price = getattr(pos, "avg_price", None)

        events: List[Any] = []

        if qty != 0:
            # Update trailing peak
            is_long = qty > 0
            if self._trailing_peak is None:
                self._trailing_peak = c
            elif is_long:
                self._trailing_peak = max(self._trailing_peak, c)
            else:
                self._trailing_peak = min(self._trailing_peak, c)

            # Check exits
            exit_reason = self._check_exit(features, regime, float(qty), float(avg_price) if avg_price else c)
            if exit_reason is not None:
                side = "sell" if is_long else "buy"
                abs_qty = abs(qty)
                events.extend(self._make_order_pair(
                    side=side,
                    qty=abs_qty,
                    reason_code=exit_reason,
                    event_id=event_id,
                ))
                # Track stop loss for cooldown/consecutive losses
                if exit_reason in ("stop_loss", "trailing_stop"):
                    self._cooldown = self.cfg.cooldown_bars
                    # Only count as loss if underwater
                    if avg_price is not None:
                        pnl = (c - float(avg_price)) * float(qty)
                        if pnl < 0:
                            self._consecutive_losses += 1
                        else:
                            self._consecutive_losses = 0
                    else:
                        self._consecutive_losses += 1
                else:
                    self._consecutive_losses = 0
                self._entry_atr = None
                self._entry_regime = None
                self._entry_price = None
                self._trailing_peak = None
        else:
            # Check entry
            if self._cooldown > 0:
                return ()

            signal = combine_signals(
                features,
                regime,
                trend_threshold=self.cfg.trend_threshold,
                range_threshold=self.cfg.range_threshold,
            )
            if signal.direction == 0:
                return ()

            # Long-term trend filter: no shorts above SMA(200)
            if self.cfg.long_only_above_trend and features.sma_trend is not None:
                if signal.direction == -1 and c > features.sma_trend:
                    return ()
                # Also: no longs below SMA(200) for ranging (mean-reversion longs need structural support)
                # Keep trend longs below SMA(200) — they can catch reversals

            # Ranging: long-only
            if self.cfg.range_long_only and regime == Regime.RANGING and signal.direction == -1:
                return ()

            if features.atr is None or features.atr <= 0:
                return ()

            equity = self._get_equity(account, positions, c)
            entry_qty = self._compute_qty(equity, c, features.atr, signal.direction)
            if entry_qty <= Decimal("0"):
                return ()

            side = "buy" if signal.direction > 0 else "sell"
            reason = f"{regime.value}_{side}"
            events.extend(self._make_order_pair(
                side=side,
                qty=entry_qty,
                reason_code=reason,
                event_id=event_id,
            ))
            self._entry_atr = features.atr
            self._entry_regime = regime
            self._entry_price = c
            self._trailing_peak = c

        return events

    def _get_equity(self, account: Any, positions: Mapping[str, Any], current_price: float) -> float:
        bal = float(getattr(account, "balance", 10000) if account else 10000)
        pos = positions.get(self.symbol) or PositionState.empty(self.symbol)
        qty = float(getattr(pos, "qty", 0))
        avg = getattr(pos, "avg_price", None)
        unreal = 0.0
        if qty != 0 and avg is not None:
            unreal = (current_price - float(avg)) * qty
        return bal + unreal

    def _compute_qty(self, equity: float, price: float, atr: float, direction: int) -> Decimal:
        risk_budget = equity * self.cfg.risk_per_trade

        # Reduce size after consecutive losses
        if self._consecutive_losses >= self.cfg.max_consecutive_losses:
            risk_budget *= self.cfg.loss_reduction_factor

        stop_dist = atr * self.cfg.atr_stop_multiple
        if stop_dist <= 0:
            return Decimal("0")

        raw_qty = risk_budget / stop_dist
        max_qty = self.cfg.max_position_pct * equity / price if price > 0 else 0

        qty = min(raw_qty, max_qty)
        if qty <= 0:
            return Decimal("0")

        # Round to 5 decimal places (BTC precision)
        return Decimal(str(round(qty, 5)))

    def _check_exit(
        self,
        features: MultiFactorFeatures,
        regime: Regime,
        qty: float,
        avg_price: float,
    ) -> Optional[str]:
        close = features.close
        is_long = qty > 0

        # Hard stop loss: entry_atr * atr_stop_multiple from entry
        if self._entry_atr is not None and self._entry_price is not None:
            stop_dist = self._entry_atr * self.cfg.atr_stop_multiple
            if is_long and close <= self._entry_price - stop_dist:
                return "stop_loss"
            if not is_long and close >= self._entry_price + stop_dist:
                return "stop_loss"

        # SMA(200) trend cross: exit longs below SMA(200), shorts above
        if self.cfg.long_only_above_trend and features.sma_trend is not None:
            if is_long and close < features.sma_trend:
                return "trend_cross"
            if not is_long and close > features.sma_trend:
                return "trend_cross"

        # Trailing stop: trailing_atr_multiple * current ATR from peak
        if self._trailing_peak is not None and features.atr is not None and features.atr > 0:
            trail_dist = features.atr * self.cfg.trailing_atr_multiple
            if is_long and close <= self._trailing_peak - trail_dist:
                return "trailing_stop"
            if not is_long and close >= self._trailing_peak + trail_dist:
                return "trailing_stop"

        # Mean reversion: exit at BB middle (only for ranging entries)
        if self._entry_regime == Regime.RANGING:
            if features.bb_middle is not None:
                if is_long and close >= features.bb_middle:
                    return "target_reached"
                if not is_long and close <= features.bb_middle:
                    return "target_reached"

        return None

    def _make_order_pair(
        self,
        *,
        side: str,
        qty: Decimal,
        reason_code: str,
        event_id: Optional[str],
    ) -> Sequence[Any]:
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
                side=side,
                target_qty=qty,
                reason_code=reason_code,
                origin=self.origin,
            ),
            OrderEvent(
                header=order_h,
                order_id=order_id,
                intent_id=intent_id,
                symbol=self.symbol,
                side=side,
                qty=qty,
                price=None,
            ),
        )
