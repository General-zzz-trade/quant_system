"""AlphaDecisionModule — framework-native alpha decision engine.

Replaces the 2571-line AlphaRunner god class with a composable
DecisionModule that reads an immutable StateSnapshot and emits
OrderEvents.  Pure decision logic — no venue state, no I/O.
"""
from __future__ import annotations

import json
import logging
import os
import time
from decimal import Decimal
from typing import Any, Iterable

import numpy as np

from decision.signals.alpha_signal import EnsemblePredictor, SignalDiscretizer
from decision.sizing.adaptive import AdaptivePositionSizer
from monitoring.decision_audit import DecisionAuditLogger
from decision.modules.alpha_orders import make_open_order, make_close_order
from event.header import EventHeader
from event.types import EventType, OrderEvent, RiskEvent, SignalEvent
from state import PortfolioState, RiskState, RiskLimits
from _quant_hotpath import (  # type: ignore[import-untyped]
    RustRegimeParams,
    RustRidgePredictor,
)

# Type aliases for Rust-accelerated components used in framework-native path
RegimeParamsType = RustRegimeParams
RidgePredictorType = RustRidgePredictor

logger = logging.getLogger(__name__)

# IC health status → scale multiplier
_IC_SCALE_MAP: dict[str, float] = {
    "GREEN": 1.2,
    "YELLOW": 0.8,
    "RED": 0.4,
}

# Runner key → model name for IC health lookup
_RUNNER_MODEL_MAP: dict[str, str] = {
    "BTCUSDT": "BTCUSDT_gate_v2",
    "ETHUSDT": "ETHUSDT_gate_v2",
    "BTCUSDT_4h": "BTCUSDT_4h",
    "ETHUSDT_4h": "ETHUSDT_4h",
}

_IC_HEALTH_PATH = "data/runtime/ic_health.json"
_IC_REFRESH_SECS = 600  # 10 minutes


class AlphaDecisionModule:
    """Framework-native alpha decision module.

    Implements the ``DecisionModule`` protocol::

        def decide(self, snapshot) -> Iterable[OrderEvent]

    Composes EnsemblePredictor, SignalDiscretizer, and
    AdaptivePositionSizer into a stateless-friendly pipeline.
    """

    def __init__(
        self,
        symbol: str,
        runner_key: str,
        predictor: EnsemblePredictor,
        discretizer: SignalDiscretizer,
        sizer: AdaptivePositionSizer,
        leverage: float = 10.0,
    ) -> None:
        self._symbol = symbol
        self._runner_key = runner_key
        self._predictor = predictor
        self._discretizer = discretizer
        self._sizer = sizer
        self._leverage = leverage

        # Pure decision state
        self._signal: int = 0
        self._current_qty: Decimal = Decimal("0")
        self._entry_price: float = 0.0
        self._trade_peak: float = 0.0
        self._bars_processed: int = 0
        self._last_trade_bar: int = -9999  # cooldown: bar index of last trade (init allows first trade)

        # Regime filter buffers
        self._closes: list[float] = []
        self._ema10: float = 0.0  # EMA(10) for trend filter (updated each bar)
        self._rets: list[float] = []
        self._vol_history: list[float] = []
        self._trend_history: list[float] = []
        self._regime_active: bool = True
        self._trend_factor: float = 1.0

        # Stop-loss
        self._atr_buffer: list[float] = []
        self._last_stop_bar: int = -9999  # bar index of last forced exit
        self._last_stop_direction: int = 0  # direction of last stopped position
        self._stop_cooldown_bars: int = 6  # bars to wait after stop before same-dir entry
        self._consecutive_stops: int = 0   # count of consecutive forced exits
        self._stop_pause_until: int = 0    # bar index until which trading is paused

        # Graduated entry: position size scales continuously with |z|.
        # Replaces the old binary deadzone + tier1/tier2 system.
        # Soft deadzone: below _soft_dz_floor → 0%; above → sigmoid ramp to 100%.
        # Data shows edge is continuous (z=1.8 ~ z=2.0), not a cliff at deadzone.
        # Floor optimized per-symbol via full-sample sweep:
        #   BTC (dz=2.0): floor=0.7×dz=1.4 → Sharpe 2.89 (vs 2.75 hard)
        #   ETH (dz=1.5): floor=0.4×dz=0.6 → Sharpe 3.11 (vs 2.92 hard)
        if "ETH" in symbol:
            floor_ratio = 0.4
        else:
            floor_ratio = 0.7
        self._soft_dz_floor: float = discretizer.deadzone * floor_ratio

        # Cross-symbol consensus
        self._consensus: dict[str, int] = {}

        # Timeframe detection
        self._is_4h: bool = "4h" in runner_key
        self._ma_window: int = 120 if self._is_4h else 480
        self._adaptive_window: int = 200

        # Vol baseline
        self._vol_median: float = 0.013 if self._is_4h else 0.0063

        # IC health cache
        self._ic_scale: float = 1.0
        self._ic_cache_ts: float = 0.0

        # Adaptive parameter bases (vol-scaled at runtime)
        self._deadzone_base: float = discretizer.deadzone
        self._min_hold_base: int = discretizer.min_hold
        self._max_hold_base: int = discretizer.max_hold

        # Microstructure VPIN scaling (optional, live-only)
        self._vpin_caution_thresh: float = 0.5
        self._vpin_scale_factor: float = 0.7  # reduce size by 30% when VPIN > threshold

        # Batch prediction override: when set, decide() uses this instead of
        # incremental predictor.predict(). Cleared after each use.
        # Prevents incremental features from contaminating the z-score buffer.
        self._batch_pred_override: float | None = None

        # Decision audit logger (best-effort, never affects trading)
        self._audit = DecisionAuditLogger()
        self._audit_enabled = True  # disabled during warmup to prevent fake entries

    def set_consensus(self, signals: dict[str, int]) -> None:
        """Update cross-symbol consensus signals."""
        self._consensus.update(signals)

    def update_predictor(self, predictor: EnsemblePredictor) -> None:
        """Hot-swap the ensemble predictor (SIGHUP reload)."""
        self._predictor = predictor

    def decide(self, snapshot: Any) -> Iterable[RiskEvent | SignalEvent | OrderEvent]:
        """Read-only snapshot → opinion events.  No side effects on venue."""
        self._bars_processed += 1
        mkt = snapshot.markets[self._symbol]
        # Prefer close_f (float) over close (may be Fd8 i64 on Rust types)
        _cf = getattr(mkt, "close_f", None)
        if isinstance(_cf, (int, float)) and _cf > 0:
            close = float(_cf)
        else:
            # Fallback: raw .close may be Fd8 i64 (×10^8)
            raw = float(mkt.close)
            close = raw / 100_000_000 if raw > 1_000_000 else raw

        # Guard: skip duplicate bar timestamps (same bar replayed → no new information)
        _mkt_obj = snapshot.markets.get(self._symbol) if isinstance(snapshot.markets, dict) else None
        bar_ts = getattr(_mkt_obj, 'last_ts', None) if _mkt_obj is not None else None
        if isinstance(bar_ts, (int, float)) and bar_ts > 0:
            if hasattr(self, '_last_bar_ts') and bar_ts == self._last_bar_ts:
                return ()
            self._last_bar_ts = bar_ts
        features: dict = dict(snapshot.features) if snapshot.features else {}
        self._last_features = features  # cache for intra-bar preview

        # 0. Portfolio exposure + risk limit checks
        portfolio: PortfolioState | None = getattr(snapshot, "portfolio", None)
        risk: RiskState | None = getattr(snapshot, "risk", None)
        if portfolio is not None:
            leverage = float(getattr(portfolio, "leverage", 0) or 0)
            if leverage > 5.0:
                logger.warning(
                    "%s high portfolio leverage: %.1fx", self._runner_key, leverage,
                )
        if risk is not None:
            margin_used = float(getattr(risk, "margin_used_pct", 0) or 0)
            if margin_used > 0.8:
                logger.warning(
                    "%s high margin usage: %.0f%%", self._runner_key, margin_used * 100,
                )

        # RiskLimits available for downstream gate checks if needed
        _ = RiskLimits  # ensure wired

        # 0b. Online Ridge update: feed realized return from previous bar
        if len(self._closes) >= 2:
            try:
                prev_close = self._closes[-1]  # before appending current close
                if prev_close > 0:
                    realized_ret = np.log(close / prev_close)
                    self._predictor.update_online_ridge(realized_ret)
            except Exception:
                pass  # never crash the trading loop

        # 1. Regime filter
        regime_ok = self._check_regime(close)

        # 2. Update ATR
        self._update_atr(snapshot)

        # 3. Predict — prefer batch override to avoid incremental divergence
        if self._batch_pred_override is not None:
            pred = self._batch_pred_override
            self._batch_pred_override = None
        else:
            pred = self._predictor.predict(features)
        if pred is None:
            return ()

        # 4. Discretize
        new_signal, z = self._discretizer.discretize(
            pred,
            self._bars_processed,
            regime_ok,
            current_signal=self._signal,
        )

        # 4b. Graduated entry: soft deadzone replaces binary tier1/tier2.
        # If z exceeds the soft floor (0.4×dz) but is below hard deadzone,
        # enter with a fraction proportional to signal strength.
        sw = self._signal_weight(z)
        if new_signal == 0 and self._signal == 0 and sw > 0.02:
            # Signal weight is meaningful — enter with graduated size
            new_signal = 1 if z > 0 else -1

        # 5. Force exits
        force_exit, exit_reason = self._check_force_exits(close, z)
        if force_exit:
            new_signal = 0

        # Audit: log every signal evaluation (best-effort, skipped during warmup)
        if self._audit_enabled:
            try:
                self._audit.log_signal(
                    symbol=self._symbol, runner_key=self._runner_key,
                    z_score=z, signal=new_signal, confidence=abs(z),
                    features=features, force_exit=exit_reason or None,
                )
            except Exception:
                pass

        # Pre-allocate events list for risk/signal events + orders
        events: list[RiskEvent | SignalEvent | OrderEvent] = []

        # Emit RiskEvent on force exit
        if force_exit:
            risk_header = EventHeader.new_root(
                event_type=EventType.RISK,
                version=1,
                source=f"alpha.{self._runner_key}.risk",
            )
            events.append(RiskEvent(
                header=risk_header,
                rule_id=exit_reason,
                level="block",
                message=f"{self._symbol} force exit: {exit_reason}",
            ))

        # 6. Direction alignment (ETH follows BTC)
        if (
            new_signal != 0
            and self._signal == 0
            and "ETH" in self._symbol
        ):
            btc_keys = [k for k in self._consensus if "BTC" in k]
            if btc_keys:
                btc_dir = self._consensus.get(btc_keys[0], 0)
                if btc_dir != 0 and btc_dir != new_signal:
                    logger.info(
                        "%s direction alignment: blocked %+d (BTC=%+d)",
                        self._symbol, new_signal, btc_dir,
                    )
                    # Emit RiskEvent for direction alignment block
                    align_header = EventHeader.new_root(
                        event_type=EventType.RISK,
                        version=1,
                        source=f"alpha.{self._runner_key}.risk",
                    )
                    events.append(RiskEvent(
                        header=align_header,
                        rule_id="direction_alignment",
                        level="block",
                        message=f"{self._symbol} ETH blocked: opposing BTC ({new_signal:+d} vs {btc_dir:+d})",
                    ))
                    new_signal = 0

        # 6b. 4h direction filter (1h entries only)
        # If 4h signal opposes a new 1h entry, block the entry.
        # Strong 4h conviction should prevent counter-trend 1h trades.
        if (
            new_signal != 0
            and self._signal == 0
            and not self._is_4h
        ):
            base_sym = self._symbol.replace("_15m", "")
            tf4h_key = f"{base_sym}_4h"
            tf4h_signal = self._consensus.get(tf4h_key, 0)
            if tf4h_signal != 0 and tf4h_signal != new_signal:
                logger.info(
                    "%s 4h direction filter: blocked %+d entry (4h=%+d)",
                    self._runner_key, new_signal, tf4h_signal,
                )
                try:
                    align_header = EventHeader.new_root(
                        event_type=EventType.RISK,
                        version=1,
                        source=f"alpha.{self._runner_key}.risk",
                    )
                    events.append(RiskEvent(
                        header=align_header,
                        rule_id="4h_direction_filter",
                        level="block",
                        message=f"{self._symbol} 1h blocked: 4h opposes ({new_signal:+d} vs {tf4h_signal:+d})",
                    ))
                except Exception:
                    pass
                new_signal = 0

        # 6c. Multi-trend filter: scale signal_weight by trend alignment.
        # Replaces the old EMA10 hard block which prevented 48% of trades
        # that had HIGHER win rates than those it allowed through.
        # Now: counter-trend → reduce sw; aligned → boost sw.
        if new_signal != 0 and len(self._closes) >= 20:
            ema = self._ema10
            ema_dev = (close - ema) / ema if ema > 0 else 0.0
            # alignment: positive = signal agrees with EMA direction
            alignment = ema_dev * new_signal
            ema_score = np.clip(0.85 + alignment * 15, 0.5, 1.2)

            # 20-bar momentum alignment
            if len(self._closes) >= 21:
                ret20 = close / self._closes[-21] - 1
                mom_align = ret20 * new_signal
                mom_score = np.clip(0.85 + mom_align * 10, 0.6, 1.2)
            else:
                mom_score = 1.0

            self._trend_factor = float(np.clip(
                np.sqrt(ema_score * mom_score), 0.4, 1.3,
            ))
        else:
            self._trend_factor = 1.0

        # Apply trend factor to signal weight — counter-trend reduces,
        # aligned boosts.  This replaces the old hard EMA10 block.
        sw = sw * self._trend_factor
        if sw < 0.02 and new_signal != 0 and self._signal == 0:
            logger.info(
                "%s trend scaled out: z=%+.2f, sw_raw=%.0f%%, tf=%.2f → sw=%.1f%%",
                self._runner_key, z,
                self._signal_weight(z) * 100, self._trend_factor, sw * 100,
            )
            new_signal = 0

        # 6d. Consecutive stop pause: after 3+ consecutive stops, pause trading.
        if new_signal != 0 and self._signal == 0 and self._bars_processed < self._stop_pause_until:
            logger.info(
                "%s stop pause active: blocked until bar %d (current %d, %d consecutive stops)",
                self._runner_key, self._stop_pause_until,
                self._bars_processed, self._consecutive_stops,
            )
            new_signal = 0

        # 6e. Stop-loss cooldown: after forced exit, wait extra bars
        # Prevents "stop → re-enter same direction → stop again" loops.
        if new_signal != 0 and self._signal == 0:
            bars_since_stop = self._bars_processed - self._last_stop_bar
            if (
                bars_since_stop < self._stop_cooldown_bars
                and new_signal == self._last_stop_direction
            ):
                logger.info(
                    "%s stop cooldown: %+d blocked (%d/%d bars since stop)",
                    self._symbol, new_signal, bars_since_stop,
                    self._stop_cooldown_bars,
                )
                new_signal = 0

        # 7. Trade cooldown: prevent rapid-fire flat→entry cycles
        # After closing a position, wait min_hold bars before opening a new one.
        # This matches the Rust backtest behavior and prevents warmup-induced churn.
        # Note: signal FLIPS (long→short) are allowed — cooldown only gates flat→entry.
        if new_signal != 0 and self._signal == 0 and not force_exit:
            bars_since_last = self._bars_processed - self._last_trade_bar
            try:
                min_hold = int(self._discretizer.min_hold)
            except (TypeError, ValueError):
                min_hold = 6
            if bars_since_last < min_hold:
                new_signal = 0  # too soon after last trade, stay flat

        # 7b. Graduated scale-up: if already in position and z strengthened,
        # increase position toward target = base_qty × signal_weight(z).
        if (
            self._signal != 0
            and new_signal == self._signal  # same direction
            and sw > 0
        ):
            self._refresh_ic_scale()
            full_qty = self._sizer.target_qty(
                snapshot, self._symbol,
                leverage=self._leverage,
                ic_scale=self._ic_scale,
                regime_active=self._regime_active,
                z_scale=1.0,
            )
            try:
                _rounded = self._sizer._round_to_step(float(full_qty) * sw)
                target_qty = _rounded if isinstance(_rounded, Decimal) else Decimal(str(float(full_qty) * sw))
            except Exception:
                target_qty = Decimal(str(float(full_qty) * sw))
            add_qty = target_qty - self._current_qty
            try:
                _rounded = self._sizer._round_to_step(float(add_qty))
                if isinstance(_rounded, Decimal):
                    add_qty = _rounded
            except Exception:
                pass
            if add_qty > Decimal("0") and float(add_qty) > float(full_qty) * 0.05:
                events.extend(self._make_open_order(close, self._signal, add_qty))
                avg_entry = (
                    (self._entry_price * float(self._current_qty) + close * float(add_qty))
                    / (float(self._current_qty) + float(add_qty))
                )
                self._entry_price = avg_entry
                self._current_qty += add_qty
                logger.info(
                    "%s SCALE-UP: z=%+.2f sw=%.0f%%, added %.4f → total %.4f",
                    self._runner_key, z, sw * 100,
                    float(add_qty), float(self._current_qty),
                )

        # Reset entry_tier on exit
        if new_signal == 0 and self._signal != 0:
            self._entry_tier = 0

        # 8. Emit events on signal change
        if new_signal != self._signal:
            old_signal = self._signal
            self._last_trade_bar = self._bars_processed

            # Emit SignalEvent for signal transition
            signal_header = EventHeader.new_root(
                event_type=EventType.SIGNAL,
                version=1,
                source=f"alpha.{self._runner_key}",
            )
            if new_signal != 0:
                side = "long" if new_signal > 0 else "short"
            else:
                side = "flat"
            events.append(SignalEvent(
                header=signal_header,
                signal_id=signal_header.event_id,
                symbol=self._symbol,
                side=side,
                strength=Decimal(str(abs(z))),
            ))

            # Close existing position
            if old_signal != 0:
                reason = exit_reason if force_exit else "signal_change"
                if self._audit_enabled:
                    try:
                        self._audit.log_exit(
                            symbol=self._symbol,
                            side="sell" if old_signal == 1 else "buy",
                            qty=float(self._current_qty), price=close, reason=reason,
                            entry_price=self._entry_price,
                        )
                    except Exception:
                        pass
                events.extend(self._make_close_order(close, old_signal, reason))
                self._current_qty = Decimal("0")
                # Record stop for cooldown (prevents same-direction re-entry)
                if force_exit:
                    self._last_stop_bar = self._bars_processed
                    self._last_stop_direction = old_signal
                    self._consecutive_stops += 1
                    # After 3 consecutive stops, pause trading for 12 bars
                    if self._consecutive_stops >= 3:
                        self._stop_pause_until = self._bars_processed + 12
                        logger.warning(
                            "%s STOP PAUSE: %d consecutive stops → paused until bar %d",
                            self._runner_key, self._consecutive_stops,
                            self._stop_pause_until,
                        )
                else:
                    # Normal exit (signal_change) resets counter
                    self._consecutive_stops = 0

            # Open new position — size proportional to signal_weight
            if new_signal != 0:
                self._refresh_ic_scale()
                qty = self._sizer.target_qty(
                    snapshot,
                    self._symbol,
                    leverage=self._leverage,
                    ic_scale=self._ic_scale,
                    regime_active=self._regime_active,
                    z_scale=1.0,
                )
                # Graduated sizing: qty × signal_weight(z)
                qty = Decimal(str(float(qty) * sw))
                if sw < 0.95:
                    logger.info(
                        "%s GRADUATED entry: z=%+.2f, sw=%.0f%%, qty=%.4f",
                        self._runner_key, z, sw * 100, float(qty),
                    )
                # 4h consensus boost: if 4h signal agrees, scale up position
                # ETH 1h+4h同向 → 仓位 ×1.25 (backtest: +25% return improvement)
                if not self._is_4h and "ETH" in self._symbol:
                    tf4h_key = f"{self._symbol}_4h"
                    tf4h_signal = self._consensus.get(tf4h_key, 0)
                    if tf4h_signal != 0 and tf4h_signal == new_signal:
                        qty = Decimal(str(float(qty) * 1.25))
                        logger.info(
                            "%s 4h consensus boost: %+d agrees, size ×1.25",
                            self._runner_key, tf4h_signal,
                        )

                # VPIN-based size reduction: if microstructure data shows
                # high toxicity, reduce position size (optional, live-only)
                vpin = features.get("vpin")
                if vpin is not None and vpin > self._vpin_caution_thresh:
                    qty = Decimal(str(float(qty) * self._vpin_scale_factor))
                    logger.info(
                        "%s VPIN=%.3f > %.2f — size reduced to %.4f (×%.1f)",
                        self._runner_key, vpin, self._vpin_caution_thresh,
                        qty, self._vpin_scale_factor,
                    )
                # Re-apply lot-size rounding after all multiplications
                # (sw, 4h boost, VPIN all introduce float precision errors)
                try:
                    _rounded = self._sizer._round_to_step(float(qty))
                    if isinstance(_rounded, Decimal):
                        qty = _rounded
                except Exception:
                    pass
                if qty <= 0:
                    return events  # skip zero/negative qty (warmup, edge case)
                events.extend(self._make_open_order(close, new_signal, qty))
                entry_reason = "graduated" if sw < 0.95 else "signal"
                if self._audit_enabled:
                    try:
                        self._audit.log_entry(
                            symbol=self._symbol,
                            side="buy" if new_signal == 1 else "sell",
                            qty=float(qty), price=close,
                            reason=entry_reason, z_score=z, ic_scale=self._ic_scale,
                        )
                    except Exception:
                        pass
                self._entry_price = close
                self._trade_peak = close
                self._current_qty = qty
            else:
                self._entry_price = 0.0
                self._trade_peak = 0.0

            self._signal = new_signal

        # 8. Update consensus
        self._consensus[self._runner_key] = self._signal

        return events

    def _check_regime(self, close: float) -> bool:
        """Adaptive p20/p25 percentile regime filter."""
        self._closes.append(close)
        # Update EMA10 for trend filter (alpha = 2/(10+1) ≈ 0.1818)
        if self._ema10 <= 0:
            self._ema10 = close
        else:
            self._ema10 = close * 0.1818 + self._ema10 * 0.8182
        if len(self._closes) >= 2:
            log_ret = np.log(self._closes[-1] / self._closes[-2])
            self._rets.append(log_ret)

        max_buf = self._ma_window + 100
        if len(self._closes) > max_buf:
            self._closes = self._closes[-max_buf:]
        if len(self._rets) > max_buf:
            self._rets = self._rets[-max_buf:]

        if len(self._rets) < 20:
            self._regime_active = True
            return True

        vol_20 = float(np.std(self._rets[-20:]))
        ma_vals = self._closes[-self._ma_window:]
        ma = np.mean(ma_vals)
        trend = abs(close / ma - 1.0)

        self._vol_history.append(vol_20)
        self._trend_history.append(trend)

        if len(self._vol_history) > self._adaptive_window:
            self._vol_history = self._vol_history[-self._adaptive_window:]
        if len(self._trend_history) > self._adaptive_window:
            self._trend_history = self._trend_history[-self._adaptive_window:]

        if len(self._vol_history) >= 50:
            vol_thresh = float(np.percentile(self._vol_history, 20))
            trend_thresh = float(np.percentile(self._trend_history, 20))
            self._regime_active = vol_20 > vol_thresh or trend > trend_thresh
        else:
            self._regime_active = True

        # Fixed deadzone/hold — backtest shows fixed outperforms vol-adaptive
        # (vol-adaptive reduced Return by 43% due to widening dz on strong signals)
        self._discretizer.deadzone = self._deadzone_base
        self._discretizer.min_hold = self._min_hold_base
        self._discretizer.max_hold = self._max_hold_base

        return self._regime_active

    def _update_atr(self, snapshot: Any) -> None:
        """Update ATR buffer from OHLC data."""
        if len(self._closes) < 2:
            return
        mkt = snapshot.markets[self._symbol]
        _hf = getattr(mkt, "high_f", None)
        high = float(_hf) if isinstance(_hf, (int, float)) and _hf > 0 else float(mkt.high)
        if high > 1_000_000:
            high /= 100_000_000
        _lf = getattr(mkt, "low_f", None)
        low = float(_lf) if isinstance(_lf, (int, float)) and _lf > 0 else float(mkt.low)
        if low > 1_000_000:
            low /= 100_000_000
        prev_close = self._closes[-2]
        close = self._closes[-1]

        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        atr_pct = tr / close if close > 0 else 0.0
        self._atr_buffer.append(atr_pct)

        if len(self._atr_buffer) > 50:
            self._atr_buffer = self._atr_buffer[-50:]

    def _current_atr(self) -> float:
        """Mean ATR over last 14 bars, fallback 0.015."""
        if not self._atr_buffer:
            return 0.015
        window = self._atr_buffer[-14:]
        return float(np.mean(window))

    def _vol_factor(self) -> float:
        """Compute vol regime factor: current vol / median vol.

        Returns a clamped ratio [0.5, 2.0] that scales exit parameters:
          < 1.0 → low-vol: tighten stops, take profit faster
          = 1.0 → normal: baseline parameters
          > 1.0 → high-vol: widen stops, give trends room
        IC-health modulation: when IC is RED (scale≤0.4), force vf toward
        the conservative (low-vol) side to protect capital.
        """
        if len(self._vol_history) < 20:
            return 1.0
        vol_20 = self._vol_history[-1] if self._vol_history else self._vol_median
        vf = vol_20 / self._vol_median if self._vol_median > 0 else 1.0
        vf = np.clip(vf, 0.5, 2.0)
        # IC-health modulation: poor model → tighter exits
        if self._ic_scale <= 0.4:
            vf = min(vf, 0.7)
        return float(vf)

    def _check_force_exits(self, close: float, z: float) -> tuple[bool, str]:
        """Check for forced exit conditions.  Priority order.

        All thresholds scale with vol_factor (vf):
          low-vol  → tighter stops, faster profit-taking
          high-vol → wider stops, let trends run
        """
        if self._signal == 0 or self._entry_price <= 0:
            return False, ""

        atr = self._current_atr()
        vf = self._vol_factor()

        # Update trade peak
        if self._signal == 1:
            self._trade_peak = max(self._trade_peak, close)
        else:
            self._trade_peak = min(self._trade_peak, close)

        # ATR 3-phase trailing stop
        if self._signal == 1:
            profit_pct = (self._trade_peak / self._entry_price) - 1.0
            drawdown_pct = (self._trade_peak - close) / self._trade_peak
        else:
            profit_pct = 1.0 - (self._trade_peak / self._entry_price)
            drawdown_pct = (close - self._trade_peak) / self._trade_peak if self._trade_peak > 0 else 0.0

        # Phase selection — multipliers scale with vf:
        #   vf=0.5 (low-vol):  trail=0.15, brkev=0.08, init=0.9
        #   vf=1.0 (baseline): trail=0.20, brkev=0.10, init=1.2
        #   vf=2.0 (high-vol): trail=0.30, brkev=0.15, init=1.6
        trail_mult = 0.1 + 0.1 * vf     # 0.15 – 0.30
        brkev_mult = 0.05 + 0.05 * vf   # 0.075 – 0.15
        init_mult = 0.6 + 0.6 * vf      # 0.9 – 1.8
        floor = 0.001 + 0.002 * vf       # 0.002 – 0.005

        if profit_pct >= 1.0 * atr:
            stop_dist = atr * trail_mult
        elif profit_pct >= 0.5 * atr:
            stop_dist = atr * brkev_mult
        else:
            stop_dist = atr * init_mult

        # Hard floor/ceiling
        stop_dist = np.clip(stop_dist, floor, 0.05)

        # Profit-lock: once profit > 2×ATR, cap giveback at 50% of peak
        # profit. Use the TIGHTER of ATR trailing and profit-lock.
        exit_reason = "atr_stop"
        if profit_pct >= 2.0 * atr:
            max_giveback = profit_pct * 0.5
            if max_giveback < stop_dist:
                stop_dist = max_giveback
                exit_reason = "profit_lock"

        if drawdown_pct > stop_dist:
            return True, f"{exit_reason}({drawdown_pct:.3f}>{stop_dist:.3f})"

        # Quick loss: ATR-based adverse move from entry
        # vf scales ceiling: low-vol → tighter cap, high-vol → more room
        if self._signal == 1:
            adverse = (self._entry_price - close) / self._entry_price
        else:
            adverse = (close - self._entry_price) / self._entry_price

        ql_threshold = max(2.0 * atr, 0.003)
        ql_ceiling = 0.01 + 0.01 * vf  # 0.015 – 0.03 (was fixed 0.02)
        ql_threshold = min(ql_threshold, ql_ceiling)
        if adverse > ql_threshold:
            return True, f"quick_loss({adverse:.3f}>{ql_threshold:.3f})"

        # Z-fade profit exit: signal weakening while in profit.
        # Low-vol: exit at 0.6×dz (earlier take-profit)
        # High-vol: exit at 0.4×dz (let winners run)
        zfade_ratio = 0.7 - 0.15 * vf   # 0.55 – 0.40
        if profit_pct > atr and abs(z) < zfade_ratio * self._deadzone_base:
            return True, (
                f"z_fade_tp(z={z:+.2f},thresh={zfade_ratio*self._deadzone_base:.1f},"
                f"profit={profit_pct:.3f},vf={vf:.2f})"
            )

        # Z reversal
        if self._signal == 1 and z < -0.3:
            return True, f"z_reversal(long,z={z:.2f})"
        if self._signal == -1 and z > 0.3:
            return True, f"z_reversal(short,z={z:.2f})"

        # 4h reversal (non-4h runners only)
        if not self._is_4h:
            for k, v in self._consensus.items():
                if "4h" in k and self._symbol.replace("_4h", "").replace("_15m", "") in k:
                    if v != 0 and v != self._signal:
                        return True, f"4h_reversal({k}={v})"

        # Direction alignment exit (ETH follows BTC)
        if "ETH" in self._symbol and not self._is_4h:
            btc_key = self._symbol.replace("ETH", "BTC")
            btc_signal = self._consensus.get(btc_key, 0)
            if btc_signal != 0 and btc_signal != self._signal:
                return True, f"alignment_exit(eth={self._signal},btc={btc_signal})"

        # Max hold: scaled by vf — high-vol gets more time.
        try:
            max_hold = int(self._max_hold_base * vf)
        except (TypeError, ValueError):
            max_hold = 0
        if max_hold > 0:
            bars_held = self._bars_processed - self._last_trade_bar
            if bars_held >= max_hold:
                return True, f"max_hold({bars_held}>={max_hold},vf={vf:.2f})"

        return False, ""

    def _signal_weight(self, z: float) -> float:
        """Map |z| to position fraction [0, 1] via smooth sigmoid.

        Replaces binary deadzone + tier1/tier2 with a continuous curve:
          |z| < floor (0.4×dz):  → 0%   (noise, no position)
          |z| = dz:              → ~60%  (confirmed signal)
          |z| = 1.5×dz:         → ~95%  (strong signal)
          |z| > 1.5×dz:         → 100%  (max conviction)

        The sigmoid midpoint is at dz (original deadzone), steepness tuned
        so the ramp starts at ~0.4×dz and saturates at ~1.5×dz.
        """
        abs_z = abs(z)
        if abs_z < self._soft_dz_floor:
            return 0.0
        # Sigmoid: 1 / (1 + exp(-k*(x - mid)))
        # mid = deadzone, k chosen so weight(0.4*dz) ≈ 0.02, weight(1.5*dz) ≈ 0.95
        dz = self._deadzone_base
        if dz <= 0:
            return 1.0 if abs_z > 0.5 else 0.0
        k = 4.0 / dz  # steepness: 4/dz gives good ramp shape
        w = 1.0 / (1.0 + np.exp(-k * (abs_z - dz)))
        return float(min(w, 1.0))

    def _refresh_ic_scale(self) -> None:
        """Read IC health JSON every 10 minutes."""
        now = time.time()
        if now - self._ic_cache_ts < _IC_REFRESH_SECS:
            return
        self._ic_cache_ts = now

        model_name = _RUNNER_MODEL_MAP.get(self._runner_key, self._runner_key)
        try:
            if not os.path.exists(_IC_HEALTH_PATH):
                return
            # Stale file detection: if >2h old, degrade to YELLOW
            file_age = now - os.path.getmtime(_IC_HEALTH_PATH)
            if file_age > 7200:  # 2 hours
                logger.warning("IC health file stale (%.0fs old), degrading to YELLOW", file_age)
                self._ic_scale = _IC_SCALE_MAP.get("YELLOW", 0.8)
                return
            with open(_IC_HEALTH_PATH) as f:
                data = json.load(f)
            status = data.get(model_name, {}).get("status", "GREEN")
            self._ic_scale = _IC_SCALE_MAP.get(status, 1.0)
        except Exception:
            logger.debug("IC health read failed, keeping scale=%.1f", self._ic_scale)

    # ── event factories (delegated to alpha_orders module) ──────

    def _make_open_order(
        self, price: float, signal: int, qty: Decimal,
    ) -> list[OrderEvent]:
        """Create OrderEvent for opening a new position."""
        return make_open_order(self._symbol, self._runner_key, price, signal, qty)

    def _make_close_order(
        self, price: float, old_signal: int, reason: str,
    ) -> list[OrderEvent]:
        """Create OrderEvent for closing current position."""
        return make_close_order(
            self._symbol, self._runner_key, price, old_signal, reason,
            self._current_qty, self._sizer.min_size,
        )
