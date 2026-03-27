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
        self._rets: list[float] = []
        self._vol_history: list[float] = []
        self._trend_history: list[float] = []
        self._regime_active: bool = True

        # Stop-loss
        self._atr_buffer: list[float] = []

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

        # Decision audit logger (best-effort, never affects trading)
        self._audit = DecisionAuditLogger()

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

        # 3. Predict
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

        # 5. Force exits
        force_exit, exit_reason = self._check_force_exits(close, z)
        if force_exit:
            new_signal = 0

        # Audit: log every signal evaluation (best-effort)
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

            # Open new position
            if new_signal != 0:
                self._refresh_ic_scale()
                z_scale = self._compute_z_scale(z)
                qty = self._sizer.target_qty(
                    snapshot,
                    self._symbol,
                    leverage=self._leverage,
                    ic_scale=self._ic_scale,
                    regime_active=self._regime_active,
                    z_scale=z_scale,
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
                if qty <= 0:
                    return events  # skip zero/negative qty (warmup, edge case)
                events.extend(self._make_open_order(close, new_signal, qty))
                try:
                    self._audit.log_entry(
                        symbol=self._symbol,
                        side="buy" if new_signal == 1 else "sell",
                        qty=float(qty), price=close,
                        reason="signal", z_score=z, ic_scale=self._ic_scale,
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

    def _check_force_exits(self, close: float, z: float) -> tuple[bool, str]:
        """Check for forced exit conditions.  Priority order."""
        if self._signal == 0 or self._entry_price <= 0:
            return False, ""

        atr = self._current_atr()

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

        # Phase selection (3-phase: trailing → breakeven → initial)
        if profit_pct >= 1.0 * atr:
            # Trailing phase: tight stop near peak
            stop_dist = atr * 0.2
        elif profit_pct >= 0.5 * atr:
            # Breakeven phase: moderate stop near entry
            stop_dist = atr * 0.1
        else:
            # Initial phase: wide stop for new positions
            stop_dist = atr * 1.2

        # Hard floor/ceiling
        stop_dist = np.clip(stop_dist, 0.003, 0.05)

        if drawdown_pct > stop_dist:
            return True, f"atr_stop({drawdown_pct:.3f}>{stop_dist:.3f})"

        # Quick loss: -1% adverse move from entry
        if self._signal == 1:
            adverse = (self._entry_price - close) / self._entry_price
        else:
            adverse = (close - self._entry_price) / self._entry_price

        if adverse > 0.005:  # 0.5% adverse = 5% account loss at 10x
            return True, f"quick_loss({adverse:.3f})"

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

        return False, ""

    @staticmethod
    def _compute_z_scale(z: float) -> float:
        """Map |z| to confidence-based position scale."""
        abs_z = abs(z)
        if abs_z > 2.0:
            return 1.2   # cap at 1.2x (was 1.5x) — prevents 15x spikes
        if abs_z > 1.0:
            return 1.0
        if abs_z > 0.5:
            return 0.8   # slightly more aggressive at moderate z
        return 0.5

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
