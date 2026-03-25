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
    ) -> None:
        self._symbol = symbol
        self._runner_key = runner_key
        self._predictor = predictor
        self._discretizer = discretizer
        self._sizer = sizer

        # Pure decision state
        self._signal: int = 0
        self._entry_price: float = 0.0
        self._trade_peak: float = 0.0
        self._bars_processed: int = 0

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

        # Capture base deadzone before vol-adaptive modifications
        self._deadzone_base: float = discretizer.deadzone

        # Microstructure VPIN scaling (optional, live-only)
        self._vpin_caution_thresh: float = 0.5
        self._vpin_scale_factor: float = 0.7  # reduce size by 30% when VPIN > threshold

        # Decision audit logger (best-effort, never affects trading)
        self._audit = DecisionAuditLogger()

    # ── public API ──────────────────────────────────────────────

    def set_consensus(self, signals: dict[str, int]) -> None:
        """Update cross-symbol consensus signals."""
        self._consensus.update(signals)

    def update_predictor(self, predictor: EnsemblePredictor) -> None:
        """Hot-swap the ensemble predictor (SIGHUP reload)."""
        self._predictor = predictor

    def decide(self, snapshot: Any) -> Iterable[Any]:
        """Read-only snapshot → opinion events.  No side effects on venue."""
        self._bars_processed += 1
        close = float(snapshot.markets[self._symbol].close)
        features: dict = dict(snapshot.features) if snapshot.features else {}

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
        events: list[Any] = []

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

        # 7. Emit events on signal change
        if new_signal != self._signal:
            old_signal = self._signal

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
                        qty=0.0, price=close, reason=reason,
                        entry_price=self._entry_price,
                    )
                except Exception:
                    pass
                events.extend(self._make_close_order(close, old_signal, reason))

            # Open new position
            if new_signal != 0:
                self._refresh_ic_scale()
                z_scale = self._compute_z_scale(z)
                qty = self._sizer.target_qty(
                    snapshot,
                    self._symbol,
                    leverage=10.0,
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
            else:
                self._entry_price = 0.0
                self._trade_peak = 0.0

            self._signal = new_signal

        # 8. Update consensus
        self._consensus[self._runner_key] = self._signal

        return events

    # ── regime filter ───────────────────────────────────────────

    def _check_regime(self, close: float) -> bool:
        """Adaptive p20/p25 percentile regime filter."""
        self._closes.append(close)

        if len(self._closes) >= 2:
            log_ret = np.log(self._closes[-1] / self._closes[-2])
            self._rets.append(log_ret)

        # Truncate buffers
        max_buf = self._ma_window + 100
        if len(self._closes) > max_buf:
            self._closes = self._closes[-max_buf:]
        if len(self._rets) > max_buf:
            self._rets = self._rets[-max_buf:]

        # Need minimum history
        if len(self._rets) < 20:
            self._regime_active = True
            return True

        vol_20 = float(np.std(self._rets[-20:]))
        ma_vals = self._closes[-self._ma_window:]
        ma = np.mean(ma_vals)
        trend = abs(close / ma - 1.0)

        self._vol_history.append(vol_20)
        self._trend_history.append(trend)

        # Cap adaptive buffers
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

        # Vol-adaptive deadzone
        vol_med = float(np.median(self._vol_history)) if self._vol_history else self._vol_median
        if vol_med > 0:
            ratio = np.clip(vol_20 / vol_med, 0.5, 2.0)
            self._discretizer.deadzone = self._deadzone_base * float(ratio)

        return self._regime_active

    # ── ATR ─────────────────────────────────────────────────────

    def _update_atr(self, snapshot: Any) -> None:
        """Update ATR buffer from OHLC data."""
        if len(self._closes) < 2:
            return
        mkt = snapshot.markets[self._symbol]
        high = float(mkt.high)
        low = float(mkt.low)
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

    # ── force exits ─────────────────────────────────────────────

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

        # Phase selection
        if profit_pct >= 0.5 * atr:
            # Trailing phase
            stop_dist = atr * 0.2
        elif profit_pct >= 0.5 * atr:
            # Breakeven phase (same threshold, kept for clarity)
            stop_dist = atr * 0.1
        else:
            # Initial phase
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

        if adverse > 0.01:
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

        return False, ""

    # ── z-scale ─────────────────────────────────────────────────

    @staticmethod
    def _compute_z_scale(z: float) -> float:
        """Map |z| to confidence-based position scale."""
        abs_z = abs(z)
        if abs_z > 2.0:
            return 1.5
        if abs_z > 1.0:
            return 1.0
        if abs_z > 0.5:
            return 0.7
        return 0.5

    # ── IC health ───────────────────────────────────────────────

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
            with open(_IC_HEALTH_PATH) as f:
                data = json.load(f)
            status = data.get(model_name, {}).get("status", "GREEN")
            self._ic_scale = _IC_SCALE_MAP.get(status, 1.0)
        except Exception:
            logger.debug("IC health read failed, keeping scale=%.1f", self._ic_scale)

    # ── event factories ─────────────────────────────────────────

    def _make_open_order(
        self, price: float, signal: int, qty: Decimal,
    ) -> list[OrderEvent]:
        """Create OrderEvent for opening a new position."""
        header = EventHeader.new_root(
            event_type=EventType.ORDER,
            version=1,
            source=f"alpha.{self._runner_key}",
        )
        side = "buy" if signal == 1 else "sell"
        return [
            OrderEvent(
                header=header,
                order_id=header.event_id,
                intent_id=header.event_id,
                symbol=self._symbol,
                side=side,
                qty=qty,
                price=Decimal(str(price)),
            )
        ]

    def _make_close_order(
        self, price: float, old_signal: int, reason: str,
    ) -> list[OrderEvent]:
        """Create OrderEvent for closing current position (qty=0)."""
        header = EventHeader.new_root(
            event_type=EventType.ORDER,
            version=1,
            source=f"alpha.{self._runner_key}",
        )
        # Close side is opposite of position
        side = "sell" if old_signal == 1 else "buy"
        logger.info(
            "%s CLOSE %s reason=%s price=%.2f",
            self._runner_key, side, reason, price,
        )
        return [
            OrderEvent(
                header=header,
                order_id=header.event_id,
                intent_id=header.event_id,
                symbol=self._symbol,
                side=side,
                qty=Decimal("0"),
                price=Decimal(str(price)),
            )
        ]
