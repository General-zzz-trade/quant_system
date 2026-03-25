# monitoring/engine_hook.py
"""EngineMonitoringHook — bridges pipeline output to health monitor and Prometheus."""
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Deque, Dict, Optional, Tuple

from engine.pipeline import PipelineOutput
from monitoring.health import SystemHealthMonitor
from monitoring.metrics.prometheus import PrometheusExporter

logger = logging.getLogger(__name__)


@dataclass
class EngineMonitoringHook:
    """Pipeline output hook that feeds health monitor and Prometheus metrics.

    Inject via CoordinatorConfig.on_pipeline_output.

    Usage:
        hook = EngineMonitoringHook(health=monitor, metrics=exporter)
        cfg = CoordinatorConfig(..., on_pipeline_output=hook)
    """

    health: Optional[SystemHealthMonitor] = None
    metrics: Optional[PrometheusExporter] = None
    kill_switch: Optional[Any] = None
    inference_bridge: Optional[Any] = None
    alpha_health_monitor: Optional[Any] = None
    drawdown_breaker: Optional[Any] = None
    live_signal_tracker: Optional[Any] = None
    latency_tracker: Optional[Any] = None
    regime_sizer: Optional[Any] = None  # Direction 17: RegimePositionSizer
    _vol_detector: Optional[Any] = field(default=None, init=False)  # cached detector
    _latency_sla_threshold_ms: float = 5000.0
    _latency_sla_breach_start: Optional[float] = field(default=None, init=False)
    _latency_sla_reduce_duration_sec: float = 300.0  # 5 minutes reduce-only
    _rest_fallback_count: int = field(default=0, init=False)
    _rest_fallback_window_start: float = field(default=0.0, init=False)
    _REST_FALLBACK_WINDOW_SEC: float = 300.0  # 5 minute window
    _REST_FALLBACK_THRESHOLD: int = 5  # fallback count threshold for alerting
    _event_count: int = field(default=0, init=False)
    _order_count: int = field(default=0, init=False)
    _order_rejection_count: int = field(default=0, init=False)
    _reconcile_failure_count: int = field(default=0, init=False)
    _reconcile_drift_count: int = field(default=0, init=False)
    _equity_hwm: float = field(default=0.0, init=False)
    _last_signal_ts: float = field(default=0.0, init=False)
    execution_quality: Optional[Any] = None
    decision_store: Optional[Any] = None  # DecisionStore for replay recording
    # Alpha health: buffer (close, ml_score) per symbol for lagged IC computation
    _alpha_pred_buffer: Dict[str, Deque[Tuple[float, float]]] = field(default_factory=dict, init=False)
    _ALPHA_MAX_LAG: int = 50  # max horizon to support
    _weight_recs: Dict[str, float] = field(default_factory=dict, init=False)
    _WEIGHT_REC_INTERVAL: int = 100  # recompute every N bars

    def on_order(self) -> None:
        self._order_count += 1

    def on_order_rejection(self) -> None:
        self._order_rejection_count += 1

    def on_reconcile_failure(self) -> None:
        self._reconcile_failure_count += 1

    def on_reconcile_drift(self) -> None:
        self._reconcile_drift_count += 1

    @property
    def weight_recommendations(self) -> Dict[str, float]:
        """Latest weight recommendations from attribution feedback loop."""
        return self._weight_recs

    def on_rest_fallback(self) -> None:
        """Record a REST fallback event (WS order failed, fell back to REST)."""
        now = time.time()
        if now - self._rest_fallback_window_start > self._REST_FALLBACK_WINDOW_SEC:
            self._rest_fallback_count = 0
            self._rest_fallback_window_start = now
        self._rest_fallback_count += 1
        if self.metrics is not None:
            self.metrics.inc_counter("ws_rest_fallback_total")

    def on_signal(self, symbol: str, value: float, latency_sec: float) -> None:
        self._last_signal_ts = time.time()
        if self.metrics is not None:
            self.metrics.observe_histogram("signal_latency_seconds", latency_sec, labels={"symbol": symbol})
            self.metrics.set_gauge("signal_value", value, labels={"symbol": symbol})

    def on_feature_compute(self, latency_sec: float) -> None:
        if self.metrics is not None:
            self.metrics.observe_histogram("feature_compute_seconds", latency_sec)

    def on_model_inference(self, latency_sec: float, model: str = "default") -> None:
        if self.metrics is not None:
            self.metrics.observe_histogram("model_inference_seconds", latency_sec, labels={"model": model})

    # Metrics decimation: non-critical gauges update every _METRICS_DECIMATION ticks.
    # Critical metrics (equity, balance, drawdown, event counter) update every tick.
    _METRICS_DECIMATION: int = 10

    def __call__(self, out: PipelineOutput) -> None:
        self._event_count += 1

        # ── Alpha Health Monitor: feed predictions and check IC ──
        if self.alpha_health_monitor is not None and out.features is not None:
            self._update_alpha_health(out)

        # ── Decision recording: append ML score for replay support ──
        if self.decision_store is not None and out.features is not None:
            features = out.features
            ml_score = features.get("ml_score") if features else None
            if ml_score is not None:
                sym = features.get("_symbol", "")
                self.decision_store.append({
                    "ts": time.time(),
                    "symbol": sym,
                    "ml_score": float(ml_score),
                    "event_index": out.event_index,
                })

        # ── Regime Position Sizer: feed volatility regime (Direction 17) ──
        if self.regime_sizer is not None and out.features is not None:
            self._update_regime_sizer(out)

        # Health: mark data freshness
        if self.health is not None:
            self.health.on_market_data()

        # Extract balance/equity
        account = out.account
        balance = getattr(account, "balance", None)
        equity = getattr(account, "equity", None) or balance

        if self.health is not None and (balance is not None or equity is not None):
            self.health.on_balance_update(
                balance=Decimal(balance) if isinstance(balance, (int, float)) else balance,
                equity=Decimal(equity) if isinstance(equity, (int, float)) else equity,
            )

        # Prometheus metrics
        if self.metrics is not None:
            # ── Critical metrics: every tick ──
            if balance is not None:
                self.metrics.set_gauge("balance_usdt", float(balance))
            if equity is not None:
                self.metrics.set_gauge("equity_usdt", float(equity))
                eq_f = float(equity)
                if eq_f > self._equity_hwm:
                    self._equity_hwm = eq_f
                if self._equity_hwm > 0:
                    dd_pct = (self._equity_hwm - eq_f) / self._equity_hwm * 100.0
                    self.metrics.set_gauge("drawdown_pct", dd_pct)

            # ── Drawdown circuit breaker: continuous equity monitoring ──
            if self.drawdown_breaker is not None and equity is not None:
                self.drawdown_breaker.on_equity_update(float(equity))

            self.metrics.inc_counter("pipeline_events_total")

            # ── Non-critical metrics: decimated (always fire on first tick) ──
            if self._event_count > 1 and self._event_count % self._METRICS_DECIMATION != 0:
                return

            # ── Live signal attribution export ──
            if self.live_signal_tracker is not None:
                self.live_signal_tracker.export_metrics()

            # ── Weight recommendations (execution feedback loop) ──
            if (self._event_count % self._WEIGHT_REC_INTERVAL == 0
                    and self.live_signal_tracker is not None):
                try:
                    self._weight_recs = self.live_signal_tracker.compute_weight_recommendations(
                        alpha_health_monitor=self.alpha_health_monitor,
                    )
                except Exception:
                    logger.debug("Weight recommendation compute failed", exc_info=True)

            self.metrics.set_gauge("event_index", float(out.event_index))

            # Market data age + prices + positions
            now = time.time()
            for sym, mkt in out.markets.items():
                ts = getattr(mkt, "ts", None) or getattr(mkt, "last_ts", None)
                if ts is not None:
                    try:
                        self.metrics.set_gauge("market_data_age_seconds", now - float(ts), labels={"symbol": sym})
                    except (TypeError, ValueError):
                        pass
                price = getattr(mkt, "close", None) or getattr(mkt, "last_price", None)
                if price is not None:
                    self.metrics.set_gauge("price", float(price), labels={"symbol": sym})

            for sym, pos in out.positions.items():
                qty = getattr(pos, "qty", None) or getattr(pos, "quantity", None)
                if qty is not None:
                    qty_f = float(qty)
                    self.metrics.set_gauge("position_qty", qty_f, labels={"symbol": sym})
                    self.metrics.set_gauge("position_size", abs(qty_f), labels={"symbol": sym})
                upnl = getattr(pos, "unrealized_pnl", None)
                if upnl is not None:
                    self.metrics.set_gauge("unrealized_pnl", float(upnl), labels={"symbol": sym})

            # Kill switch + counters
            if self.kill_switch is not None:
                active = getattr(self.kill_switch, "is_active", False)
                if callable(active):
                    active = active()
                self.metrics.set_gauge("kill_switch_active", 1.0 if active else 0.0)

            self.metrics.set_gauge("orders_total", float(self._order_count))
            self.metrics.set_gauge("order_rejections_total", float(self._order_rejection_count))
            self.metrics.set_gauge("reconcile_failures_total", float(self._reconcile_failure_count))
            self.metrics.set_gauge("reconcile_drift_count", float(self._reconcile_drift_count))

            # Execution quality metrics (optional)
            if self.execution_quality is not None:
                eq = self.execution_quality
                fill_rate = getattr(eq, "fill_rate", None)
                if fill_rate is not None:
                    self.metrics.set_gauge("fill_rate", float(fill_rate))
                avg_slip = getattr(eq, "avg_slippage_bps", None)
                if avg_slip is not None:
                    self.metrics.set_gauge("avg_slippage_bps", float(avg_slip))
                avg_latency = getattr(eq, "avg_fill_latency_ms", None)
                if avg_latency is not None:
                    self.metrics.set_gauge("avg_fill_latency_ms", float(avg_latency))

            # Bridge hold state
            if self.inference_bridge is not None:
                bridge = self.inference_bridge
                for sym in out.positions:
                    hold_count = bridge._hold_counter.get(sym, 0)
                    min_hold = bridge._min_hold_bars.get(sym, 0)
                    remaining = max(0, min_hold - hold_count)
                    self.metrics.set_gauge(
                        "bridge_hold_remaining", float(remaining), labels={"symbol": sym},
                    )

            # Alpha health position scale
            if self.alpha_health_monitor is not None:
                for sym in out.positions:
                    scale = self.alpha_health_monitor.position_scale(sym)
                    self.metrics.set_gauge(
                        "alpha_position_scale", scale, labels={"symbol": sym},
                    )

    def _update_alpha_health(self, out: PipelineOutput) -> None:
        """Feed alpha health monitor with predictions and lagged returns.

        Buffers (close, ml_score) per symbol. When enough bars have passed,
        computes realized return and calls monitor.update() for each horizon.
        Then calls on_bar() to check for IC state transitions.
        """
        monitor = self.alpha_health_monitor
        features = out.features
        if features is None:
            return

        symbol = features.get("_symbol")
        if symbol is None:
            return

        close = features.get("close")
        ml_score = features.get("ml_score")
        if close is None:
            return

        close_f = float(close)

        # Buffer prediction and close
        if symbol not in self._alpha_pred_buffer:
            self._alpha_pred_buffer[symbol] = deque(maxlen=self._ALPHA_MAX_LAG + 1)
        buf = self._alpha_pred_buffer[symbol]
        buf.append((close_f, float(ml_score) if ml_score is not None else 0.0))

        # Feed lagged returns to monitor for each registered horizon
        monitors = monitor._monitors.get(symbol)
        if monitors is not None:
            for horizon in monitors:
                if len(buf) > horizon:
                    # The prediction made `horizon` bars ago
                    past_close, past_pred = buf[-(horizon + 1)]
                    # Realized return over the horizon
                    actual_return = (close_f - past_close) / past_close if past_close != 0 else 0.0
                    monitor.update(symbol, horizon, past_pred, actual_return)

        # Check for IC state transitions (respects eval_interval_bars internally)
        alerts = monitor.on_bar(symbol)
        for alert in alerts:
            logger.warning(
                "AlphaHealth %s: %s h%d IC=%.4f days_neg=%d",
                alert.alert_type, alert.symbol, alert.horizon,
                alert.rolling_ic, alert.days_negative,
            )

    def _update_regime_sizer(self, out: PipelineOutput) -> None:
        """Feed regime sizer with current volatility regime (Direction 17)."""
        sizer = self.regime_sizer
        features = out.features
        if features is None:
            return

        symbol = features.get("_symbol")
        if symbol is None:
            return

        # Lazily create volatility detector
        if self._vol_detector is None:
            from strategy.regime.volatility import VolatilityRegimeDetector
            self._vol_detector = VolatilityRegimeDetector()

        from datetime import datetime, timezone
        ts = datetime.now(timezone.utc)
        label = self._vol_detector.detect(symbol=symbol, ts=ts, features=features)
        if label is not None:
            scale = sizer.update(symbol, label)
            if self.metrics is not None:
                self.metrics.set_gauge(
                    "regime_position_scale", scale, labels={"symbol": symbol},
                )
                self.metrics.set_gauge(
                    "regime_vol_score", label.score, labels={"symbol": symbol},
                )
