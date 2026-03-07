# monitoring/engine_hook.py
"""EngineMonitoringHook — bridges pipeline output to health monitor and Prometheus."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Optional

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
    _event_count: int = field(default=0, init=False)
    _order_count: int = field(default=0, init=False)
    _order_rejection_count: int = field(default=0, init=False)
    _reconcile_failure_count: int = field(default=0, init=False)
    _reconcile_drift_count: int = field(default=0, init=False)
    _equity_hwm: float = field(default=0.0, init=False)
    _last_signal_ts: float = field(default=0.0, init=False)
    execution_quality: Optional[Any] = None

    def on_order(self) -> None:
        self._order_count += 1

    def on_order_rejection(self) -> None:
        self._order_rejection_count += 1

    def on_reconcile_failure(self) -> None:
        self._reconcile_failure_count += 1

    def on_reconcile_drift(self) -> None:
        self._reconcile_drift_count += 1

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

    def __call__(self, out: PipelineOutput) -> None:
        self._event_count += 1

        # Health: mark data freshness
        if self.health is not None:
            self.health.on_market_data()

        # Extract balance/equity
        account = out.account
        balance = getattr(account, "balance", None)
        equity = getattr(account, "equity", None) or balance

        if self.health is not None and (balance is not None or equity is not None):
            self.health.on_balance_update(
                balance=Decimal(str(balance)) if balance is not None else None,
                equity=Decimal(str(equity)) if equity is not None else None,
            )

        # Prometheus metrics
        if self.metrics is not None:
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

            self.metrics.inc_counter("pipeline_events_total")
            self.metrics.set_gauge("event_index", float(out.event_index))

            # Market data age
            for sym, mkt in out.markets.items():
                ts = getattr(mkt, "ts", None) or getattr(mkt, "last_ts", None)
                if ts is not None:
                    try:
                        age = time.time() - float(ts)
                        self.metrics.set_gauge("market_data_age_seconds", age, labels={"symbol": sym})
                    except (TypeError, ValueError):
                        pass

            # Kill switch state
            if self.kill_switch is not None:
                active = getattr(self.kill_switch, "is_active", False)
                if callable(active):
                    active = active()
                self.metrics.set_gauge("kill_switch_active", 1.0 if active else 0.0)

            # Order counters
            self.metrics.set_gauge("orders_total", float(self._order_count))
            self.metrics.set_gauge("order_rejections_total", float(self._order_rejection_count))
            self.metrics.set_gauge("reconcile_failures_total", float(self._reconcile_failure_count))
            self.metrics.set_gauge("reconcile_drift_count", float(self._reconcile_drift_count))

            # Per-symbol prices
            for sym, mkt in out.markets.items():
                price = getattr(mkt, "close", None) or getattr(mkt, "last_price", None)
                if price is not None:
                    self.metrics.set_gauge("price", float(price), labels={"symbol": sym})

            # Per-symbol position sizes and unrealized PnL
            for sym, pos in out.positions.items():
                qty = getattr(pos, "qty", None) or getattr(pos, "quantity", None)
                if qty is not None:
                    self.metrics.set_gauge(
                        "position_qty", float(qty), labels={"symbol": sym},
                    )
                    self.metrics.set_gauge(
                        "position_size", abs(float(qty)), labels={"symbol": sym},
                    )
                upnl = getattr(pos, "unrealized_pnl", None)
                if upnl is not None:
                    self.metrics.set_gauge(
                        "unrealized_pnl", float(upnl), labels={"symbol": sym},
                    )

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
