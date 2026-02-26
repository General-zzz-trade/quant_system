# monitoring/engine_hook.py
"""EngineMonitoringHook — bridges pipeline output to health monitor and Prometheus."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Optional

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
    _event_count: int = field(default=0, init=False)

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

            self.metrics.inc_counter("pipeline_events_total")
            self.metrics.set_gauge("event_index", float(out.event_index))

            # Per-symbol prices
            for sym, mkt in out.markets.items():
                price = getattr(mkt, "close", None) or getattr(mkt, "last_price", None)
                if price is not None:
                    self.metrics.set_gauge("price", float(price), labels={"symbol": sym})

            # Per-symbol position sizes
            for sym, pos in out.positions.items():
                qty = getattr(pos, "qty", None) or getattr(pos, "quantity", None)
                if qty is not None:
                    self.metrics.set_gauge(
                        "position_qty", float(qty), labels={"symbol": sym},
                    )
