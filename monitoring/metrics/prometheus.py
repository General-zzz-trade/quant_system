"""Prometheus metrics exporter for quant_system.

Bridges the internal MetricsRegistry to Prometheus exposition format.

Requires: pip install prometheus-client (in [monitoring] optional deps)
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class PrometheusExporter:
    """Exports quant_system metrics to Prometheus.

    Usage:
        exporter = PrometheusExporter(port=9090)
        exporter.start()

        # Update metrics
        exporter.set_gauge("equity_usd", 10500.0)
        exporter.inc_counter("fills_total")
        exporter.set_gauge("position_btcusdt", 0.05, labels={"symbol": "BTCUSDT"})
    """

    def __init__(self, *, port: int = 9090, prefix: str = "quant") -> None:
        self._port = port
        self._prefix = prefix
        self._gauges: Dict[str, Any] = {}
        self._counters: Dict[str, Any] = {}
        self._histograms: Dict[str, Any] = {}
        self._started = False

    def start(self) -> None:
        """Start the Prometheus HTTP server."""
        try:
            from prometheus_client import start_http_server  # type: ignore[import-untyped]
        except ImportError as e:
            raise RuntimeError(
                "prometheus-client not installed. Run: pip install prometheus-client"
            ) from e

        if not self._started:
            start_http_server(self._port)
            self._started = True
            logger.info("Prometheus metrics server started on port %d", self._port)

    def set_gauge(
        self, name: str, value: float, *, labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Set a gauge metric value."""
        key = f"{self._prefix}_{name}"

        if key not in self._gauges:
            try:
                from prometheus_client import Gauge  # type: ignore[import-untyped]
                label_names = list(labels.keys()) if labels else []
                self._gauges[key] = Gauge(key, f"{name} gauge", label_names)
            except ImportError:
                return

        gauge = self._gauges[key]
        if labels:
            gauge.labels(**labels).set(value)
        else:
            gauge.set(value)

    def inc_counter(
        self, name: str, value: float = 1.0, *, labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Increment a counter metric."""
        key = f"{self._prefix}_{name}"

        if key not in self._counters:
            try:
                from prometheus_client import Counter  # type: ignore[import-untyped]
                label_names = list(labels.keys()) if labels else []
                self._counters[key] = Counter(key, f"{name} counter", label_names)
            except ImportError:
                return

        counter = self._counters[key]
        if labels:
            counter.labels(**labels).inc(value)
        else:
            counter.inc(value)

    def observe_histogram(
        self, name: str, value: float, *, labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Observe a histogram metric value (e.g., latency)."""
        key = f"{self._prefix}_{name}"

        if key not in self._histograms:
            try:
                from prometheus_client import Histogram  # type: ignore[import-untyped]
                label_names = list(labels.keys()) if labels else []
                self._histograms[key] = Histogram(key, f"{name} histogram", label_names)
            except ImportError:
                return

        histogram = self._histograms[key]
        if labels:
            histogram.labels(**labels).observe(value)
        else:
            histogram.observe(value)
