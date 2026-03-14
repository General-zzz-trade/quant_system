# execution/observability/metrics.py
"""Execution metrics — counters, gauges for monitoring."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Dict, Sequence


@dataclass(frozen=True, slots=True)
class MetricSnapshot:
    name: str
    value: float
    labels: Dict[str, str]
    ts: float


class Counter:
    """单调递增计数器。"""

    def __init__(self, name: str) -> None:
        self._name = name
        self._values: Dict[tuple, float] = {}
        self._lock = threading.Lock()

    def inc(self, value: float = 1.0, **labels: str) -> None:
        key = tuple(sorted(labels.items()))
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + value

    def get(self, **labels: str) -> float:
        key = tuple(sorted(labels.items()))
        return self._values.get(key, 0.0)

    def snapshot(self) -> Sequence[MetricSnapshot]:
        now = time.time()
        with self._lock:
            return [MetricSnapshot(self._name, v, dict(k), now) for k, v in self._values.items()]


class Gauge:
    """可上可下的指标值。"""

    def __init__(self, name: str) -> None:
        self._name = name
        self._values: Dict[tuple, float] = {}
        self._lock = threading.Lock()

    def set(self, value: float, **labels: str) -> None:
        key = tuple(sorted(labels.items()))
        with self._lock:
            self._values[key] = value

    def get(self, **labels: str) -> float:
        key = tuple(sorted(labels.items()))
        return self._values.get(key, 0.0)


class ExecutionMetrics:
    """执行层指标集合。"""

    def __init__(self) -> None:
        self.orders_submitted = Counter("exec_orders_submitted")
        self.orders_filled = Counter("exec_orders_filled")
        self.orders_rejected = Counter("exec_orders_rejected")
        self.fills_received = Counter("exec_fills_received")
        self.errors = Counter("exec_errors")
        self.retries = Counter("exec_retries")
        self.active_orders = Gauge("exec_active_orders")

    def record_submit(self, venue: str, symbol: str) -> None:
        self.orders_submitted.inc(venue=venue, symbol=symbol)

    def record_fill(self, venue: str, symbol: str) -> None:
        self.orders_filled.inc(venue=venue, symbol=symbol)
        self.fills_received.inc(venue=venue, symbol=symbol)

    def record_reject(self, venue: str, symbol: str) -> None:
        self.orders_rejected.inc(venue=venue, symbol=symbol)

    def record_error(self, venue: str, error_type: str) -> None:
        self.errors.inc(venue=venue, error_type=error_type)
