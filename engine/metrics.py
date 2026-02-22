# engine/metrics.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
import time
import threading


@dataclass(slots=True)
class Counter:
    value: int = 0

    def inc(self, n: int = 1) -> None:
        self.value += n


@dataclass(slots=True)
class Gauge:
    value: float = 0.0

    def set(self, v: float) -> None:
        self.value = float(v)


@dataclass(slots=True)
class Histogram:
    """
    极简直方图（v1.0）：只保留 count / sum / min / max
    不做分桶，避免依赖与性能成本
    """
    count: int = 0
    total: float = 0.0
    min: Optional[float] = None
    max: Optional[float] = None

    def observe(self, v: float) -> None:
        self.count += 1
        self.total += v
        self.min = v if self.min is None else min(self.min, v)
        self.max = v if self.max is None else max(self.max, v)

    @property
    def avg(self) -> Optional[float]:
        return None if self.count == 0 else self.total / self.count


class MetricsRegistry:
    """
    冻结版 v1.0 指标注册表（线程安全、低开销）
    原则：
    - 只存数值，不做 IO
    - 不绑定具体 exporter（Prometheus/OTLP 由外部适配）
    """
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.counters: Dict[str, Counter] = {}
        self.gauges: Dict[str, Gauge] = {}
        self.hists: Dict[str, Histogram] = {}

    def counter(self, name: str) -> Counter:
        with self._lock:
            return self.counters.setdefault(name, Counter())

    def gauge(self, name: str) -> Gauge:
        with self._lock:
            return self.gauges.setdefault(name, Gauge())

    def histogram(self, name: str) -> Histogram:
        with self._lock:
            return self.hists.setdefault(name, Histogram())

    # ---- helpers (engine 常用) ----
    def inc_event(self, route: str) -> None:
        self.counter(f"events_total{{route={route}}}").inc()

    def observe_latency(self, stage: str, seconds: float) -> None:
        self.histogram(f"latency_seconds{{stage={stage}}}").observe(seconds)

    def set_queue_depth(self, name: str, depth: int) -> None:
        self.gauge(f"queue_depth{{name={name}}}").set(depth)


# 轻量计时器（with 用法）
class _Timer:
    def __init__(self, observe_fn) -> None:
        self._observe = observe_fn
        self._t0 = 0.0

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        dt = time.perf_counter() - self._t0
        self._observe(dt)


def timer(registry: MetricsRegistry, stage: str) -> _Timer:
    return _Timer(lambda s: registry.observe_latency(stage, s))
