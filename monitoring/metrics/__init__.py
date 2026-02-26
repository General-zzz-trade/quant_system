from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class Counter:
    value: int = 0

    def inc(self, n: int = 1) -> None:
        self.value += int(n)


@dataclass
class Gauge:
    value: float = 0.0

    def set(self, v: float) -> None:
        self.value = float(v)


class Timer:
    def __init__(self) -> None:
        self._t0: Optional[float] = None
        self.elapsed: float = 0.0

    def start(self) -> None:
        self._t0 = time.perf_counter()

    def stop(self) -> None:
        if self._t0 is None:
            return
        self.elapsed += time.perf_counter() - self._t0
        self._t0 = None

    def __enter__(self) -> "Timer":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()


class MetricsRegistry:
    """In-memory metrics registry."""

    def __init__(self) -> None:
        self.counters: Dict[str, Counter] = {}
        self.gauges: Dict[str, Gauge] = {}
        self.timers: Dict[str, Timer] = {}

    def counter(self, name: str) -> Counter:
        if name not in self.counters:
            self.counters[name] = Counter()
        return self.counters[name]

    def gauge(self, name: str) -> Gauge:
        if name not in self.gauges:
            self.gauges[name] = Gauge()
        return self.gauges[name]

    def timer(self, name: str) -> Timer:
        if name not in self.timers:
            self.timers[name] = Timer()
        return self.timers[name]

    def snapshot(self) -> Dict[str, object]:
        return {
            "counters": {k: v.value for k, v in self.counters.items()},
            "gauges": {k: v.value for k, v in self.gauges.items()},
            "timers": {k: v.elapsed for k, v in self.timers.items()},
        }
