# execution/bridge/rate_limit.py
"""Rate limiting utilities for execution bridge."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass(slots=True)
class SlidingWindowCounter:
    """
    滑动窗口计数器 — 用于精确限速。

    比 TokenBucket 更精确的限速方式，适合交易所对「每分钟/每秒请求数」
    有严格要求的场景。
    """
    max_count: int
    window_sec: float
    _timestamps: list[float] = field(default_factory=list, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def allow(self) -> bool:
        now = time.monotonic()
        with self._lock:
            cutoff = now - self.window_sec
            self._timestamps = [t for t in self._timestamps if t > cutoff]
            if len(self._timestamps) >= self.max_count:
                return False
            self._timestamps.append(now)
            return True

    def reset(self) -> None:
        with self._lock:
            self._timestamps.clear()

    @property
    def current_count(self) -> int:
        now = time.monotonic()
        with self._lock:
            cutoff = now - self.window_sec
            self._timestamps = [t for t in self._timestamps if t > cutoff]
            return len(self._timestamps)


@dataclass(slots=True)
class WeightedRateLimiter:
    """
    带权重的限速器。

    不同操作消耗不同的权重（例如 Binance REST 接口按 weight 计算）。
    """
    max_weight: float
    window_sec: float
    _entries: list[tuple[float, float]] = field(default_factory=list, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def allow(self, weight: float = 1.0) -> bool:
        now = time.monotonic()
        with self._lock:
            cutoff = now - self.window_sec
            self._entries = [(t, w) for t, w in self._entries if t > cutoff]
            current = sum(w for _, w in self._entries)
            if current + weight > self.max_weight:
                return False
            self._entries.append((now, weight))
            return True

    @property
    def remaining_weight(self) -> float:
        now = time.monotonic()
        with self._lock:
            cutoff = now - self.window_sec
            self._entries = [(t, w) for t, w in self._entries if t > cutoff]
            current = sum(w for _, w in self._entries)
            return max(0.0, self.max_weight - current)
