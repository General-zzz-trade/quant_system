# execution/bridge/timeout.py
"""Timeout management for execution operations."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional


class TimeoutError(RuntimeError):
    """执行操作超时。"""


@dataclass(frozen=True, slots=True)
class TimeoutConfig:
    """超时配置。"""
    connect_sec: float = 10.0
    read_sec: float = 5.0
    total_sec: float = 30.0       # 包含重试的总超时


class TimeoutGuard:
    """
    超时守卫 — 追踪操作是否超时。

    用法：
        guard = TimeoutGuard(timeout_sec=30.0)
        guard.start()
        while not guard.expired:
            do_work()
    """

    def __init__(self, timeout_sec: float) -> None:
        self._timeout = timeout_sec
        self._start: Optional[float] = None

    def start(self) -> None:
        """开始计时。"""
        self._start = time.monotonic()

    @property
    def elapsed(self) -> float:
        """已经过的时间（秒）。"""
        if self._start is None:
            return 0.0
        return time.monotonic() - self._start

    @property
    def remaining(self) -> float:
        """剩余时间（秒）。"""
        return max(0.0, self._timeout - self.elapsed)

    @property
    def expired(self) -> bool:
        """是否已超时。"""
        return self.elapsed >= self._timeout

    def check(self, context: str = "") -> None:
        """检查是否超时，超时则抛异常。"""
        if self.expired:
            msg = f"timeout after {self.elapsed:.2f}s (limit={self._timeout}s)"
            if context:
                msg = f"{context}: {msg}"
            raise TimeoutError(msg)
