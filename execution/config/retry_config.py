# execution/config/retry_config.py
"""Retry policy configuration."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RetryConfig:
    """
    重试策略配置。

    用于 ExecutionBridge 的重试循环。
    与 execution_bridge.py 中 RetryPolicy 对齐。
    """
    max_attempts: int = 3
    base_delay_sec: float = 0.10
    max_delay_sec: float = 2.00
    jitter_sec: float = 0.0
    retryable_status_codes: tuple[int, ...] = (408, 429, 500, 502, 503, 504)

    def delay_for_attempt(self, attempt: int) -> float:
        """计算第 N 次重试的延迟（指数退避）。"""
        delay = min(self.max_delay_sec, self.base_delay_sec * (2 ** (attempt - 1)))
        return delay + self.jitter_sec

    def is_retryable_status(self, status_code: int) -> bool:
        return status_code in self.retryable_status_codes
