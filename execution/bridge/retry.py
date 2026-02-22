# execution/bridge/retry.py
"""Retry logic with exponential backoff for execution bridge."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Optional, TypeVar

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class RetryResult:
    """重试执行的结果。"""
    success: bool
    result: Any = None
    error: Optional[Exception] = None
    attempts: int = 0


def retry_with_backoff(
    fn: Callable[[], T],
    *,
    max_attempts: int = 3,
    base_delay: float = 0.1,
    max_delay: float = 2.0,
    jitter: float = 0.0,
    is_retryable: Optional[Callable[[Exception], bool]] = None,
    on_retry: Optional[Callable[[int, Exception], None]] = None,
    sleeper: Optional[Callable[[float], None]] = None,
) -> RetryResult:
    """
    带指数退避的重试执行。

    Args:
        fn: 要执行的函数
        max_attempts: 最大尝试次数
        base_delay: 基础延迟（秒）
        max_delay: 最大延迟（秒）
        jitter: 抖动（秒）
        is_retryable: 判断异常是否可重试
        on_retry: 重试回调 (attempt, error)
        sleeper: 休眠函数（用于测试注入）
    """
    _sleep = sleeper or time.sleep
    _retryable = is_retryable or (lambda _: True)

    last_error: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        try:
            result = fn()
            return RetryResult(success=True, result=result, attempts=attempt)
        except Exception as e:
            last_error = e
            if not _retryable(e) or attempt >= max_attempts:
                return RetryResult(
                    success=False, error=e, attempts=attempt,
                )
            if on_retry:
                on_retry(attempt, e)
            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            delay += jitter
            if delay > 0:
                _sleep(delay)

    return RetryResult(success=False, error=last_error, attempts=max_attempts)
