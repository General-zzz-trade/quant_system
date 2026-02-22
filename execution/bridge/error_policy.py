# execution/bridge/error_policy.py
"""Error handling policies for execution bridge."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from execution.bridge.error_map import ErrorCategory, ErrorMapper


class ErrorAction(str, Enum):
    """错误后的动作。"""
    RETRY = "retry"
    REJECT = "reject"
    HALT = "halt"          # 暂停交易
    LOG_AND_SKIP = "log_and_skip"


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    """错误策略决策。"""
    action: ErrorAction
    reason: str
    retry_delay_sec: float = 0.0


class ErrorPolicy:
    """
    执行错误处理策略。

    根据错误分类决定重试/拒绝/暂停。
    """

    def __init__(
        self,
        *,
        mapper: Optional[ErrorMapper] = None,
        max_retries: int = 3,
        halt_on_insufficient_balance: bool = True,
    ) -> None:
        self._mapper = mapper or ErrorMapper()
        self._max_retries = max_retries
        self._halt_on_insufficient_balance = halt_on_insufficient_balance

    def decide(
        self,
        *,
        error_code: int,
        message: str = "",
        attempt: int = 1,
    ) -> PolicyDecision:
        """根据错误信息决定处理动作。"""
        mapping = self._mapper.classify(error_code, message)

        if mapping.category == ErrorCategory.INSUFFICIENT_BALANCE:
            if self._halt_on_insufficient_balance:
                return PolicyDecision(
                    action=ErrorAction.HALT,
                    reason=f"insufficient balance: {message}",
                )
            return PolicyDecision(
                action=ErrorAction.REJECT,
                reason=f"insufficient balance: {message}",
            )

        if mapping.category == ErrorCategory.RATE_LIMITED:
            if attempt <= self._max_retries:
                delay = min(5.0, 1.0 * (2 ** (attempt - 1)))
                return PolicyDecision(
                    action=ErrorAction.RETRY,
                    reason=f"rate limited (attempt {attempt})",
                    retry_delay_sec=delay,
                )
            return PolicyDecision(
                action=ErrorAction.REJECT,
                reason=f"rate limited: max retries exceeded",
            )

        if mapping.retryable and attempt <= self._max_retries:
            delay = min(2.0, 0.1 * (2 ** (attempt - 1)))
            return PolicyDecision(
                action=ErrorAction.RETRY,
                reason=f"retryable error (attempt {attempt}): {message}",
                retry_delay_sec=delay,
            )

        if mapping.category in (ErrorCategory.INVALID_PARAMS, ErrorCategory.NON_RETRYABLE):
            return PolicyDecision(
                action=ErrorAction.REJECT,
                reason=f"non-retryable: {message}",
            )

        return PolicyDecision(
            action=ErrorAction.LOG_AND_SKIP,
            reason=f"unknown error: code={error_code} msg={message}",
        )
