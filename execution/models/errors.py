# execution/models/errors.py
"""Execution-layer error hierarchy."""
from __future__ import annotations

from enum import Enum
from typing import Any, Optional


class ExecutionErrorKind(str, Enum):
    """错误分类。"""
    VALIDATION = "validation"
    MAPPING = "mapping"
    VENUE = "venue"
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    INSUFFICIENT_BALANCE = "insufficient_balance"
    DUPLICATE = "duplicate"
    INTERNAL = "internal"


class ExecutionError(RuntimeError):
    """执行层基础错误。"""

    def __init__(
        self,
        message: str,
        *,
        kind: ExecutionErrorKind = ExecutionErrorKind.INTERNAL,
        venue: Optional[str] = None,
        symbol: Optional[str] = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.venue = venue
        self.symbol = symbol
        self.retryable = retryable


class ValidationError(ExecutionError):
    """模型验证失败。"""

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message, kind=ExecutionErrorKind.VALIDATION, retryable=False, **kwargs)


class MappingError(ExecutionError):
    """交易所字段映射失败。"""

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message, kind=ExecutionErrorKind.MAPPING, retryable=False, **kwargs)


class VenueError(ExecutionError):
    """交易所返回的错误。"""

    def __init__(self, message: str, *, retryable: bool = False, **kwargs: Any) -> None:
        super().__init__(message, kind=ExecutionErrorKind.VENUE, retryable=retryable, **kwargs)


class InsufficientBalanceError(ExecutionError):
    """余额不足。"""

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(
            message, kind=ExecutionErrorKind.INSUFFICIENT_BALANCE,
            retryable=False, **kwargs,
        )


class DuplicateError(ExecutionError):
    """重复提交。"""

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message, kind=ExecutionErrorKind.DUPLICATE, retryable=False, **kwargs)
