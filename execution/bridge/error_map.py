# execution/bridge/error_map.py
"""Map venue error codes to execution error categories."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict


class ErrorCategory(str, Enum):
    """错误分类 — 决定后续处理策略。"""
    RETRYABLE = "retryable"
    NON_RETRYABLE = "non_retryable"
    INSUFFICIENT_BALANCE = "insufficient_balance"
    RATE_LIMITED = "rate_limited"
    INVALID_PARAMS = "invalid_params"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class ErrorMapping:
    """单条错误映射规则。"""
    venue_code: int
    venue_message_pattern: str  # 空字符串匹配任意
    category: ErrorCategory
    retryable: bool


# Binance 常见错误码映射
BINANCE_ERROR_MAP: Dict[int, ErrorMapping] = {
    -1000: ErrorMapping(-1000, "", ErrorCategory.UNKNOWN, True),
    -1001: ErrorMapping(-1001, "", ErrorCategory.RETRYABLE, True),       # 内部错误
    -1003: ErrorMapping(-1003, "", ErrorCategory.RATE_LIMITED, False),    # 限速
    -1015: ErrorMapping(-1015, "", ErrorCategory.RATE_LIMITED, False),    # 下单限速
    -1021: ErrorMapping(-1021, "", ErrorCategory.RETRYABLE, True),       # 时间同步
    -2010: ErrorMapping(-2010, "", ErrorCategory.NON_RETRYABLE, False),  # 订单创建失败
    -2011: ErrorMapping(-2011, "", ErrorCategory.NON_RETRYABLE, False),  # 订单撤销失败
    -2015: ErrorMapping(-2015, "", ErrorCategory.NON_RETRYABLE, False),  # API key 无效
    -2019: ErrorMapping(-2019, "", ErrorCategory.INSUFFICIENT_BALANCE, False),
    -4003: ErrorMapping(-4003, "", ErrorCategory.INVALID_PARAMS, False),
    -4014: ErrorMapping(-4014, "", ErrorCategory.INVALID_PARAMS, False),  # 价格精度
    -4015: ErrorMapping(-4015, "", ErrorCategory.INVALID_PARAMS, False),  # 数量精度
}


class ErrorMapper:
    """交易所错误码映射器。"""

    def __init__(self, venue: str = "binance") -> None:
        self._venue = venue
        self._map: Dict[int, ErrorMapping] = {}
        if venue == "binance":
            self._map = dict(BINANCE_ERROR_MAP)

    def classify(self, code: int, message: str = "") -> ErrorMapping:
        """将交易所错误码映射为标准分类。"""
        mapping = self._map.get(code)
        if mapping is not None:
            return mapping
        # HTTP 状态码通用分类
        if 400 <= code < 500:
            return ErrorMapping(code, message, ErrorCategory.NON_RETRYABLE, False)
        if 500 <= code < 600:
            return ErrorMapping(code, message, ErrorCategory.RETRYABLE, True)
        return ErrorMapping(code, message, ErrorCategory.UNKNOWN, False)

    def is_retryable(self, code: int) -> bool:
        return self.classify(code).retryable

    def register(self, mapping: ErrorMapping) -> None:
        self._map[mapping.venue_code] = mapping
