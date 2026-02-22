# execution/adapters/binance/error_map.py
"""Binance error code mapping."""
from __future__ import annotations

from enum import Enum
from typing import Mapping


class BinanceErrorAction(Enum):
    RETRY = "retry"
    REJECT = "reject"
    HALT = "halt"
    IGNORE = "ignore"


# Binance UM Futures 常见错误码
BINANCE_ERROR_MAP: Mapping[int, tuple[BinanceErrorAction, str]] = {
    -1000: (BinanceErrorAction.RETRY, "unknown error"),
    -1001: (BinanceErrorAction.RETRY, "disconnected"),
    -1002: (BinanceErrorAction.HALT, "unauthorized"),
    -1003: (BinanceErrorAction.RETRY, "too many requests"),
    -1006: (BinanceErrorAction.RETRY, "unexpected response"),
    -1007: (BinanceErrorAction.RETRY, "timeout"),
    -1010: (BinanceErrorAction.REJECT, "received error message"),
    -1013: (BinanceErrorAction.REJECT, "invalid quantity"),
    -1014: (BinanceErrorAction.REJECT, "unknown order composition"),
    -1015: (BinanceErrorAction.RETRY, "too many orders"),
    -1021: (BinanceErrorAction.RETRY, "timestamp outside recvWindow"),
    -1022: (BinanceErrorAction.HALT, "invalid signature"),
    -1100: (BinanceErrorAction.REJECT, "illegal characters"),
    -1111: (BinanceErrorAction.REJECT, "precision over max"),
    -1116: (BinanceErrorAction.REJECT, "invalid order type"),
    -1121: (BinanceErrorAction.REJECT, "invalid symbol"),
    -2010: (BinanceErrorAction.REJECT, "new order rejected"),
    -2011: (BinanceErrorAction.REJECT, "cancel rejected"),
    -2013: (BinanceErrorAction.REJECT, "order does not exist"),
    -2014: (BinanceErrorAction.HALT, "API key format invalid"),
    -2015: (BinanceErrorAction.HALT, "invalid API key/IP/permissions"),
    -2018: (BinanceErrorAction.REJECT, "balance insufficient"),
    -2019: (BinanceErrorAction.REJECT, "margin insufficient"),
    -2022: (BinanceErrorAction.REJECT, "reduce only rejected"),
    -4003: (BinanceErrorAction.REJECT, "quantity less than zero"),
    -4014: (BinanceErrorAction.REJECT, "price not increased by tick size"),
    -4028: (BinanceErrorAction.RETRY, "leverage too high"),
    -5021: (BinanceErrorAction.REJECT, "order would immediately trigger"),
}


def classify_error(code: int) -> tuple[BinanceErrorAction, str]:
    """根据 Binance 错误码返回处理策略。"""
    return BINANCE_ERROR_MAP.get(code, (BinanceErrorAction.REJECT, f"unknown code {code}"))
