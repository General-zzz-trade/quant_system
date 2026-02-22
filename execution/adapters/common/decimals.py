# execution/adapters/common/decimals.py
"""Decimal parsing and rounding helpers for adapters."""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any


ZERO = Decimal("0")


def safe_decimal(value: Any, default: Decimal = ZERO) -> Decimal:
    """安全解析 Decimal — 失败返回 default。"""
    if value is None:
        return default
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return default


def require_decimal(value: Any, field_name: str) -> Decimal:
    """强制解析 Decimal — 失败抛异常。"""
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as e:
        raise ValueError(f"{field_name}: cannot parse as Decimal: {value!r}") from e


def round_down(value: Decimal, step: Decimal) -> Decimal:
    """向下圆整到 step 的整数倍。"""
    if step <= 0:
        return value
    return (value // step) * step


def round_to_precision(value: Decimal, precision: int) -> Decimal:
    """按小数位数圆整（向下截断）。"""
    if precision < 0:
        return value
    quantize_str = "1." + "0" * precision if precision > 0 else "1"
    return value.quantize(Decimal(quantize_str), rounding="ROUND_DOWN")


def is_positive(value: Decimal) -> bool:
    return value > ZERO


def clamp(value: Decimal, lo: Decimal, hi: Decimal) -> Decimal:
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value
