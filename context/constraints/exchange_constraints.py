# context/constraints/exchange_constraints.py
"""Exchange-level constraints — tick size, lot size, trading hours."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Optional


@dataclass(frozen=True, slots=True)
class ExchangeConstraintViolation:
    """交易所约束违规。"""
    field: str
    message: str
    expected: str
    actual: str


@dataclass(frozen=True, slots=True)
class ExchangeConstraints:
    """交易所级约束检查。"""
    venue: str = ""

    def check_price_precision(
        self, price: Decimal, tick_size: Decimal,
    ) -> Optional[ExchangeConstraintViolation]:
        if tick_size <= 0:
            return None
        remainder = price % tick_size
        if remainder != 0:
            return ExchangeConstraintViolation(
                field="price",
                message=f"price {price} not aligned to tick_size {tick_size}",
                expected=str(tick_size),
                actual=str(price),
            )
        return None

    def check_qty_precision(
        self, qty: Decimal, step_size: Decimal,
    ) -> Optional[ExchangeConstraintViolation]:
        if step_size <= 0:
            return None
        remainder = qty % step_size
        if remainder != 0:
            return ExchangeConstraintViolation(
                field="qty",
                message=f"qty {qty} not aligned to step_size {step_size}",
                expected=str(step_size),
                actual=str(qty),
            )
        return None
