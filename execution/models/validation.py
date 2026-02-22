# execution/models/validation.py
"""Pre-flight validation for execution models against instrument constraints."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Optional, Sequence

from execution.models.instruments import InstrumentInfo


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """验证结果。"""
    valid: bool
    errors: tuple[str, ...] = ()

    @classmethod
    def ok(cls) -> ValidationResult:
        return cls(valid=True)

    @classmethod
    def fail(cls, *errors: str) -> ValidationResult:
        return cls(valid=False, errors=errors)

    def merge(self, other: ValidationResult) -> ValidationResult:
        if self.valid and other.valid:
            return ValidationResult.ok()
        return ValidationResult(
            valid=False,
            errors=self.errors + other.errors,
        )


def validate_order_params(
    *,
    instrument: InstrumentInfo,
    side: str,
    qty: Decimal,
    price: Optional[Decimal] = None,
    order_type: str = "limit",
) -> ValidationResult:
    """
    验证下单参数是否符合交易对约束。

    检查项：
    1. side 必须是 buy/sell
    2. qty 精度和范围
    3. price 精度（限价单必须有 price）
    4. notional 最低要求
    """
    errors: list[str] = []

    # side
    if side not in ("buy", "sell"):
        errors.append(f"invalid side: {side!r}")

    # qty
    if qty <= 0:
        errors.append(f"qty must be > 0, got {qty}")
    else:
        valid_qty, msg = instrument.validate_qty(qty)
        if not valid_qty:
            errors.append(msg)
        # 检查精度
        rounded = instrument.round_qty(qty)
        if rounded != qty:
            errors.append(f"qty {qty} not aligned to lot_size {instrument.lot_size}")

    # price
    if order_type == "limit":
        if price is None:
            errors.append("limit order requires price")
        elif price <= 0:
            errors.append(f"price must be > 0, got {price}")
        else:
            rounded_p = instrument.round_price(price)
            if rounded_p != price:
                errors.append(
                    f"price {price} not aligned to tick_size {instrument.tick_size}"
                )

    # notional
    if price is not None and qty > 0 and price > 0:
        valid_n, msg = instrument.validate_notional(qty, price)
        if not valid_n:
            errors.append(msg)

    # trading enabled
    if not instrument.trading_enabled:
        errors.append(f"trading disabled for {instrument.symbol}")

    if errors:
        return ValidationResult.fail(*errors)
    return ValidationResult.ok()


def round_order_params(
    *,
    instrument: InstrumentInfo,
    qty: Decimal,
    price: Optional[Decimal] = None,
) -> tuple[Decimal, Optional[Decimal]]:
    """
    按 instrument 精度圆整 qty 和 price（向下截断）。

    返回 (rounded_qty, rounded_price)。
    """
    rq = instrument.round_qty(qty)
    rp = instrument.round_price(price) if price is not None else None
    return rq, rp
