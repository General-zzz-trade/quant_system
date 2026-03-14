# execution/state_machine/invariants.py
"""State machine invariants — assertions that must hold at all times."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

from execution.state_machine.transitions import OrderStatus


class InvariantViolation(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class InvariantCheckResult:
    passed: bool
    violations: tuple[str, ...] = ()


def check_order_invariants(
    *,
    status: OrderStatus,
    qty: Decimal,
    filled_qty: Decimal,
    price: Optional[Decimal],
    order_type: str,
) -> InvariantCheckResult:
    """检查单个订单的不变量。"""
    violations: list[str] = []

    # 1. qty 必须 > 0
    if qty <= 0:
        violations.append(f"qty must be > 0, got {qty}")

    # 2. filled_qty 不能超过 qty
    if filled_qty > qty:
        violations.append(f"filled_qty({filled_qty}) > qty({qty})")

    # 3. filled_qty 不能为负
    if filled_qty < 0:
        violations.append(f"filled_qty must be >= 0, got {filled_qty}")

    # 4. FILLED 状态时 filled_qty == qty
    if status == OrderStatus.FILLED and filled_qty != qty:
        violations.append(f"FILLED but filled_qty({filled_qty}) != qty({qty})")

    # 5. limit 订单必须有 price
    if order_type == "limit" and price is None:
        violations.append("limit order must have price")

    # 6. price 如果有值必须 > 0
    if price is not None and price <= 0:
        violations.append(f"price must be > 0, got {price}")

    return InvariantCheckResult(
        passed=len(violations) == 0,
        violations=tuple(violations),
    )


def assert_order_invariants(**kwargs) -> None:
    """同 check_order_invariants，但不通过时直接抛异常。"""
    result = check_order_invariants(**kwargs)
    if not result.passed:
        raise InvariantViolation(
            f"Order invariant violations: {'; '.join(result.violations)}"
        )
