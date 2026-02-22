# execution/state_machine/reconciliation_rules.py
"""Rules for reconciling expected vs actual order state."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Optional

from execution.state_machine.transitions import OrderStatus


class DriftSeverity(str, Enum):
    NONE = "none"
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass(frozen=True, slots=True)
class ReconciliationResult:
    """对账结果。"""
    order_id: str
    severity: DriftSeverity
    status_match: bool
    qty_match: bool
    filled_qty_match: bool
    detail: str = ""


def reconcile_order(
    *,
    order_id: str,
    expected_status: OrderStatus,
    actual_status: OrderStatus,
    expected_qty: Decimal,
    actual_qty: Decimal,
    expected_filled: Decimal,
    actual_filled: Decimal,
    qty_tolerance: Decimal = Decimal("0"),
) -> ReconciliationResult:
    """
    对比预期 vs 实际订单状态。

    返回包含差异严重程度的 ReconciliationResult。
    """
    status_match = expected_status == actual_status
    qty_match = abs(expected_qty - actual_qty) <= qty_tolerance
    filled_match = abs(expected_filled - actual_filled) <= qty_tolerance

    details: list[str] = []

    if not status_match:
        details.append(f"status: expected={expected_status.value} actual={actual_status.value}")
    if not qty_match:
        details.append(f"qty: expected={expected_qty} actual={actual_qty}")
    if not filled_match:
        details.append(f"filled: expected={expected_filled} actual={actual_filled}")

    if not details:
        return ReconciliationResult(
            order_id=order_id, severity=DriftSeverity.NONE,
            status_match=True, qty_match=True, filled_qty_match=True,
        )

    # 严重度判定
    if not status_match and actual_status in (OrderStatus.FILLED, OrderStatus.CANCELED):
        severity = DriftSeverity.CRITICAL  # 终态不一致是关键问题
    elif not filled_match:
        severity = DriftSeverity.WARNING   # 成交量不一致需要关注
    else:
        severity = DriftSeverity.INFO

    return ReconciliationResult(
        order_id=order_id,
        severity=severity,
        status_match=status_match,
        qty_match=qty_match,
        filled_qty_match=filled_match,
        detail="; ".join(details),
    )
