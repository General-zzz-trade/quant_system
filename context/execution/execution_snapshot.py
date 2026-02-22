# context/execution/execution_snapshot.py
"""Execution state snapshot — read-only view of execution status."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True, slots=True)
class ActiveOrderInfo:
    """活跃订单信息摘要。"""
    order_id: str
    symbol: str
    side: str
    status: str
    qty: str
    filled_qty: str


@dataclass(frozen=True, slots=True)
class ExecutionSnapshot:
    """执行状态只读快照。"""
    active_order_count: int
    pending_intent_count: int
    total_orders_submitted: int
    total_fills_received: int
    active_orders: tuple[ActiveOrderInfo, ...] = ()

    @property
    def has_active_orders(self) -> bool:
        return self.active_order_count > 0
