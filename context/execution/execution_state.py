# context/execution/execution_state.py
"""Execution state — tracks active orders and intents within Context."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

from context.execution.execution_snapshot import ExecutionSnapshot, ActiveOrderInfo


class ExecutionState:
    """
    执行状态追踪。

    维护活跃订单、待处理 intent 的计数和摘要。
    仅由 Reducer 更新。
    """

    def __init__(self) -> None:
        self._active_orders: Dict[str, Dict[str, str]] = {}  # order_id -> info
        self._pending_intents: int = 0
        self._total_submitted: int = 0
        self._total_fills: int = 0

    def record_order_submitted(self, order_id: str, symbol: str, side: str, qty: str) -> None:
        self._active_orders[order_id] = {
            "order_id": order_id, "symbol": symbol, "side": side,
            "status": "new", "qty": qty, "filled_qty": "0",
        }
        self._total_submitted += 1

    def record_order_update(self, order_id: str, status: str, filled_qty: str = "0") -> None:
        info = self._active_orders.get(order_id)
        if info:
            info["status"] = status
            info["filled_qty"] = filled_qty
            if status in ("filled", "canceled", "rejected", "expired"):
                del self._active_orders[order_id]

    def record_fill(self) -> None:
        self._total_fills += 1

    def snapshot(self) -> ExecutionSnapshot:
        orders = tuple(
            ActiveOrderInfo(**info) for info in self._active_orders.values()
        )
        return ExecutionSnapshot(
            active_order_count=len(self._active_orders),
            pending_intent_count=self._pending_intents,
            total_orders_submitted=self._total_submitted,
            total_fills_received=self._total_fills,
            active_orders=orders,
        )
