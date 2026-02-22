# execution/state_machine/machine.py
"""Order state machine — tracks order lifecycle and enforces valid transitions."""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from threading import RLock
from typing import Any, Dict, Mapping, Optional, Sequence

from execution.state_machine.transitions import OrderStatus, Transition, VALID_TRANSITIONS


class OrderStateMachineError(RuntimeError):
    pass


class InvalidTransitionError(OrderStateMachineError):
    pass


@dataclass
class OrderState:
    """可变的订单实时状态（仅在 state machine 内部修改）。"""
    order_id: str
    client_order_id: Optional[str]
    symbol: str
    side: str
    order_type: str
    status: OrderStatus = OrderStatus.PENDING_NEW
    qty: Decimal = Decimal("0")
    price: Optional[Decimal] = None
    filled_qty: Decimal = Decimal("0")
    avg_price: Optional[Decimal] = None
    last_update_ts: int = 0
    transitions: list[Transition] = field(default_factory=list)

    @property
    def remaining_qty(self) -> Decimal:
        return self.qty - self.filled_qty

    @property
    def is_terminal(self) -> bool:
        return self.status in (OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED, OrderStatus.EXPIRED)


class OrderStateMachine:
    """
    订单状态机

    管理所有活跃订单的生命周期：
    - 只允许有效的状态转换
    - 追踪所有状态变更历史
    - 在终态时自动归档
    """

    def __init__(self) -> None:
        self._lock = RLock()
        self._orders: Dict[str, OrderState] = {}
        self._archived: Dict[str, OrderState] = {}

    def register(
        self,
        *,
        order_id: str,
        client_order_id: Optional[str] = None,
        symbol: str,
        side: str,
        order_type: str,
        qty: Decimal,
        price: Optional[Decimal] = None,
    ) -> OrderState:
        with self._lock:
            if order_id in self._orders:
                raise OrderStateMachineError(f"order {order_id} already registered")
            state = OrderState(
                order_id=order_id,
                client_order_id=client_order_id,
                symbol=symbol,
                side=side,
                order_type=order_type,
                qty=qty,
                price=price,
                status=OrderStatus.PENDING_NEW,
            )
            self._orders[order_id] = state
            return state

    def transition(
        self,
        *,
        order_id: str,
        new_status: OrderStatus,
        filled_qty: Optional[Decimal] = None,
        avg_price: Optional[Decimal] = None,
        ts_ms: int = 0,
        reason: str = "",
    ) -> OrderState:
        with self._lock:
            state = self._orders.get(order_id)
            if state is None:
                raise OrderStateMachineError(f"unknown order: {order_id}")

            old_status = state.status
            allowed = VALID_TRANSITIONS.get(old_status, set())
            if new_status not in allowed:
                raise InvalidTransitionError(
                    f"order {order_id}: {old_status.value} → {new_status.value} not allowed"
                )

            t = Transition(
                from_status=old_status,
                to_status=new_status,
                ts_ms=ts_ms,
                reason=reason,
            )
            state.status = new_status
            state.transitions.append(t)
            if filled_qty is not None:
                state.filled_qty = filled_qty
            if avg_price is not None:
                state.avg_price = avg_price
            state.last_update_ts = ts_ms

            if state.is_terminal:
                self._archived[order_id] = self._orders.pop(order_id)

            return state

    def get(self, order_id: str) -> Optional[OrderState]:
        with self._lock:
            return self._orders.get(order_id) or self._archived.get(order_id)

    def active_orders(self) -> Sequence[OrderState]:
        with self._lock:
            return list(self._orders.values())

    def active_count(self) -> int:
        with self._lock:
            return len(self._orders)
