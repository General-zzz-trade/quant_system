# execution/state_machine/machine.py
"""Order state machine backed by Rust.

Note: OrderStateMachine is an execution lifecycle tracker, NOT a position truth source.
Position state is owned by RustStateStore in the pipeline. OSM provides order-level
audit trail for timeout detection, reconciliation, and operational logging.
The only decision-path read is RiskGate.get_open_order_count (execution safety),
which checks active_orders() to enforce max_open_orders limits.
No signal generation or position sizing should depend on OSM state.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Optional, Sequence

from execution.state_machine.transitions import OrderStatus, Transition

from _quant_hotpath import RustOrderStateMachine as _RustOrderStateMachine


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
        return self.status in (
            OrderStatus.FILLED,
            OrderStatus.CANCELED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
        )


def _status_from_any(value: object) -> OrderStatus:
    raw = getattr(value, "value", value)
    return OrderStatus(str(raw).lower())


def _dec_opt(value: object) -> Optional[Decimal]:
    if value is None:
        return None
    return Decimal(str(value))


def _transition_from_rust(raw: object) -> Transition:
    return Transition(
        from_status=_status_from_any(getattr(raw, "from_status")),
        to_status=_status_from_any(getattr(raw, "to_status")),
        ts_ms=int(getattr(raw, "ts_ms", 0)),
        reason=str(getattr(raw, "reason", "")),
    )


def _state_from_rust(raw: object) -> OrderState:
    return OrderState(
        order_id=str(getattr(raw, "order_id")),
        client_order_id=getattr(raw, "client_order_id", None),
        symbol=str(getattr(raw, "symbol")),
        side=str(getattr(raw, "side")),
        order_type=str(getattr(raw, "order_type")),
        status=_status_from_any(getattr(raw, "status")),
        qty=Decimal(str(getattr(raw, "qty"))),
        price=_dec_opt(getattr(raw, "price", None)),
        filled_qty=Decimal(str(getattr(raw, "filled_qty", "0"))),
        avg_price=_dec_opt(getattr(raw, "avg_price", None)),
        last_update_ts=int(getattr(raw, "last_update_ts", 0)),
        transitions=[_transition_from_rust(t) for t in getattr(raw, "transitions", ())],
    )


class OrderStateMachine:
    """Rust-backed order state machine with Python compatibility wrappers."""

    def __init__(self) -> None:
        self._rust = _RustOrderStateMachine()

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
        try:
            raw = self._rust.register(
                order_id=str(order_id),
                client_order_id=client_order_id,
                symbol=str(symbol),
                side=str(side),
                order_type=str(order_type),
                qty=str(qty),
                price=(str(price) if price is not None else None),
            )
        except RuntimeError as exc:
            raise OrderStateMachineError(str(exc)) from exc
        return _state_from_rust(raw)

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
        status = _status_from_any(new_status)
        try:
            raw = self._rust.transition(
                order_id=str(order_id),
                new_status=status.value,
                filled_qty=(str(filled_qty) if filled_qty is not None else None),
                avg_price=(str(avg_price) if avg_price is not None else None),
                ts_ms=int(ts_ms),
                reason=str(reason),
            )
        except RuntimeError as exc:
            msg = str(exc)
            if "not allowed" in msg or "unknown order status" in msg:
                raise InvalidTransitionError(msg) from exc
            raise OrderStateMachineError(msg) from exc
        return _state_from_rust(raw)

    def get(self, order_id: str) -> Optional[OrderState]:
        raw = self._rust.get(str(order_id))
        if raw is None:
            return None
        return _state_from_rust(raw)

    def active_orders(self) -> Sequence[OrderState]:
        return [_state_from_rust(state) for state in self._rust.active_orders()]

    def active_count(self) -> int:
        return int(self._rust.active_count())
