"""Order events carrying an explicit reduce_only flag (gates and executors read it)."""
from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal

from event.header import EventHeader
from event.types import EventType, OrderEvent


@dataclass(frozen=True, slots=True)
class TargetOrderEvent(OrderEvent):
    reduce_only: bool = False


def make_order(*, order_id: str, source: str, symbol: str, side: str, qty: Decimal, price: Decimal,
               reduce_only: bool, reason: str) -> TargetOrderEvent:
    header = EventHeader.new_root(event_type=EventType.ORDER, version=1, source=source)
    header = replace(header, event_id=order_id)
    return TargetOrderEvent(header=header, order_id=order_id, intent_id=order_id, symbol=symbol, side=side,
                            qty=qty, price=price, reason=reason, reduce_only=reduce_only)
