from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal

from state.snapshot import StateSnapshot
from decision.types import OrderSpec


@dataclass(frozen=True, slots=True)
class PassivePolicy:
    offset_bps: Decimal = Decimal("5")
    name: str = "passive"

    def apply(self, snapshot: StateSnapshot, order: OrderSpec) -> OrderSpec:
        m = snapshot.market
        px = Decimal(str(getattr(m, "close", None) or getattr(m, "last_price", None)))
        bps = self.offset_bps / Decimal("10000")
        if order.side == "buy":
            price = px * (Decimal("1") - bps)
        else:
            price = px * (Decimal("1") + bps)
        return replace(order, price=price, order_type="limit")
