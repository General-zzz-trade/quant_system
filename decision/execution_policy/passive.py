from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal

from _quant_hotpath import rust_limit_price as _rust_limit_price

from decision.market_access import get_decimal_attr
from state.snapshot import StateSnapshot
from decision.types import OrderSpec


@dataclass(frozen=True, slots=True)
class PassivePolicy:
    offset_bps: Decimal = Decimal("5")
    name: str = "passive"

    def apply(self, snapshot: StateSnapshot, order: OrderSpec) -> OrderSpec:
        m = snapshot.market
        px = get_decimal_attr(m, "close", "last_price")
        if px is None:
            raise ValueError("No market price available for passive policy")
        price = Decimal(str(_rust_limit_price(order.side, str(px), str(self.offset_bps), False)))
        return replace(order, price=price, order_type="limit")
