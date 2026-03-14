from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from decision.types import OrderSpec
from decision.errors import PolicyViolation

from _quant_hotpath import rust_validate_order_constraints as _rust_validate_order_constraints


@dataclass(frozen=True, slots=True)
class IntentValidator:
    min_notional: Decimal = Decimal("0")
    min_qty: Decimal = Decimal("0")

    def validate(self, order: OrderSpec, *, price_hint: Decimal | None = None) -> None:
        err = _rust_validate_order_constraints(
            str(order.qty),
            (str(order.price) if order.price is not None else None),
            (str(price_hint) if price_hint is not None else None),
            str(self.min_qty),
            str(self.min_notional),
        )
        if err is not None:
            raise PolicyViolation(str(err))
