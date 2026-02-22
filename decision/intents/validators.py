from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Sequence

from decision.types import OrderSpec
from decision.errors import PolicyViolation


@dataclass(frozen=True, slots=True)
class IntentValidator:
    min_notional: Decimal = Decimal("0")
    min_qty: Decimal = Decimal("0")

    def validate(self, order: OrderSpec, *, price_hint: Decimal | None = None) -> None:
        if order.qty <= 0:
            raise PolicyViolation("order.qty must be > 0")
        if self.min_qty and order.qty < self.min_qty:
            raise PolicyViolation(f"order.qty < min_qty ({order.qty} < {self.min_qty})")
        p = order.price if order.price is not None else price_hint
        if p is not None:
            if p <= 0:
                raise PolicyViolation("order.price must be > 0 when provided")
            if self.min_notional and (order.qty * p) < self.min_notional:
                raise PolicyViolation("order.notional below min_notional")
