from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from state.snapshot import StateSnapshot
from decision.types import TargetPosition, OrderSpec


@dataclass(frozen=True, slots=True)
class TargetPositionIntentBuilder:
    """Convert target position into a single order (delta vs current position)."""
    order_type: str = "limit"
    tif: str = "GTC"

    def build(self, snapshot: StateSnapshot, target: TargetPosition) -> OrderSpec | None:
        pos = snapshot.positions.get(target.symbol)
        cur = pos.qty if pos is not None else Decimal("0")
        delta = target.target_qty - cur
        if delta == 0:
            return None
        side = "buy" if delta > 0 else "sell"
        qty = abs(delta)
        # price decided by execution policy later
        return OrderSpec(
            order_id="",
            intent_id="",
            symbol=target.symbol,
            side=side,
            qty=qty,
            order_type=self.order_type,  # type: ignore[arg-type]
            price=None,
            tif=self.tif,
            meta={"reason_code": target.reason_code, "origin": target.origin},
        )
