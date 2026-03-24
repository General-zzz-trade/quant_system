from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from _quant_hotpath import rust_build_delta_order_fields as _rust_build_delta_order_fields

from state.snapshot import StateSnapshot
from decision.types import TargetPosition, OrderSpec


@dataclass(frozen=True, slots=True)
class TargetPositionIntentBuilder:
    """Convert target position into a single order (delta vs current position)."""
    order_type: str = "limit"
    tif: str = "GTC"

    def build(self, snapshot: StateSnapshot, target: TargetPosition) -> OrderSpec | None:
        pos = snapshot.positions.get(target.symbol)
        # Rust PositionState stores qty as i64 (Fd8); use qty_f for Decimal-string interface
        if pos is not None:
            cur_f = getattr(pos, "qty_f", None)
            cur = cur_f if cur_f is not None else pos.qty
        else:
            cur = Decimal("0")
        built = _rust_build_delta_order_fields(str(target.target_qty), str(cur))
        if built is None:
            return None
        side, qty_raw = built
        qty = Decimal(str(qty_raw))
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
