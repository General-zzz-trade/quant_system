# execution/adapters/bitget/mapper_position.py
"""Map Bitget position payloads to canonical VenuePosition."""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any, Mapping, Sequence

from execution.models.positions import VenuePosition
from execution.adapters.bitget.schemas import (
    POS_SYMBOL,
    POS_TOTAL,
    POS_HOLD_SIDE,
    POS_ENTRY_PRICE,
    POS_UNREALIZED,
    POS_LEVERAGE,
    POS_MARGIN_MODE,
)


def _dec(x: Any) -> Decimal:
    if x is None:
        return Decimal("0")
    try:
        return Decimal(str(x))
    except (InvalidOperation, ValueError):
        return Decimal("0")


def map_position(raw: Mapping[str, Any]) -> VenuePosition:
    """Map single Bitget position to VenuePosition.

    Bitget uses holdSide (long/short) + unsigned total qty.
    Convert to signed qty: short → negative.
    """
    qty_unsigned = _dec(raw.get(POS_TOTAL, "0"))
    hold_side = str(raw.get(POS_HOLD_SIDE, "")).lower()
    qty = -qty_unsigned if hold_side == "short" else qty_unsigned

    margin_mode = str(raw.get(POS_MARGIN_MODE, "")).lower() or None

    return VenuePosition(
        venue="bitget",
        symbol=str(raw.get(POS_SYMBOL, "")).upper(),
        qty=qty,
        entry_price=_dec(raw.get(POS_ENTRY_PRICE)),
        unrealized_pnl=_dec(raw.get(POS_UNREALIZED)),
        leverage=int(_dec(raw.get(POS_LEVERAGE, 1))),
        margin_type=margin_mode,
    )


def map_positions(raws: Sequence[Mapping[str, Any]]) -> list[VenuePosition]:
    """Batch map positions, filtering zero positions."""
    result = []
    for r in raws:
        pos = map_position(r)
        if pos.qty != 0:
            result.append(pos)
    return result
