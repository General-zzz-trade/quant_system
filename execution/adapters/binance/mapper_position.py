# execution/adapters/binance/mapper_position.py
"""Map Binance position payloads to canonical VenuePosition."""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any, Mapping, Sequence

from execution.models.positions import VenuePosition
from execution.adapters.binance.schemas import (
    POS_AMT,
    POS_ENTRY_PRICE,
    POS_LEVERAGE,
    POS_SIDE,
    POS_SYMBOL,
    POS_UNREALIZED,
)


def _dec(x: Any) -> Decimal:
    if x is None:
        return Decimal("0")
    try:
        return Decimal(str(x))
    except (InvalidOperation, ValueError):
        return Decimal("0")


def map_position(raw: Mapping[str, Any]) -> VenuePosition:
    """单条持仓映射。"""
    return VenuePosition(
        symbol=str(raw.get(POS_SYMBOL, "")),
        qty=_dec(raw.get(POS_AMT)),
        entry_price=_dec(raw.get(POS_ENTRY_PRICE)),
        unrealized_pnl=_dec(raw.get(POS_UNREALIZED)),
        leverage=int(_dec(raw.get(POS_LEVERAGE, 1))),
    )


def map_positions(raws: Sequence[Mapping[str, Any]]) -> list[VenuePosition]:
    """批量映射持仓, 过滤零仓位。"""
    result = []
    for r in raws:
        pos = map_position(r)
        if pos.qty != 0:
            result.append(pos)
    return result
