# execution/adapters/binance/mapper_balance.py
"""Map Binance balance payloads to canonical CanonicalBalance."""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any, Mapping, Sequence

from execution.models.balances import CanonicalBalance
from execution.adapters.binance.schemas import (
    BAL_ASSET,
    BAL_AVAILABLE,
    BAL_BALANCE,
)


def _dec(x: Any, field: str) -> Decimal:
    if x is None:
        return Decimal("0")
    try:
        return Decimal(str(x))
    except (InvalidOperation, ValueError):
        return Decimal("0")


def map_balance(raw: Mapping[str, Any]) -> CanonicalBalance:
    """单条余额映射。"""
    asset = str(raw.get(BAL_ASSET, ""))
    total = _dec(raw.get(BAL_BALANCE, 0), "balance")
    free = _dec(raw.get(BAL_AVAILABLE, 0), "available")
    locked = total - free
    return CanonicalBalance(
        venue="binance",
        asset=asset,
        free=free,
        locked=max(Decimal("0"), locked),
        total=total,
    )


def map_balances(raws: Sequence[Mapping[str, Any]]) -> list[CanonicalBalance]:
    """批量映射余额。"""
    return [map_balance(r) for r in raws]
