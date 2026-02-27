# execution/adapters/bitget/mapper_balance.py
"""Map Bitget account payloads to canonical CanonicalBalance."""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any, Mapping, Sequence

from execution.models.balances import CanonicalBalance
from execution.adapters.bitget.schemas import (
    BAL_MARGIN_COIN,
    BAL_AVAILABLE,
    BAL_LOCKED,
    BAL_EQUITY,
)


def _dec(x: Any) -> Decimal:
    if x is None:
        return Decimal("0")
    try:
        return Decimal(str(x))
    except (InvalidOperation, ValueError):
        return Decimal("0")


def map_balance(raw: Mapping[str, Any]) -> CanonicalBalance:
    """Map single Bitget account entry to CanonicalBalance."""
    asset = str(raw.get(BAL_MARGIN_COIN, "")).upper()
    free = _dec(raw.get(BAL_AVAILABLE, 0))
    locked = _dec(raw.get(BAL_LOCKED, 0))
    # Prefer equity as total if available, otherwise free + locked
    equity = _dec(raw.get(BAL_EQUITY, 0))
    total = equity if equity > 0 else free + locked
    return CanonicalBalance(
        venue="bitget",
        asset=asset,
        free=free,
        locked=max(Decimal("0"), locked),
        total=total,
    )


def map_balances(raws: Sequence[Mapping[str, Any]]) -> list[CanonicalBalance]:
    """Batch map balances."""
    return [map_balance(r) for r in raws]
