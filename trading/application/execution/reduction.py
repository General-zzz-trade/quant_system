"""Reduction sizing: from the venue position, never above what the owner confirmed (moved in step 9c).

Moved from ``BinanceExecutionAdapter._send_order``. A reduction that cannot be matched to exactly one
venue position, that would not reduce it, or that this owner has no confirmed share of raises
``ValueError``; the caller then sends nothing.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from trading.application.ports import ExchangeAccount, OrderJournal


@dataclass(frozen=True)
class Reduction:
    qty: Decimal
    side: str
    consumer_revision: str | None    # the owner's checkpoint revision the reduction was sized from


def size_reduction(account: ExchangeAccount, journal: OrderJournal, *, owner: str, symbol: str, side: str,
                   qty: Decimal) -> Reduction:
    """``qty == 0`` closes the whole owned position, whatever ``side`` says."""
    positions = [p for p in account.get_positions(symbol=symbol, strict=True)
                 if p.symbol == symbol and Decimal(str(p.qty)) != 0]
    if len(positions) != 1:
        raise ValueError("Reduction requires one unambiguous venue position")
    actual = Decimal(str(positions[0].qty))
    if qty == 0:
        qty, side = abs(actual), "sell" if actual > 0 else "buy"
    if side != ("sell" if actual > 0 else "buy"):
        raise ValueError("Reduction side does not reduce venue position")
    qty = min(qty, abs(actual))
    consumer_revision = None
    if owner:
        checkpoint = journal.checkpoint(owner)
        owned = Decimal(checkpoint["position"]["qty"]) if checkpoint else Decimal(0)
        if not checkpoint or not owned.is_finite() or owned * actual <= 0:
            raise ValueError("No confirmed position owned by this strategy")
        qty = min(qty, abs(owned))
        consumer_revision = checkpoint["revision"]
    return Reduction(qty, side, consumer_revision)
