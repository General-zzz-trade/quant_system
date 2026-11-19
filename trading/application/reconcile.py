"""Read-only account reconciliation for one symbol (moved from BinanceExecutionAdapter in step 9c).

Every mismatch between the venue and the durable journal is returned as an issue; the caller blocks
entries while any issue remains. Nothing here places, cancels or fabricates anything.
"""
from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal
from typing import Any

from trading.application.issues import Issue
from trading.application.ports import ExchangeAccount, OrderJournal, PositionView
from trading.domain.issues import IssueKind as K

UNRESOLVED_ORDERS = "unresolved orders or legacy journal records"


def booked_position(journal: OrderJournal, rows: list[dict[str, Any]], symbol: str, owner: str | None) -> Decimal:
    """The ledger's position in ``symbol``: the checkpoints of every owner that traded it, and always ``owner``'s
    own (a position set by hand on a symbol never traded has a checkpoint but no order row). The one definition
    reconciliation and the start-up account check share (ADR-0010 audit, finding 5)."""
    owners = {row["owner"] for row in rows if row["symbol"] == symbol}
    if owner:
        owners.add(owner)
    total = Decimal(0)
    for name in sorted(owners):
        checkpoint = journal.checkpoint(name)
        if checkpoint:
            total += Decimal(str(checkpoint["position"]["qty"]))
    return total


def reconcile_account(account: ExchangeAccount, journal: OrderJournal, *, owner: str, symbol: str,
                      positions: Mapping[str, PositionView] | None = None, venue_reads: bool = True) -> list[Issue]:
    """``positions``: the account's position rows read once for every symbol this cycle (plan R5); a symbol without
    a row there, or no snapshot, is read alone. ``venue_reads`` False (plan D2: an idle symbol) sends nothing of the
    symbol's own: its open orders are left to the whole-account check, and a position not in ``positions`` is not
    read, nor compared (the caller blocks every entry while it is unknown). The journal checks run either way."""
    issues: list[Issue] = []

    def add(kind: K, text: str) -> None:
        issues.append(Issue(kind, symbol, f"{symbol}: {text}"))

    try:
        rows = [positions[symbol]] if positions is not None and symbol in positions \
            else account.get_positions(symbol=symbol, strict=True) if venue_reads else None
        if rows is not None:
            nonflat = [p for p in rows if p.symbol == symbol and Decimal(str(p.qty)) != 0]
            if len(nonflat) > 1:
                add(K.LEDGER_MISMATCH, "multiple positions require explicit reconciliation")
            actual = sum((Decimal(str(p.qty)) for p in nonflat), Decimal(0))
            expected = booked_position(journal, journal.orders(), symbol, owner)
            if not actual.is_finite() or not expected.is_finite() or actual != expected:
                add(K.LEDGER_MISMATCH, f"venue qty={actual}, ledger qty={expected}")
        if any(row["symbol"] == symbol for row in journal.orders(unresolved=True)):
            add(K.UNRESOLVED_ORDER, UNRESOLVED_ORDERS)
        if venue_reads:
            known = {row["order_id"] for row in journal.orders() if row["symbol"] == symbol}
            for order in account.get_open_orders(symbol=symbol, strict=True):
                if order.order_id not in known:
                    add(K.FOREIGN_HOLDING, f"unowned open order {order.order_id}")
        if journal.fills(owner):
            add(K.UNDELIVERED_FILL, "undelivered fills")
    except Exception as exc:
        add(K.RECONCILE_FAILED, f"reconciliation failed ({type(exc).__name__})")
    return issues
