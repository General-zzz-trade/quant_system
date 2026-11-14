"""Read-only account reconciliation for one symbol (moved from BinanceExecutionAdapter in step 9c).

Every mismatch between the venue and the durable journal is returned as an issue; the caller blocks
entries while any issue remains. Nothing here places, cancels or fabricates anything.
"""
from __future__ import annotations

from decimal import Decimal

from trading.application.ports import ExchangeAccount, OrderJournal


def reconcile_account(account: ExchangeAccount, journal: OrderJournal, *, owner: str, symbol: str) -> list[str]:
    issues: list[str] = []
    try:
        positions = account.get_positions(symbol=symbol, strict=True)
        nonflat = [p for p in positions if p.symbol == symbol and Decimal(str(p.qty)) != 0]
        if len(nonflat) > 1:
            issues.append(f"{symbol}: multiple positions require explicit reconciliation")
        actual = sum((Decimal(str(p.qty)) for p in nonflat), Decimal(0))
        expected = Decimal(0)
        for other in {r["owner"] for r in journal.orders() if r["symbol"] == symbol}:
            checkpoint = journal.checkpoint(other)
            if checkpoint:
                expected += Decimal(checkpoint["position"]["qty"])
        if not actual.is_finite() or not expected.is_finite() or actual != expected:
            issues.append(f"{symbol}: venue qty={actual}, ledger qty={expected}")
        if any(row["symbol"] == symbol for row in journal.orders(unresolved=True)):
            issues.append(f"{symbol}: unresolved orders or legacy journal records")
        known = {row["order_id"] for row in journal.orders() if row["symbol"] == symbol}
        for order in account.get_open_orders(symbol=symbol, strict=True):
            if order.order_id not in known:
                issues.append(f"{symbol}: unowned open order {order.order_id}")
        if journal.fills(owner):
            issues.append(f"{symbol}: undelivered fills")
    except Exception as exc:
        issues.append(f"{symbol}: reconciliation failed ({type(exc).__name__})")
    return issues
