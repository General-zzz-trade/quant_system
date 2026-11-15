"""Does the account hold only what the strategy owns? The execution start-up check of ADR-0010.

Read-only: it compares the whole account (positions, open orders and open conditional orders on every symbol, and
the leverage and margin mode of the strategy's symbols) with the order journal and the configured settings, and
returns every finding. The caller refuses to start on any finding. Nothing here places, cancels or changes
anything.
"""
from __future__ import annotations

from decimal import Decimal
from typing import Any

from trading.application.ports import AccountSnapshot, OrderJournal


def owner_of(symbol: str) -> str:
    """The journal owner of a symbol's target session (``trading.infrastructure.target``)."""
    return f"target.{symbol}"


def foreign_holdings(account: AccountSnapshot, journal: OrderJournal, *, symbols: tuple[str, ...], leverage: int,
                     margin: str) -> list[str]:
    """Every way the account differs from the strategy ledger or the configuration; empty when it does not.

    ``margin`` is spelled as the exchange reports it: "cross" or "isolated". An account or journal that cannot
    be read is a finding too (fail closed)."""
    try:
        positions = account.get_positions(strict=True)
        orders = account.get_open_orders(strict=True)
        stops = account.get_open_algo_orders()
        settings = {symbol: account.get_symbol_settings(symbol) for symbol in symbols}
    except Exception as exc:
        return [f"account could not be read ({type(exc).__name__}: {exc})"]
    try:
        rows = journal.orders()
        ledger = {symbol: _ledger_qty(journal, rows, symbol, symbols)
                  for symbol in {*symbols, *(p.symbol for p in positions)}}
    except Exception as exc:
        return [f"strategy journal could not be read ({type(exc).__name__}: {exc})"]

    findings: list[str] = []
    known_stops: set[str] = set()
    if stops:
        try:
            known_stops = {row["client_algo_id"] for symbol in symbols
                           for row in journal.stop_rows(owner_of(symbol), symbol, live_only=False)}
        except Exception as exc:
            # A new state directory has no stop table until the runtime first runs (until ADR-0009). Unread
            # records only make every conditional order count as foreign: more refusals, never fewer.
            findings.append(f"strategy stop records could not be read ({type(exc).__name__}: {exc})")
    venue: dict[str, Decimal] = {}
    for p in positions:
        venue[p.symbol] = venue.get(p.symbol, Decimal(0)) + Decimal(str(p.qty))
    # A symbol with an unsettled strategy order is left to order recovery: after a crash between a fill and its
    # booking the venue is ahead of the ledger, and reconciliation blocks that symbol's entries until it settles.
    unsettled = {row["symbol"] for row in rows if not row["settled"]}
    for symbol in sorted(ledger):
        actual, expected = venue.get(symbol, Decimal(0)), ledger[symbol]
        if actual == expected or symbol in unsettled:
            continue
        if actual == 0:
            findings.append(f"{symbol}: the venue is flat but the strategy ledger holds {expected}")
        else:
            findings.append(f"{symbol}: position {actual} is not in the strategy ledger (ledger {expected})")
    known = {str(row[key]) for row in rows for key in ("order_id", "client_id") if row.get(key)}
    for order in orders:
        if order.order_id not in known and order.client_order_id not in known:
            findings.append(f"{order.symbol}: open order {order.order_id} is not in the strategy journal")
    for stop in stops:
        if stop.client_algo_id not in known_stops:
            findings.append(f"{stop.symbol}: conditional order {stop.client_algo_id} is not in the strategy journal")
    for symbol in symbols:
        venue_leverage, venue_margin = settings[symbol]
        if venue_leverage != leverage:
            findings.append(f"{symbol}: exchange leverage {venue_leverage}x, configured {leverage}x")
        if venue_margin != margin:
            findings.append(f"{symbol}: exchange margin {venue_margin}, configured {margin}")
    return findings


def _ledger_qty(journal: OrderJournal, rows: list[dict[str, Any]], symbol: str, symbols: tuple[str, ...]) -> Decimal:
    """The strategy's booked position in ``symbol``: the sum over the owners that traded it (as reconciliation)."""
    owners = {row["owner"] for row in rows if row["symbol"] == symbol}
    if symbol in symbols:
        owners.add(owner_of(symbol))
    total = Decimal(0)
    for owner in owners:
        checkpoint = journal.checkpoint(owner)
        if checkpoint:
            total += Decimal(str(checkpoint["position"]["qty"]))
    return total


__all__ = ["foreign_holdings", "owner_of"]
