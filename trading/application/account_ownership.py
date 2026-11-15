"""Does the account hold only what the strategy owns? The execution start-up check of ADR-0010.

Read-only: it compares the whole account (positions, open orders and open conditional orders on every symbol, and
the leverage and margin mode of the strategy's symbols) with the order journal and the configured settings, and
returns every finding. The caller refuses to start on any finding. Nothing here places, cancels or changes
anything.

What the strategy's own activity can explain is bounded (independent audit of ADR-0010, findings 1, 2 and 4).
A foreign position inside those bounds passes this check; reconciliation then blocks the symbol's entries:
- the booked position, plus fills that are stored but not yet consumed (a crash between the two);
- each unsettled strategy order lets the venue be ahead of the ledger by at most its unfilled quantity, in its own
  direction; a reducing order only towards zero, and a close-all reduction (quantity 0) or a protective stop that
  fired while the runtime was stopped all the way to zero. A row without a side or quantity allows nothing;
- an open order or stop is the strategy's only if a journal row of the same symbol names it.
"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any

from trading.application.ports import AccountSnapshot, OrderJournal
from trading.application.reconcile import booked_position
from trading.domain.venue import STOP_TRIGGERED

FIRED = "awaiting_order"     # a stop row the runtime saw fire but whose order it has not adopted yet


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
    except Exception as exc:
        return [f"strategy journal could not be read ({type(exc).__name__}: {exc})"]

    findings: list[str] = []
    stop_rows: list[dict[str, Any]] = []
    try:
        stop_rows = [row for symbol in symbols for row in journal.stop_rows(owner_of(symbol), symbol, live_only=False)]
    except Exception as exc:
        # Unread records (a damaged database; ADR-0009 init creates the stop table) only add refusals: every
        # conditional order counts as foreign and no stop counts as fired.
        if stops:
            findings.append(f"strategy stop records could not be read ({type(exc).__name__}: {exc})")
    open_stops = {(stop.symbol, stop.client_algo_id) for stop in stops}
    try:
        fired = {row["symbol"] for row in stop_rows if _fired(account, row, open_stops)}
    except Exception as exc:
        return [f"account could not be read ({type(exc).__name__}: {exc})"]

    venue: dict[str, Decimal] = {}
    for p in positions:
        venue[p.symbol] = venue.get(p.symbol, Decimal(0)) + Decimal(str(p.qty))
    try:
        for symbol in sorted({*symbols, *venue}):
            base, low, high = _allowed(journal, rows, symbol, owner_of(symbol) if symbol in symbols else None,
                                       fired=symbol in fired)
            actual = venue.get(symbol, Decimal(0))
            if low <= actual <= high:
                continue
            if actual == 0:
                findings.append(f"{symbol}: the venue is flat but the strategy ledger holds {base}")
            else:
                window = f"; in-flight strategy orders allow {low} to {high}" if (low, high) != (base, base) else ""
                findings.append(f"{symbol}: position {actual} is not in the strategy ledger (ledger {base}{window})")
    except Exception as exc:
        return [f"strategy journal could not be read ({type(exc).__name__}: {exc})"]

    known = {(str(row["symbol"]), str(row[key])) for row in rows for key in ("order_id", "client_id") if row.get(key)}
    for order in orders:
        if (order.symbol, order.order_id) not in known and (order.symbol, order.client_order_id) not in known:
            findings.append(f"{order.symbol}: open order {order.order_id} is not in the strategy journal")
    known_stops = {(str(row["symbol"]), str(row["client_algo_id"])) for row in stop_rows}
    for stop in stops:
        if (stop.symbol, stop.client_algo_id) not in known_stops:
            findings.append(f"{stop.symbol}: conditional order {stop.client_algo_id} is not in the strategy journal")
    for symbol in symbols:
        venue_leverage, venue_margin = settings[symbol]
        if venue_leverage != leverage:
            findings.append(f"{symbol}: exchange leverage {venue_leverage}x, configured {leverage}x")
        if venue_margin != margin:
            findings.append(f"{symbol}: exchange margin {venue_margin}, configured {margin}")
    return findings


def _fired(account: AccountSnapshot, row: dict[str, Any], open_stops: set[tuple[str, str]]) -> bool:
    """A live stop of the strategy that went through its trigger while the runtime was stopped."""
    if row["status"] == FIRED:
        return True
    if row["status"] not in ("active", "unknown", "canceling") or (row["symbol"], row["client_algo_id"]) in open_stops:
        return False
    state = account.get_algo_order(client_algo_id=row["client_algo_id"])
    return state is not None and state.state == STOP_TRIGGERED


def _allowed(journal: OrderJournal, rows: list[dict[str, Any]], symbol: str, owner: str | None, *,
             fired: bool) -> tuple[Decimal, Decimal, Decimal]:
    """(booked position incl. stored fills, lowest and highest venue position the strategy's own in-flight
    orders and fired stops can explain)."""
    owners = {row["owner"] for row in rows if row["symbol"] == symbol} | ({owner} if owner else set())
    base = booked_position(journal, rows, symbol, owner)
    for name in sorted(owners):
        for fill in journal.fills(name, undelivered=True):
            if fill.get("symbol") == symbol:
                base += _fill_qty(fill)
    low = high = base
    for row in rows:
        if row["symbol"] != symbol or row["settled"]:
            continue
        parsed = _parse(row.get("side"), row.get("qty"))
        if parsed is None:
            continue                                           # a legacy row without side or quantity
        side, amount = parsed
        remaining = max(amount - sum((Decimal(str(f["qty"])) for f in journal.intent_fills(row["intent_id"])),
                                     Decimal(0)), Decimal(0))
        if not row.get("reducing"):
            high, low = (high + remaining, low) if side == "buy" else (high, low - remaining)
        elif (base > 0 and side == "sell") or (base < 0 and side == "buy"):   # a reduction only moves to zero
            if amount == 0:                                    # close-all: an adopted stop or venue-sized close
                reach = Decimal(0)
            else:
                reach = max(Decimal(0), base - remaining) if base > 0 else min(Decimal(0), base + remaining)
            low, high = min(low, reach), max(high, reach)
    if fired and base != 0:
        low, high = min(low, Decimal(0)), max(high, Decimal(0))
    return base, low, high


def _parse(side: object, qty: object) -> tuple[str, Decimal] | None:
    """(side, quantity) of a journal row or fill; None if either is missing or not a finite non-negative number."""
    side = str(side or "").lower()
    if qty is None or side not in ("buy", "sell"):
        return None
    try:
        amount = Decimal(str(qty))
    except (InvalidOperation, ValueError):
        return None
    return (side, amount) if amount.is_finite() and amount >= 0 else None


def _fill_qty(fill: dict[str, Any]) -> Decimal:
    parsed = _parse(fill.get("side"), fill.get("qty"))
    if parsed is None:
        raise ValueError(f"stored fill {fill.get('fill_id')} has no valid side and quantity")
    return parsed[1] if parsed[0] == "buy" else -parsed[1]


__all__ = ["foreign_holdings", "owner_of"]
