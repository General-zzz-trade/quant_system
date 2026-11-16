"""Exchange-side catastrophic stop for each confirmed strategy position (moved from protective_stops.py).

The strategy's own exits run on hourly closes. This manager keeps one
exchange-side close-all stop order per open position so a
large intrabar move, a crashed process or a lost connection still closes the
position at roughly ``stop_pct`` from the confirmed entry price.

Each ``sync()`` (called every few seconds from the runtime's single thread):

- ledger and venue flat  -> cancel any active stop;
- position, no stop      -> place one covering the venue position, triggered from the venue's entry price
                            (ADR-0012; deterministic client id);
- venue position changed -> place the stop of the new revision, then cancel the old one once it is in place;
- ledger 0               -> no strategy stop (decision N);
- ledger not 0, venue flat, opposite or unknown -> keep the stop in place, place nothing, report the issue;
- stop triggered         -> adopt the resulting exchange order into the
                            durable journal so its fills reach the strategy
                            ledger exactly once; while the venue has not named
                            that order yet the stop stays followed and no new
                            stop is placed (step 12, R8);
- stop gone (canceled, expired, ...) -> place a new one (next attempt id);
- placement refused because price is already beyond the trigger -> report
                            ``stop_breached`` so the runtime closes the
                            position with a market order.

State lives in the order journal database (table ``protective_stops``). The venue, the executor that
adopts triggered stops and the clock are injected (migration step 9b). The venue reports a stop in one of
three states (``trading.domain.venue``) and flags a breached placement itself: no exchange status word
or error code is known here (step 10b).
"""
from __future__ import annotations

import hashlib
import logging
from decimal import Decimal
from typing import Any, Protocol

from trading.application.issues import Issue
from trading.application.ports import Clock, OrderJournal, OrderView, StopOrderGateway
from trading.domain.issues import Effect, IssueKind as K
from trading.domain.venue import STOP_ACTIVE, STOP_GONE, STOP_TRIGGERED

logger = logging.getLogger(__name__)

MIN_STOP_PCT, MAX_STOP_PCT = Decimal("0.005"), Decimal("0.15")
AWAITING_ORDER = "awaiting_order"     # triggered; the order it became is not known yet (a live row)
BREACHED = "breached"                 # placement refused because price was already beyond the trigger
NO_ORDER = "triggered_no_order"       # triggered, then ended at the venue without any order
# Rows that mean "price went through the stop": each pauses the symbol's entries until a person releases it.
# A stop row's status after the price went through it, as an issue kind; the pause cause is the text.
TRIGGER_KINDS = {"triggered": K.STOP_TRIGGERED, AWAITING_ORDER: K.STOP_TRIGGERED, NO_ORDER: K.STOP_TRIGGERED,
                 BREACHED: K.STOP_BREACHED}
PAUSE_CAUSES = {K.STOP_TRIGGERED: "protective stop triggered",
                K.STOP_BREACHED: "protective stop breached; position flattened"}


class TriggeredOrder(OrderView, Protocol):
    """The order a triggered stop became; its side is needed to adopt it."""

    @property
    def side(self) -> str: ...


class PositionView(Protocol):
    @property
    def symbol(self) -> str: ...

    @property
    def qty(self) -> Decimal: ...

    @property
    def entry_price(self) -> Decimal | None: ...

    @property
    def mark_price(self) -> Decimal | None: ...


class StopVenue(StopOrderGateway, Protocol):
    """Places and looks up stops, looks up the order a triggered stop became, and reads one symbol's venue position
    (flat included; None when not reported - ADR-0012)."""

    def get_position(self, symbol: str) -> PositionView | None: ...

    def get_order(self, symbol: str, *, order_id: str = "",
                  client_order_id: str = "") -> TriggeredOrder | None: ...


class StopOwner(Protocol):
    """The durable executor of the position's owner: its journal and how it adopts a venue order."""

    @property
    def journal(self) -> OrderJournal: ...

    @property
    def scope(self) -> str: ...

    @property
    def owner(self) -> str: ...

    def adopt_exchange_order(self, *, intent_id: str, symbol: str, side: str, order_id: str,
                             client_order_id: str, reason: str) -> bool: ...


class ProtectiveStopManager:
    def __init__(self, adapter: StopVenue, executor: StopOwner, symbol: str, *, stop_pct: Decimal,
                 clock: Clock) -> None:
        stop_pct = Decimal(str(stop_pct))
        if not MIN_STOP_PCT <= stop_pct <= MAX_STOP_PCT:
            raise ValueError(f"stop_pct must be between {MIN_STOP_PCT} and {MAX_STOP_PCT}")
        self.adapter, self.executor, self.symbol = adapter, executor, symbol
        self.stop_pct, self.clock = stop_pct, clock
        self.journal = executor.journal

    # ------------------------------------------------------------------
    def _client_id(self, revision: str, attempt: int) -> str:
        digest = hashlib.sha256(f"{self.executor.scope}:{self.executor.owner}:{revision}:{attempt}".encode())
        return "qsstop" + digest.hexdigest()[:28]  # 34 chars; the venue accepts at most 36

    def _rows(self, *, live_only: bool = True) -> list[dict[str, Any]]:
        return self.journal.stop_rows(self.executor.owner, self.symbol, live_only=live_only)

    def _set(self, client_algo_id: str, status: str) -> None:
        self.journal.set_stop_status(client_algo_id, status)

    def _position(self) -> tuple[Decimal, Decimal, str]:
        checkpoint = self.journal.checkpoint(self.executor.owner)
        if not checkpoint:
            return Decimal(0), Decimal(0), ""
        position = checkpoint["position"]
        return Decimal(position["qty"]), Decimal(position["entry_price"]), str(checkpoint.get("revision") or "")

    # ------------------------------------------------------------------
    def _issue(self, kind: K, text: str) -> Issue:
        return Issue(kind, self.symbol, f"{self.symbol}: {text}")

    def sync(self) -> list[Issue]:
        """Bring the exchange stop in line with the venue position (ADR-0012). Returns issues.

        The stop covers the venue position: its revision and trigger come from the venue's quantity and entry price
        (normalized, so the same position in another text form keeps its stop). With a ledger of 0 the strategy
        places no stop. When the ledger holds a position but the venue is flat, opposite or unknown, the stop in place
        is kept, nothing new is placed, and the issue blocks entries. A stop is cancelled only when ledger and venue
        are both flat, when the stop of the new revision is in place, or while a fired stop is followed."""
        issues: list[Issue] = []
        ledger = self._position()[0]
        venue = self._venue_position()
        target = self._target(ledger, venue)
        following = unsure = False
        matching: dict[str, Any] | None = None
        others: list[dict[str, Any]] = []
        for row in self._rows():
            state = self.adapter.get_algo_order(client_algo_id=row["client_algo_id"])
            status = state.state if state is not None else None
            if state is not None and status == STOP_TRIGGERED and state.actual_order_id \
                    and self._adopt(row, state.actual_order_id):
                self._set(row["client_algo_id"], "triggered")
                issues.append(self._issue(K.STOP_TRIGGERED, "protective stop triggered"))
                following = True   # the ledger is stale until the adopted fill is consumed: place nothing now
                continue
            if row["status"] == AWAITING_ORDER and state is not None and status == STOP_GONE \
                    and not state.actual_order_id:
                # It went through the trigger but never became an order: the position is unprotected again.
                self._set(row["client_algo_id"], NO_ORDER)
                issues.append(self._issue(K.STOP_MISSING, "protective stop triggered but ended without an order; "
                                          "a new stop is placed"))
                continue
            if (state is not None and status == STOP_TRIGGERED) or row["status"] == AWAITING_ORDER:
                # Keep following it: its fill must reach the ledger, and a new stop now could close a later
                # position at a stale price.
                if row["status"] != AWAITING_ORDER:
                    self._set(row["client_algo_id"], AWAITING_ORDER)
                issues.append(self._issue(K.STOP_UNCERTAIN, "protective stop triggered; the order it became is "
                                          "not known yet"))
                following = True
                continue
            if status is None and row["status"] == "unknown" and self.clock() - row["created_at"] < 90:
                unsure = True      # ambiguous placement; keep looking it up before replacing anything
                continue
            if status != STOP_ACTIVE:
                self._set(row["client_algo_id"], "inactive")
                continue
            if row["status"] == "canceling":
                # A stale stop must never survive to close a later position.
                self.adapter.cancel_algo_order(client_algo_id=row["client_algo_id"])
                continue
            if row["status"] != "active":
                self._set(row["client_algo_id"], "active")
            if target is not None and row["revision"] == target[0] and matching is None:
                matching = row
            else:
                others.append(row)
        both_flat = ledger == 0 and venue is not None and venue.qty == 0
        if both_flat or matching is not None or following:
            for row in others:
                self.adapter.cancel_algo_order(client_algo_id=row["client_algo_id"])
                self._set(row["client_algo_id"], "canceling")
        if ledger == 0 or following or unsure or matching is not None:
            return issues
        if target is None:
            return issues + [self._issue(K.STOP_UNCERTAIN, self._no_target(ledger, venue))]
        revision, side, trigger = target
        attempts = [r["attempt"] for r in self._rows(live_only=False) if r["revision"] == revision]
        attempt = (max(attempts) + 1) if attempts else 0
        client_algo_id = self._client_id(revision, attempt)
        self.journal.add_stop(owner=self.executor.owner, client_algo_id=client_algo_id, symbol=self.symbol,
                              revision=revision, attempt=attempt, side=side, trigger_price=str(trigger),
                              created_at=float(self.clock()))
        result = self.adapter.place_stop_market(self.symbol, side, trigger, client_algo_id=client_algo_id)
        if result.get("status") == "submitted":
            self._set(client_algo_id, "active")
            logger.info("%s protective stop %s %s @ %s", self.symbol, side, client_algo_id, trigger)
            for row in others:      # the stop of the new revision is in place: the old ones go (ADR-0012 point 3)
                self.adapter.cancel_algo_order(client_algo_id=row["client_algo_id"])
                self._set(row["client_algo_id"], "canceling")
        elif result.get("status") == "rejected":
            breached = result.get("breached") is True
            self._set(client_algo_id, BREACHED if breached else "rejected")     # one write: a crash cannot lose it
            if breached:
                issues.append(self._issue(K.STOP_BREACHED, f"stop_breached (price already beyond {trigger})"))
            else:
                issues.append(self._issue(K.STOP_MISSING, f"protective stop rejected: {result.get('msg')}"))
        else:
            issues.append(self._issue(K.STOP_UNCERTAIN, "protective stop placement outcome unknown; will query"))
        return issues

    def _venue_position(self) -> PositionView | None:
        """The symbol's venue position, flat included; None when it cannot be read or is not reported."""
        try:
            return self.adapter.get_position(self.symbol)
        except Exception:
            logger.warning("%s venue position unavailable", self.symbol, exc_info=True)
            return None

    def _target(self, ledger: Decimal, venue: PositionView | None) -> tuple[str, str, Decimal] | None:
        """(revision, side, trigger) of the stop to have: one covering the venue position, from the venue's entry
        price, when the ledger holds a position in the same direction; None otherwise."""
        if ledger == 0 or venue is None or venue.qty == 0 or (venue.qty > 0) != (ledger > 0):
            return None
        entry = Decimal(str(venue.entry_price or 0))
        if not entry.is_finite() or entry <= 0:
            return None
        side = "sell" if venue.qty > 0 else "buy"
        trigger = entry * (1 - self.stop_pct) if venue.qty > 0 else entry * (1 + self.stop_pct)
        revision = f"{format(abs(Decimal(venue.qty)).normalize(), 'f')}@{format(entry.normalize(), 'f')}"
        return revision, side, trigger

    def _no_target(self, ledger: Decimal, venue: PositionView | None) -> str:
        """The wording these issues have always had; the stop in place is kept (ADR-0012 point 3)."""
        if venue is None:
            return "venue position unavailable; protective stop not placed"
        if venue.qty != 0 and (venue.qty > 0) == (ledger > 0):
            return "venue reports no entry price; protective stop not placed"
        held = Decimal(venue.qty) if venue.qty != 0 else Decimal(0)
        return f"venue holds {held}, ledger {ledger}; protective stop not placed"

    def _adopt(self, row: dict[str, Any], actual_order_id: str) -> bool:
        """True once the order the stop became is in the journal (adopted now or before)."""
        order = self.adapter.get_order(self.symbol, order_id=actual_order_id)
        if order is None:
            return False
        self.executor.adopt_exchange_order(
            intent_id=f"stop-{row['client_algo_id']}", symbol=self.symbol, side=order.side,
            order_id=order.order_id, client_order_id=order.client_order_id, reason="protective_stop")
        return True

    def triggers(self) -> dict[str, Issue]:
        """Every stop of this position that the price went through and whose kind pauses entries (ADR-0014
        table), with the pause cause as its text."""
        found = {}
        for row in self._rows(live_only=False):
            kind = TRIGGER_KINDS.get(row["status"])
            if kind is not None and Effect.PAUSE_SYMBOL in Issue(kind, self.symbol, "").effects:
                found[row["client_algo_id"]] = Issue(kind, self.symbol, PAUSE_CAUSES[kind])
        return found

    def has_active_stop(self) -> bool:
        return any(r["status"] == "active" for r in self._rows())

    def unprotected(self) -> bool:
        """A confirmed position without an active exchange stop (ADR-0005 4b: entries are then blocked)."""
        return self._position()[0] != 0 and not self.has_active_stop()
