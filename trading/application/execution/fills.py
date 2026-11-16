"""Durable consumption of confirmed fills into the owner's position checkpoint."""
from __future__ import annotations

import logging

from decimal import Decimal
from types import SimpleNamespace
from typing import Any, Protocol

from trading.application.ports import EventLog, OrderJournal
from trading.domain.events import EventKind
from trading.domain.issues import IssueKind
from trading.domain.position import StrategyPosition

logger = logging.getLogger(__name__)


class FillSource(Protocol):
    """The owner's executor: its journal, its name and the durable fact behind a fill id."""

    @property
    def journal(self) -> OrderJournal: ...

    @property
    def owner(self) -> str: ...

    def fill_fact(self, fill_id: str) -> dict[str, Any]: ...


class FillEventLike(Protocol):
    fill_id: str
    order_id: str
    symbol: str
    side: str
    qty: Decimal
    price: Decimal


class TargetFillConsumer:
    """Durable consumption of confirmed fills into the owner's position checkpoint.

    A reduction's fill never takes the ledger past zero, nor opens a ledger position from zero (ADR-0012 point 5;
    the tech lead extended it on 2026-11-15 to every reduction: stops, breached-stop, kill-switch and drawdown
    flattens that close the whole venue position). Only the ledger's part is consumed; the rest is stored as a
    foreign fill (owner ``foreign:<symbol>``, in no owner's checkpoint), with a FOREIGN_FILL issue and an alert. An
    entry never crosses zero either: the strategy splits a flip into a reduction and an entry.

    The foreign part is a record (S4-S5 audit): it is stored under its own intent id (``<intent>:foreign``), so it
    never counts as the order's fill - the order settles on what the venue filled - and never as a fill waiting to
    be consumed. A fill consumed already is never split again. The issue and alert are written before the
    ledger's part is consumed: at least once (again after a crash before that), once per process while a
    consumption keeps failing."""

    def __init__(self, executor: FillSource, events: EventLog | None = None) -> None:
        self.executor, self.journal, self.owner, self.events = executor, executor.journal, executor.owner, events
        self._alerted: set[str] = set()
        saved = self.journal.checkpoint(self.owner)
        self.position = StrategyPosition.from_dict(saved["position"]) if saved else StrategyPosition()
        self.revision = saved["revision"] if saved else None

    def consume(self, event: FillEventLike) -> bool:
        fact = self.executor.fill_fact(event.fill_id)
        if (str(event.order_id) != fact["intent_id"] or event.symbol != fact["symbol"] or event.side != fact["side"]
                or event.qty != Decimal(fact["qty"]) or event.price != Decimal(fact["price"])):
            raise ValueError("Event differs from its durable exchange fact")
        if not any(f.get("fill_id") == event.fill_id for f in self.journal.fills(self.owner)):
            return False                             # consumed already: never split or applied again
        before = self.position.to_dict()
        owned = self._owned_part(event, fact)

        def apply() -> dict[str, Any]:
            if Decimal(owned["qty"]) > 0:            # all foreign: the ledger is unchanged, the fill is consumed
                self.position.apply(SimpleNamespace(**owned))
            return {"position": self.position.to_dict(), "revision": event.fill_id, "strategy": {}}

        def rollback() -> None:
            self.position = StrategyPosition.from_dict(before)

        changed = self.journal.consume(self.owner, event.fill_id, apply, rollback, expected_revision=self.revision)
        if changed:
            self.revision = event.fill_id
        return changed

    def _reducing(self, intent_id: str) -> bool:
        return any(row["intent_id"] == intent_id and row["reducing"] for row in self.journal.orders(self.owner))

    def _owned_part(self, event: FillEventLike, fact: dict[str, Any]) -> dict[str, Any]:
        qty, old = Decimal(fact["qty"]), self.position.qty
        signed = qty if fact["side"] == "buy" else -qty
        if self._reducing(fact["intent_id"]):
            mine = min(qty, abs(old)) if old * signed < 0 else Decimal(0)
        elif old != 0 and (old + signed) * old < 0:
            mine = abs(old)
        else:
            return fact
        if mine == qty:
            return fact
        excess = qty - mine
        fee = Decimal(str(fact.get("fee") or 0))
        owned = dict(fact, qty=str(mine), fee=str(fee * mine / qty))
        foreign = dict(fact, qty=str(excess), fee=str(fee - Decimal(owned["fee"])))
        # Stored before the ledger's part is consumed; storing it again after a restart is ignored. Its own intent
        # id keeps it out of the order's fills and out of the fills the journal waits for (S4-S5 audit, 1 and 2).
        self.journal.insert_fill(fill_id=f"{event.fill_id}:foreign", intent_id=f"{fact['intent_id']}:foreign",
                                 owner=f"foreign:{fact['symbol']}", payload=foreign)
        if event.fill_id not in self._alerted:
            text = (f"{fact['symbol']}: {excess} of fill {event.fill_id} was beyond the strategy ledger "
                    f"({mine}); booked as a foreign fill")
            logger.warning(text)
            if self.events is not None:
                self.events.write(EventKind.ISSUE, symbol=fact["symbol"], issue=text,
                                  issue_kind=IssueKind.FOREIGN_FILL.value)
                self.events.write(EventKind.ALERT, symbol=fact["symbol"], reason=text)
            self._alerted.add(event.fill_id)
        return owned


__all__ = ["TargetFillConsumer"]
