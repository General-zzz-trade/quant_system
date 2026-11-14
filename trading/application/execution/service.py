"""Durable execution for one strategy owner (moved from BinanceExecutionAdapter in migration step 9d).

Decides whether and how an order intent reaches the order lifecycle:

- a repeated intent never places again;
- an entry needs a clean reconciliation (``ready``) and a closed circuit breaker;
- a maker entry rests at ``reference x (1 -/+ offset)``;
- a reduction first cancels resting entries, then is sized by ``size_reduction``;
- any failure blocks entries until the next clean reconciliation. Nothing is retried.

The venue, the lifecycle, the clocks, the lock and the observability log are injected. Event formats
belong to the caller: it supplies an opaque ``parent`` record that is stored with the intent (built only
for an intent that passed every check), and it receives exchange fill facts.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Callable, Protocol

from trading.application.execution.lifecycle import OrderLifecycle, PendingOrder
from trading.application.execution.reduction import size_reduction
from trading.application.ports import Clock, EventLog, ExchangeAccount, Lock, OrderGateway, OrderJournal
from trading.application.reconcile import reconcile_account
from trading.domain.orders import BLOCKED, OrderIntent

logger = logging.getLogger(__name__)
MAX_OFFSET = Decimal("0.01")
ENTRY_MODES = ("maker", "market")


class Venue(ExchangeAccount, OrderGateway, Protocol):
    """Account snapshot plus order placement and lookup."""


class CircuitBreaker(Protocol):
    def allow_request(self) -> bool: ...

    def record_success(self) -> None: ...

    def record_failure(self) -> None: ...


@dataclass(frozen=True)
class SendResult:
    submitted: bool               # the intent passed every check and reached the order lifecycle
    facts: tuple[Any, ...]        # exchange fill facts confirmed during this call


class ExecutionService:
    def __init__(self, venue: Venue, orders: OrderLifecycle, *, owner: str, entry_mode: str, limit_offset: Decimal,
                 lock: Lock, clock: Clock, monotonic: Clock, breaker: CircuitBreaker | None = None,
                 event_log: EventLog | None = None) -> None:
        if not limit_offset.is_finite() or not 0 <= limit_offset < MAX_OFFSET:
            raise ValueError("limit offset must be in [0, 1%)")
        if entry_mode not in ENTRY_MODES:
            raise ValueError("entry_mode must be maker or market")
        self._venue, self._orders, self.owner = venue, orders, owner
        self._entry_mode, self._offset, self._breaker = entry_mode, limit_offset, breaker
        self._lock, self._clock, self._monotonic, self._event_log = lock, clock, monotonic, event_log
        self.ready = not bool(owner)
        self.status_callback: Callable[[str, str], None] = lambda *_: None
        self._submitted_at: dict[str, float] = {}
        self._reference: dict[str, str] = {}      # intent id -> reference price recorded with the intent
        for row in self.journal.orders(owner):
            if row["parent"]:
                self._reference[row["intent_id"]] = str(json.loads(row["parent"]).get("price") or 0)

    @property
    def journal(self) -> OrderJournal:
        return self._orders.journal

    def _journaled(self, intent_id: str, owner: str | None = None) -> bool:
        return any(row["intent_id"] == intent_id for row in self.journal.orders(owner))

    def _log_fills(self, intent_id: str, facts: tuple[Any, ...]) -> None:
        """Transaction-cost records; the fill is already durable, so a logging failure is ignored."""
        if not facts or self._event_log is None:
            return
        try:
            reference = float(self._reference[intent_id] or 0)
            for fact in facts:
                latency = (self._monotonic() - self._submitted_at.get(intent_id, self._monotonic())) * 1000
                self._event_log.write("fill", symbol=fact.symbol, side=fact.side, qty=float(fact.qty),
                                      ref_price=reference, fill_price=float(fact.price), latency_ms=latency,
                                      order_id=fact.order_id, fill_id=fact.fill_id, ref_source="decision_price")
        except Exception:
            logger.debug("Confirmed fill committed; TCA unavailable", exc_info=True)

    def send(self, intent: OrderIntent, parent: Callable[[], dict[str, Any]]) -> SendResult:
        """Place ``intent`` at most once. ``parent()`` is the record stored with it; its ``price`` is the
        reference for transaction-cost records."""
        with self._lock:
            return self._send(intent, parent)

    def _send(self, intent: OrderIntent, parent: Callable[[], dict[str, Any]]) -> SendResult:
        intent_id, submitted = intent.order_id, False
        try:
            if not intent_id:
                raise ValueError("Order intent identity is required")
            if self.owner and intent.source != self.owner:
                raise ValueError("Order belongs to another strategy")
            if self._journaled(intent_id):
                return SendResult(False, ())  # a repeated intent never places again
            qty = intent.qty
            if not qty.is_finite() or qty < 0:
                raise ValueError("Invalid order quantity")
            reducing = qty == 0 or intent.reduce_only is True
            if not reducing and (not self.ready or (self._breaker is not None and not self._breaker.allow_request())):
                self.status_callback(intent_id, BLOCKED)
                return SendResult(False, ())
            side = intent.side.lower()
            limit_price = None
            if not reducing and self._entry_mode == "maker":
                reference = intent.price
                if not reference.is_finite() or reference <= 0:
                    raise ValueError("Entry requires a positive reference price")
                limit_price = reference * (1 - (self._offset if side == "buy" else -self._offset))
            consumer_revision = None
            if reducing:
                # Never leave an entry remainder able to re-open after an exit.
                for cancelled_intent, fact in self._orders.cancel_entries(intent.symbol):
                    self._log_fills(cancelled_intent, (fact,))
                sized = size_reduction(self._venue, self.journal, owner=self.owner, symbol=intent.symbol, side=side,
                                       qty=qty)
                qty, side, consumer_revision = sized.qty, sized.side, sized.consumer_revision
            record = parent()
            if reducing and self.owner:
                record["consumer_revision"] = consumer_revision
            self._reference[intent_id] = str(record.get("price") or 0)
            self._submitted_at[intent_id] = self._monotonic()
            submitted = True
            facts = self._orders.submit(intent_id=intent_id, symbol=intent.symbol, side=side, qty=qty,
                                        reduce_only=reducing, parent=record, limit_price=limit_price)
            if not self._journaled(intent_id, self.owner):
                self.status_callback(intent_id, BLOCKED)
            self._log_fills(intent_id, facts)
            if facts and self._breaker is not None:
                self._breaker.record_success()
            return SendResult(True, facts)
        except Exception:
            self.fail(intent_id)
            return SendResult(submitted, ())

    def fail(self, intent_id: str) -> None:
        """Failure handling while an exception is being handled: block entries, never retry the placement."""
        self.ready = False
        if self._breaker is not None:
            self._breaker.record_failure()
        if not self._journaled(intent_id, self.owner):
            self.status_callback(intent_id, BLOCKED)
        logger.exception("Order unresolved or blocked; no placement retry")

    def adopt(self, *, intent_id: str, symbol: str, side: str, order_id: str, client_order_id: str,
              parent: dict[str, Any]) -> bool:
        """Journal a venue-originated reduction (a triggered stop) for this owner. Idempotent per intent."""
        with self._lock:
            if self._journaled(intent_id):
                return False
            self.journal.adopt(intent_id=intent_id, symbol=symbol, client_id=client_order_id, order_id=order_id,
                               side=side, owner=self.owner, parent=parent, created_at=self._clock())
            self._reference[intent_id] = str(parent.get("price") or 0)
            self._orders.track(PendingOrder(intent_id, symbol, side, client_order_id, order_id, is_limit=False,
                                            created_at=self._clock()))
            return True

    def fill_fact(self, fill_id: str) -> dict[str, Any]:
        fact = self.journal.fill(fill_id, owner=self.owner)
        if fact is None:
            raise ValueError("Unknown fill for this strategy")
        return fact

    def notify_status(self) -> None:
        for row in self.journal.orders(self.owner):
            self.status_callback(row["intent_id"], row["status"])

    def poll(self) -> list[dict[str, Any]]:
        """Collect new exchange fills of working orders; returns every fill not yet consumed by the owner."""
        with self._lock:
            for intent_id, fact in self._orders.poll():
                self._log_fills(intent_id, (fact,))
            return self.journal.fills(self.owner)

    def reconcile(self, symbol: str) -> list[str]:
        """Read-only verification; any issue blocks entries until a clean pass."""
        self.ready = False
        blockers = getattr(self._venue, "execution_blockers", None)   # optional on legacy venues
        issues = list(blockers()) if blockers is not None else []
        issues += reconcile_account(self._venue, self.journal, owner=self.owner, symbol=symbol)
        self.ready = not issues
        if issues:
            logger.error("Entry blocked: %s", "; ".join(issues))
        return issues


__all__ = ["CircuitBreaker", "ExecutionService", "SendResult", "Venue"]
