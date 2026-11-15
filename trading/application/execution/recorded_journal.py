"""The order journal with an audit trail: every order and stop status change is also an event (data and state
architecture decision, 2026-11-15; ADR-0014 point 2).

The journal stays the state; the events only record how it got there. Writing an event never fails the order
path: a failing event log is logged and ignored, the journal operation has already happened.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from trading.application.ports import EventLog, OrderJournal
from trading.domain.events import EventKind

logger = logging.getLogger(__name__)


class RecordedOrderJournal:
    def __init__(self, inner: OrderJournal, events: EventLog) -> None:
        self.inner, self.events = inner, events

    def _record(self, kind: EventKind, **fields: Any) -> None:
        try:
            self.events.write(kind, **fields)
        except Exception:
            logger.warning("Audit event not written: %s %s", kind.value, fields, exc_info=True)

    # -- status changes ----------------------------------------------------------------------------------------
    def reserve(self, *, intent_id: str, symbol: str, client_id: str, side: str, qty: str, reducing: bool,
                owner: str, parent: dict[str, Any] | None, order_type: str,
                clock: Callable[[], float], venue_close: bool = False) -> tuple[str, float]:
        refusal, created_at = self.inner.reserve(intent_id=intent_id, symbol=symbol, client_id=client_id, side=side,
                                                 qty=qty, reducing=reducing, owner=owner, parent=parent,
                                                 order_type=order_type, clock=clock, venue_close=venue_close)
        if not refusal:
            self._record(EventKind.ORDER, symbol=symbol, order_id=intent_id, status="reserved", side=side, qty=qty,
                         reduce_only=reducing, order_type=order_type, owner=owner)
        return refusal, created_at

    def adopt(self, *, intent_id: str, symbol: str, client_id: str, order_id: str, side: str, owner: str,
              parent: dict[str, Any], created_at: float) -> None:
        self.inner.adopt(intent_id=intent_id, symbol=symbol, client_id=client_id, order_id=order_id, side=side,
                         owner=owner, parent=parent, created_at=created_at)
        self._record(EventKind.ORDER, symbol=symbol, order_id=intent_id, status="adopted", venue_order_id=order_id,
                     side=side, owner=owner)

    def set_order_id(self, intent_id: str, order_id: str) -> None:
        self.inner.set_order_id(intent_id, order_id)
        self._record(EventKind.ORDER, order_id=intent_id, status="acknowledged", venue_order_id=order_id)

    def update_status(self, intent_id: str, status: str, *, settle: bool) -> str | None:
        previous = self.inner.update_status(intent_id, status, settle=settle)
        if previous is not None and (previous != status or settle):
            self._record(EventKind.ORDER, order_id=intent_id, status=status, previous=previous, settled=settle)
        return previous

    def add_stop(self, *, owner: str, client_algo_id: str, symbol: str, revision: str, attempt: int, side: str,
                 trigger_price: str, created_at: float) -> None:
        self.inner.add_stop(owner=owner, client_algo_id=client_algo_id, symbol=symbol, revision=revision,
                            attempt=attempt, side=side, trigger_price=trigger_price, created_at=created_at)
        # The journal stores a new stop as "unknown" until the placement's outcome is known (S3 audit finding 7).
        self._record(EventKind.STOP, symbol=symbol, client_algo_id=client_algo_id, status="unknown", side=side,
                     trigger_price=trigger_price, owner=owner)

    def set_stop_status(self, client_algo_id: str, status: str) -> None:
        self.inner.set_stop_status(client_algo_id, status)
        self._record(EventKind.STOP, client_algo_id=client_algo_id, status=status)

    # -- reads and the rest: unchanged ---------------------------------------------------------------------------
    def orders(self, owner: str | None = None, *, unresolved: bool = False) -> list[dict[str, Any]]:
        return self.inner.orders(owner, unresolved=unresolved)

    def fills(self, owner: str, *, undelivered: bool = True) -> list[dict[str, Any]]:
        return self.inner.fills(owner, undelivered=undelivered)

    def checkpoint(self, owner: str) -> dict[str, Any] | None:
        return self.inner.checkpoint(owner)

    def consume(self, owner: str, fill_id: str, apply: Callable[[], dict[str, Any]], rollback: Callable[[], None],
                *, expected_revision: str | None = None) -> bool:
        return self.inner.consume(owner, fill_id, apply, rollback, expected_revision=expected_revision)

    def intent_fills(self, intent_id: str) -> list[dict[str, Any]]:
        return self.inner.intent_fills(intent_id)

    def fill(self, fill_id: str, *, owner: str | None = None) -> dict[str, Any] | None:
        return self.inner.fill(fill_id, owner=owner)

    def insert_fill(self, *, fill_id: str, intent_id: str, owner: str, payload: dict[str, Any]) -> bool:
        return self.inner.insert_fill(fill_id=fill_id, intent_id=intent_id, owner=owner, payload=payload)

    def stop_rows(self, owner: str, symbol: str, *, live_only: bool) -> list[dict[str, Any]]:
        return self.inner.stop_rows(owner, symbol, live_only=live_only)

    def __getattr__(self, name: str) -> Any:
        # Adapter extras outside the port (scope, connect, the manual tools' methods) pass straight through.
        return getattr(self.inner, name)


__all__ = ["RecordedOrderJournal"]
