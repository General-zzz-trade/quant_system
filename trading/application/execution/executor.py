"""The execution service as one strategy owner's session, fill consumer and stop manager use it.

Order intents go in and typed fills come out; no event type of the legacy system is involved (migration
step 10). The record stored with each intent holds plain fields only:
``{"order_id", "symbol", "side", "price", "reason"}``. A record written by the legacy adapter carries an
extra event header, which is ignored here.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from trading.application.execution.service import ExecutionService
from trading.application.issues import Issue
from trading.application.ports import OrderJournal, PositionView
from trading.domain.orders import ConfirmedFill, OrderIntent


@dataclass(frozen=True)
class Fill:
    """A confirmed exchange trade of one order intent."""

    fill_id: str
    order_id: str        # the intent id
    symbol: str
    side: str
    qty: Decimal
    price: Decimal


def _fill(intent_id: str, fact: ConfirmedFill) -> Fill:
    return Fill(fill_id=fact.fill_id, order_id=intent_id, symbol=fact.symbol, side=fact.side,
                qty=fact.qty, price=fact.price)


class OwnerExecutor:
    def __init__(self, service: ExecutionService, *, scope: str) -> None:
        if not scope or scope != scope.strip():
            raise ValueError("durable execution requires an explicit account/environment scope")
        self._service, self._scope = service, scope

    @property
    def journal(self) -> OrderJournal:
        return self._service.journal

    @property
    def owner(self) -> str:
        return self._service.owner

    @property
    def scope(self) -> str:
        return self._scope

    @property
    def ready(self) -> bool:
        return self._service.ready

    def on_status(self, callback: Callable[[str, str], None]) -> None:
        self._service.status_callback = callback

    def block_entries(self) -> None:
        """Refuse entries until the next clean reconciliation; reductions still pass."""
        self._service.ready = False

    def send_order(self, intent: OrderIntent) -> tuple[Fill, ...]:
        """Place ``intent`` at most once; returns the fills confirmed during the call."""
        def parent() -> dict[str, Any]:
            return {"order_id": intent.order_id, "symbol": intent.symbol, "side": intent.side,
                    "price": str(intent.price or 0), "reason": intent.reason}

        result = self._service.send(intent, parent)
        return tuple(_fill(intent.order_id, fact) for fact in result.facts)

    def adopt_exchange_order(self, *, intent_id: str, symbol: str, side: str, order_id: str,
                             client_order_id: str, reason: str) -> bool:
        """Journal a venue-originated reduction (a triggered stop) for this owner. Idempotent per intent."""
        parent = {"order_id": intent_id, "symbol": symbol, "side": side, "price": "0", "reason": reason}
        return self._service.adopt(intent_id=intent_id, symbol=symbol, side=side, order_id=order_id,
                                   client_order_id=client_order_id, parent=parent)

    def cancel_entries(self, symbol: str) -> tuple[Fill, ...]:
        """Cancel resting entries for ``symbol``; returns fills confirmed while doing so."""
        return tuple(_fill(fact.intent_id, fact) for fact in self._service.cancel_entries(symbol))

    def take_refusal(self, intent_id: str) -> str | None:
        """Why the order journal refused to reserve ``intent_id`` (read once), or None."""
        return self._service.take_refusal(intent_id)

    def fill_fact(self, fill_id: str) -> dict[str, Any]:
        return self._service.fill_fact(fill_id)

    def notify_status(self) -> None:
        self._service.notify_status()

    def reconcile_pending(self) -> tuple[Fill, ...]:
        """Collect new exchange fills; returns every fill the owner has not consumed yet."""
        return tuple(Fill(fill_id=row["fill_id"], order_id=row["intent_id"], symbol=row["symbol"], side=row["side"],
                          qty=Decimal(str(row["qty"])), price=Decimal(str(row["price"])))
                     for row in self._service.poll())

    def reconcile_account(self, symbol: str, positions: Mapping[str, PositionView] | None = None) -> list[Issue]:
        """Read-only verification; a blocking issue blocks entries until a clean pass."""
        return self._service.reconcile(symbol, positions)


__all__ = ["Fill", "OwnerExecutor"]
