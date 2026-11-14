"""Durable consumption of confirmed fills into the owner's position checkpoint."""
from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace
from typing import Any, Protocol

from trading.application.ports import OrderJournal
from trading.domain.position import StrategyPosition


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
    """Durable consumption of confirmed fills into the owner's position checkpoint."""

    def __init__(self, executor: FillSource) -> None:
        self.executor, self.journal, self.owner = executor, executor.journal, executor.owner
        saved = self.journal.checkpoint(self.owner)
        self.position = StrategyPosition.from_dict(saved["position"]) if saved else StrategyPosition()
        self.revision = saved["revision"] if saved else None

    def consume(self, event: FillEventLike) -> bool:
        fact = self.executor.fill_fact(event.fill_id)
        if (str(event.order_id) != fact["intent_id"] or event.symbol != fact["symbol"] or event.side != fact["side"]
                or event.qty != Decimal(fact["qty"]) or event.price != Decimal(fact["price"])):
            raise ValueError("Event differs from its durable exchange fact")
        before = self.position.to_dict()

        def apply() -> dict[str, Any]:
            self.position.apply(SimpleNamespace(**fact))
            return {"position": self.position.to_dict(), "revision": event.fill_id, "strategy": {}}

        def rollback() -> None:
            self.position = StrategyPosition.from_dict(before)

        changed = self.journal.consume(self.owner, event.fill_id, apply, rollback, expected_revision=self.revision)
        if changed:
            self.revision = event.fill_id
        return changed


__all__ = ["TargetFillConsumer"]
