"""Bybit market execution backed by durable intents and confirmed trade facts.

Submission ACKs never create fills. Ambiguous submissions are queried by their
persisted client ID; neither timeout nor restart resubmits the placement.
"""
from __future__ import annotations

from dataclasses import asdict, replace
from decimal import Decimal
import json
import logging
import os
from pathlib import Path
import threading
import time
from types import SimpleNamespace
from typing import Any

from event.header import EventHeader
from event.types import EventType, FillEvent
from monitoring.tca import TCALogger
from shared.confirmed_execution import ConfirmedExecutor

logger = logging.getLogger(__name__)


class BybitExecutionAdapter:
    def __init__(self, adapter: Any, circuit_breaker: Any = None, *,
                 journal_path: Path | None = None, scope: str | None = None,
                 owner: str = "", wait_seconds: float = 0.0) -> None:
        self._adapter, self._cb, self.owner = adapter, circuit_breaker, owner
        self._scope = scope or os.environ.get("RISK_SCOPE", "")
        if not self._scope or self._scope != self._scope.strip():
            raise ValueError("Bybit durable execution requires an explicit account/environment RISK_SCOPE")
        self._ready = not bool(owner)
        self._status_callback = lambda *_: None
        self._lock = threading.RLock()
        self._orders = ConfirmedExecutor(
            adapter, journal_path or Path(os.environ.get("ORDER_JOURNAL_PATH", "data/runtime/orders.sqlite")),
            self._scope, owner=owner, wait_seconds=wait_seconds, entry_order_type="market",
        )
        self._parents: dict[str, Any] = {}
        self._submitted_at: dict[str, float] = {}
        for row in self.journal.orders(owner):
            if row["parent"]:
                data = json.loads(row["parent"])
                header = data.pop("header")
                header["event_type"] = EventType(header["event_type"])
                self._parents[row["intent_id"]] = SimpleNamespace(header=EventHeader(**header), **data)

    @property
    def journal(self):
        return self._orders.journal

    def _events(self, intent_id: str, facts: tuple[Any, ...]) -> tuple[FillEvent, ...]:
        parent = self._parents[intent_id]
        events = []
        for fact in facts:
            header = EventHeader.from_parent(parent=parent.header, event_type=EventType.FILL,
                                             version=1, source="bybit")
            header = replace(header, event_id=fact.fill_id)
            events.append(FillEvent(
                header=header, fill_id=fact.fill_id, order_id=parent.order_id,
                symbol=fact.symbol, qty=Decimal(str(fact.qty)),
                price=Decimal(str(fact.price)), side=fact.side,
            ))
        return tuple(events)

    def _record_fills(self, intent_id: str, facts: tuple[Any, ...]) -> None:
        # Auxiliary logging occurs only for newly inserted facts, not outbox replays.
        if not facts:
            return
        tca = None
        try:
            tca = TCALogger("bybit", path=self._orders.path.parent / "tca_bybit.jsonl")
            parent = self._parents[intent_id]
            for fact in facts:
                latency = (time.monotonic() - self._submitted_at.get(intent_id, time.monotonic())) * 1000
                tca.record_fill(symbol=fact.symbol, side=fact.side, qty=float(fact.qty),
                                ref_price=float(parent.price or 0), fill_price=float(fact.price),
                                latency_ms=latency, order_id=fact.order_id, fill_id=fact.fill_id,
                                ref_source="decision_price")
        except Exception:
            logger.debug("Confirmed fill committed; TCA unavailable", exc_info=True)
        finally:
            if tca is not None:
                try:
                    tca.close()
                except Exception:
                    logger.debug("Confirmed fill committed; TCA close failed", exc_info=True)

    def send_order(self, order_event: Any) -> tuple[FillEvent, ...]:
        with self._lock:
            return self._send_order(order_event)

    def _send_order(self, event: Any) -> tuple[FillEvent, ...]:
        intent_id = str(getattr(event, "order_id", "") or "")
        try:
            if not intent_id:
                raise ValueError("Order intent identity is required")
            if self.owner and event.header.source != self.owner:
                raise ValueError("Order belongs to another strategy")
            # A repeated intent cannot replace its original parent or place again.
            if any(row["intent_id"] == intent_id for row in self.journal.orders()):
                return ()
            qty = Decimal(str(event.qty))
            if not qty.is_finite() or qty < 0:
                raise ValueError("Invalid order quantity")
            reducing = qty == 0 or getattr(event, "reduce_only", False) is True
            if not reducing and (not self._ready or (self._cb is not None and not self._cb.allow_request())):
                self._status_callback(intent_id, "blocked")
                return ()
            side = str(event.side).lower()
            consumer_revision = None
            if reducing:
                positions = self._adapter.get_positions(symbol=event.symbol, strict=True)
                positions = [p for p in positions if p.symbol == event.symbol and Decimal(str(p.qty)) != 0]
                if len(positions) != 1:
                    raise ValueError("Reduction requires one unambiguous venue position")
                actual = Decimal(str(positions[0].qty))
                if not actual.is_finite():
                    raise ValueError("Invalid venue position")
                if qty == 0:
                    qty, side = abs(actual), "sell" if actual > 0 else "buy"
                if side != ("sell" if actual > 0 else "buy"):
                    raise ValueError("Reduction side does not reduce venue position")
                qty = min(qty, abs(actual))
                if self.owner:
                    checkpoint = self.journal.checkpoint(self.owner)
                    owned = Decimal(checkpoint["position"]["qty"]) if checkpoint else Decimal(0)
                    if not owned.is_finite() or owned * actual <= 0:
                        raise ValueError("No confirmed position owned by this strategy")
                    qty = min(qty, abs(owned))
                    consumer_revision = checkpoint["revision"]
            header = asdict(event.header)
            header["event_type"] = event.header.event_type.value
            parent = {"header": header, "order_id": intent_id, "symbol": event.symbol,
                      "side": event.side, "price": str(event.price or 0),
                      "reason": getattr(event, "reason", None)}
            if reducing and self.owner:
                parent["consumer_revision"] = consumer_revision
            self._parents[intent_id] = event
            self._submitted_at[intent_id] = time.monotonic()
            facts = self._orders.submit(intent_id=intent_id, symbol=event.symbol, side=side,
                                        qty=qty, reduce_only=reducing, parent=parent)
            if not any(row["intent_id"] == intent_id for row in self.journal.orders(self.owner)):
                self._status_callback(intent_id, "blocked")
            self._record_fills(intent_id, facts)
            if facts and self._cb is not None:
                self._cb.record_success()
            return self._events(intent_id, facts)
        except Exception:
            self._ready = False
            if self._cb is not None:
                self._cb.record_failure()
            if not any(row["intent_id"] == intent_id for row in self.journal.orders(self.owner)):
                self._status_callback(intent_id, "blocked")
            logger.exception("Bybit order unresolved or blocked; no placement retry")
            return ()

    def fill_fact(self, fill_id: str) -> dict[str, Any]:
        with self.journal.connect() as db:
            row = db.execute("SELECT payload FROM execution_fills WHERE scope=? AND fill_id=? AND owner=?",
                             (self._scope, fill_id, self.owner)).fetchone()
        if row is None:
            raise ValueError("Unknown fill for this strategy")
        return json.loads(row[0])

    def notify_status(self) -> None:
        for row in self.journal.orders(self.owner):
            self._status_callback(row["intent_id"], row["status"])

    def reconcile_pending(self) -> tuple[FillEvent, ...]:
        with self._lock:
            for intent_id, fact in self._orders.poll():
                self._record_fills(intent_id, (fact,))
            return tuple(event for fact in self.journal.fills(self.owner)
                         for event in self._events(fact["intent_id"], (SimpleNamespace(**fact),)))

    def reconcile_account(self, symbol: str) -> list[str]:
        self._ready = False
        issues = []
        try:
            positions = self._adapter.get_positions(symbol=symbol, strict=True)
            nonflat = [p for p in positions if p.symbol == symbol and Decimal(str(p.qty)) != 0]
            if len(nonflat) > 1:
                issues.append(f"{symbol}: hedge-mode positions require explicit reconciliation")
            actual = sum((Decimal(str(p.qty)) for p in nonflat), Decimal(0))
            expected = Decimal(0)
            for owner in {r["owner"] for r in self.journal.orders() if r["symbol"] == symbol}:
                checkpoint = self.journal.checkpoint(owner)
                if checkpoint:
                    expected += Decimal(checkpoint["position"]["qty"])
            if not actual.is_finite() or not expected.is_finite() or actual != expected:
                issues.append(f"{symbol}: venue qty={actual}, ledger qty={expected}")
            if any(row["symbol"] == symbol for row in self._orders.unresolved()):
                issues.append(f"{symbol}: unresolved orders or legacy journal records")
            known = {row["order_id"] for row in self.journal.orders() if row["symbol"] == symbol}
            for order in self._adapter.get_open_orders(symbol=symbol, strict=True):
                if order.order_id not in known:
                    issues.append(f"{symbol}: unowned open order {order.order_id}")
            if self.journal.fills(self.owner):
                issues.append(f"{symbol}: undelivered fills")
        except Exception as exc:
            issues.append(f"{symbol}: reconciliation failed ({type(exc).__name__})")
        self._ready = not issues
        if issues:
            logger.error("Entry blocked: %s", "; ".join(issues))
        return issues
