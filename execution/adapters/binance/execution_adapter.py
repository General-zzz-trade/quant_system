# execution/adapters/binance/execution_adapter.py
"""Binance USDⓈ-M execution backed by durable intents and confirmed trade facts.

Same contract as the Bybit durable adapter:

- Every order intent is reserved in the SQLite journal (deterministic client
  id) BEFORE the request is sent; a timeout is answered by querying that id,
  never by resending. There are no retries, splits or market fallbacks.
- Only exchange trade records (``userTrades`` for the order id) become
  ``FillEvent``s; a submission ACK never does.
- Reductions (qty=0 close or reduce_only) are sized from the venue position
  and never exceed the quantity this strategy owner has confirmed.
- Binance rejects requests older than its receive window, so an intent that is
  still absent after ``NOT_FOUND_GRACE_SECONDS`` is settled as ``not_found``
  and stops blocking later exits (see ``shared.confirmed_execution``).
- Entries mirror the research fill model (``runner.offline.broker``): one
  post-only (GTX) limit at ``reference x (1 -/+ offset)`` that rests for up to
  ``ttl_bars`` hours, partial fills allowed; the remainder is canceled at
  expiry or before any reduction. Reductions are reduce-only market orders.
- ``adopt_exchange_order`` brings a venue-originated order (a triggered
  protective stop) into the same journal so its fills reach the strategy
  ledger exactly once.
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
from shared.confirmed_execution import ConfirmedExecutor, PendingOrder

logger = logging.getLogger(__name__)

NOT_FOUND_GRACE_SECONDS = 90.0  # > Binance max recvWindow (60s) + clock skew margin
HOUR_SECONDS = 3600.0


class BinanceExecutionAdapter:
    venue = "binance"

    def __init__(self, adapter: Any, circuit_breaker: Any = None, *,
                 journal_path: Path | None = None, scope: str | None = None,
                 owner: str = "", wait_seconds: float = 0.5, clock: Any = time.time,
                 limit_offset_bps: Decimal = Decimal("2"), ttl_bars: int = 3,
                 entry_mode: str = "maker") -> None:
        self._adapter, self._cb, self.owner = adapter, circuit_breaker, owner
        self._scope = scope or os.environ.get("RISK_SCOPE", "")
        if not self._scope or self._scope != self._scope.strip():
            raise ValueError("Binance durable execution requires an explicit account/environment RISK_SCOPE")
        self._ready = not bool(owner)
        self._status_callback = lambda *_: None
        self._lock = threading.RLock()
        self._offset = Decimal(str(limit_offset_bps)) / Decimal(10000)
        if not self._offset.is_finite() or not 0 <= self._offset < Decimal("0.01"):
            raise ValueError("limit_offset_bps must be in [0, 100)")
        if ttl_bars < 1:
            raise ValueError("ttl_bars must be positive")
        if entry_mode not in ("maker", "market"):
            raise ValueError("entry_mode must be maker or market")
        self._entry_mode = entry_mode
        self._orders = ConfirmedExecutor(
            adapter, journal_path or Path(os.environ.get("ORDER_JOURNAL_PATH", "data/runtime/orders.sqlite")),
            self._scope, owner=owner, wait_seconds=wait_seconds,
            entry_order_type="limit" if entry_mode == "maker" else "market",
            not_found_grace_seconds=NOT_FOUND_GRACE_SECONDS, clock=clock,
            entry_ttl_seconds=ttl_bars * HOUR_SECONDS if entry_mode == "maker" else None,
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

    @property
    def scope(self) -> str:
        return self._scope

    def _events(self, intent_id: str, facts: tuple[Any, ...]) -> tuple[FillEvent, ...]:
        parent = self._parents[intent_id]
        events = []
        for fact in facts:
            header = EventHeader.from_parent(parent=parent.header, event_type=EventType.FILL,
                                             version=1, source="binance")
            header = replace(header, event_id=fact.fill_id)
            events.append(FillEvent(
                header=header, fill_id=fact.fill_id, order_id=parent.order_id,
                symbol=fact.symbol, qty=Decimal(str(fact.qty)),
                price=Decimal(str(fact.price)), side=fact.side,
                fee=Decimal(str(getattr(fact, "fee", 0) or 0)),
                fee_asset=getattr(fact, "fee_asset", None),
            ))
        return tuple(events)

    def _record_fills(self, intent_id: str, facts: tuple[Any, ...]) -> None:
        if not facts:
            return
        tca = None
        try:
            tca = TCALogger("binance", path=self._orders.path.parent / "tca_binance.jsonl")
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
                    logger.debug("TCA close failed", exc_info=True)

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
            if any(row["intent_id"] == intent_id for row in self.journal.orders()):
                return ()  # a repeated intent never places again
            qty = Decimal(str(event.qty))
            if not qty.is_finite() or qty < 0:
                raise ValueError("Invalid order quantity")
            reducing = qty == 0 or getattr(event, "reduce_only", False) is True
            if not reducing and (not self._ready or (self._cb is not None and not self._cb.allow_request())):
                self._status_callback(intent_id, "blocked")
                return ()
            side = str(event.side).lower()
            consumer_revision = None
            limit_price = None
            if not reducing and self._entry_mode == "maker":
                reference = Decimal(str(event.price or 0))
                if not reference.is_finite() or reference <= 0:
                    raise ValueError("Entry requires a positive reference price")
                limit_price = reference * (1 - (self._offset if side == "buy" else -self._offset))
            if reducing:
                # Never leave an entry remainder able to re-open after an exit.
                for cancelled_intent, fact in self._orders.cancel_entries(event.symbol):
                    self._record_fills(cancelled_intent, (fact,))
                positions = [p for p in self._adapter.get_positions(symbol=event.symbol, strict=True)
                             if p.symbol == event.symbol and Decimal(str(p.qty)) != 0]
                if len(positions) != 1:
                    raise ValueError("Reduction requires one unambiguous venue position")
                actual = Decimal(str(positions[0].qty))
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
            self._parents[intent_id] = SimpleNamespace(header=event.header, **{
                k: v for k, v in parent.items() if k != "header"})
            self._submitted_at[intent_id] = time.monotonic()
            facts = self._orders.submit(intent_id=intent_id, symbol=event.symbol, side=side,
                                        qty=qty, reduce_only=reducing, parent=parent, limit_price=limit_price)
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
            logger.exception("Binance order unresolved or blocked; no placement retry")
            return ()

    def adopt_exchange_order(self, *, intent_id: str, symbol: str, side: str, order_id: str,
                             client_order_id: str, reason: str) -> bool:
        """Journal a venue-originated reduction (e.g. a triggered stop) for this owner.

        Idempotent per ``intent_id``. Its trades are then collected by ``poll``
        and consumed like any other confirmed fill.
        """
        with self._lock:
            if any(row["intent_id"] == intent_id for row in self.journal.orders()):
                return False
            header = EventHeader.new_root(event_type=EventType.ORDER, version=1, source=self.owner or "binance")
            header = replace(header, event_id=intent_id)
            parent_header = asdict(header)
            parent_header["event_type"] = header.event_type.value
            parent = {"header": parent_header, "order_id": intent_id, "symbol": symbol, "side": side,
                      "price": "0", "reason": reason}
            with self.journal.connect() as db:
                db.execute("BEGIN IMMEDIATE")
                db.execute("""INSERT INTO submissions
                    (scope,intent_id,symbol,client_id,order_id,settled,side,qty,reducing,owner,parent,status,
                     order_type,created_at) VALUES (?,?,?,?,?,0,?,?,1,?,?,'submitted','market',?)""",
                    (self._scope, intent_id, symbol, client_order_id, order_id, side, "0", self.owner,
                     json.dumps(parent), time.time()))
            self._parents[intent_id] = SimpleNamespace(header=header, **{
                k: v for k, v in parent.items() if k != "header"})
            self._orders.pending[intent_id] = PendingOrder(intent_id, symbol, side, client_order_id, order_id,
                                                           is_limit=False, created_at=time.time())
            return True

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
        """Read-only verification; any mismatch blocks entries and never fabricates fills."""
        self._ready = False
        issues = list(getattr(self._adapter, "execution_blockers", lambda: [])())
        try:
            positions = self._adapter.get_positions(symbol=symbol, strict=True)
            nonflat = [p for p in positions if p.symbol == symbol and Decimal(str(p.qty)) != 0]
            if len(nonflat) > 1:
                issues.append(f"{symbol}: multiple positions require explicit reconciliation")
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
