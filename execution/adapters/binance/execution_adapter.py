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

Since migration step 9d the decisions above live in
``trading.application.execution.service.ExecutionService``. This class only
translates the legacy event types: an order event becomes an ``OrderIntent``
plus the parent record stored with it, and exchange fill facts become
``FillEvent``s. The scope and the journal path are passed in by the caller;
nothing here reads the environment.
"""
from __future__ import annotations

from dataclasses import asdict, replace
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace
from typing import Any

from event.header import EventHeader
from event.types import EventType, FillEvent
from monitoring.tca import TCALogger
from shared.confirmed_execution import ConfirmedExecutor
from trading.application.execution.service import ExecutionService
from trading.domain.orders import OrderIntent

NOT_FOUND_GRACE_SECONDS = 90.0  # > Binance max recvWindow (60s) + clock skew margin
HOUR_SECONDS = 3600.0


class _TcaLog:
    """The EventLog port over the TCA file next to the order journal (one append per fill)."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def write(self, kind: str, **fields: Any) -> None:
        tca = TCALogger("binance", path=self._path)
        try:
            tca.record_fill(**fields)
        finally:
            tca.close()


def _decimal(value: Any) -> Decimal:
    """Decimal(str(value)); a value that cannot be parsed becomes NaN and is refused by the service."""
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return Decimal("NaN")


class BinanceExecutionAdapter:
    venue = "binance"

    def __init__(self, adapter: Any, circuit_breaker: Any = None, *,
                 journal_path: Path | None = None, scope: str | None = None,
                 owner: str = "", wait_seconds: float = 0.5, clock: Any = time.time,
                 limit_offset_bps: Decimal = Decimal("2"), ttl_bars: int = 3,
                 entry_mode: str = "maker") -> None:
        self._adapter, self.owner = adapter, owner
        self._scope = scope or ""
        if not self._scope or self._scope != self._scope.strip():
            raise ValueError("Binance durable execution requires an explicit account/environment RISK_SCOPE")
        self._lock = threading.RLock()
        offset = Decimal(str(limit_offset_bps)) / Decimal(10000)
        if not offset.is_finite() or not 0 <= offset < Decimal("0.01"):
            raise ValueError("limit_offset_bps must be in [0, 100)")
        if ttl_bars < 1:
            raise ValueError("ttl_bars must be positive")
        if entry_mode not in ("maker", "market"):
            raise ValueError("entry_mode must be maker or market")
        if journal_path is None:
            raise ValueError("Binance durable execution requires an explicit order journal path")
        self._orders = ConfirmedExecutor(
            adapter, Path(journal_path), self._scope, owner=owner, wait_seconds=wait_seconds,
            entry_order_type="limit" if entry_mode == "maker" else "market",
            not_found_grace_seconds=NOT_FOUND_GRACE_SECONDS, clock=clock,
            entry_ttl_seconds=ttl_bars * HOUR_SECONDS if entry_mode == "maker" else None,
        )
        self._service = ExecutionService(
            adapter, self._orders, owner=owner, entry_mode=entry_mode, limit_offset=offset,
            lock=threading.RLock(), clock=time.time, monotonic=time.monotonic, breaker=circuit_breaker,
            event_log=_TcaLog(self._orders.path.parent / "tca_binance.jsonl"))
        self._parents: dict[str, Any] = {}
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

    @property
    def _ready(self) -> bool:
        return self._service.ready

    @_ready.setter
    def _ready(self, value: bool) -> None:
        self._service.ready = value

    @property
    def _status_callback(self):
        return self._service.status_callback

    @_status_callback.setter
    def _status_callback(self, callback) -> None:
        self._service.status_callback = callback

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

    def send_order(self, order_event: Any) -> tuple[FillEvent, ...]:
        with self._lock:
            return self._send_order(order_event)

    def _send_order(self, event: Any) -> tuple[FillEvent, ...]:
        intent_id = str(getattr(event, "order_id", "") or "")
        intent = OrderIntent(
            order_id=intent_id, source=getattr(getattr(event, "header", None), "source", None),
            symbol=getattr(event, "symbol", ""), side=str(getattr(event, "side", "")),
            qty=_decimal(getattr(event, "qty", "NaN")), price=_decimal(getattr(event, "price", None) or 0),
            reduce_only=getattr(event, "reduce_only", False) is True, reason=getattr(event, "reason", None))
        stored: dict[str, Any] = {}

        def parent() -> dict[str, Any]:
            header = asdict(event.header)
            header["event_type"] = event.header.event_type.value
            stored.update({"header": header, "order_id": intent_id, "symbol": event.symbol,
                           "side": event.side, "price": str(event.price or 0),
                           "reason": getattr(event, "reason", None)})
            return stored

        result = self._service.send(intent, parent)
        if not result.submitted:
            return ()        # refused, blocked or repeated: the service already did all there is to do
        try:
            self._parents[intent_id] = SimpleNamespace(header=event.header, **{
                k: v for k, v in stored.items() if k != "header"})
            return self._events(intent_id, result.facts)
        except Exception:
            self._service.fail(intent_id)
            return ()

    def adopt_exchange_order(self, *, intent_id: str, symbol: str, side: str, order_id: str,
                             client_order_id: str, reason: str) -> bool:
        """Journal a venue-originated reduction (e.g. a triggered stop) for this owner.

        Idempotent per ``intent_id``. Its trades are then collected by ``poll``
        and consumed like any other confirmed fill.
        """
        with self._lock:
            header = EventHeader.new_root(event_type=EventType.ORDER, version=1, source=self.owner or "binance")
            header = replace(header, event_id=intent_id)
            parent_header = asdict(header)
            parent_header["event_type"] = header.event_type.value
            parent = {"header": parent_header, "order_id": intent_id, "symbol": symbol, "side": side,
                      "price": "0", "reason": reason}
            if not self._service.adopt(intent_id=intent_id, symbol=symbol, side=side, order_id=order_id,
                                       client_order_id=client_order_id, parent=parent):
                return False
            self._parents[intent_id] = SimpleNamespace(header=header, **{
                k: v for k, v in parent.items() if k != "header"})
            return True

    def fill_fact(self, fill_id: str) -> dict[str, Any]:
        return self._service.fill_fact(fill_id)

    def notify_status(self) -> None:
        self._service.notify_status()

    def reconcile_pending(self) -> tuple[FillEvent, ...]:
        with self._lock:
            return tuple(event for fact in self._service.poll()
                         for event in self._events(fact["intent_id"], (SimpleNamespace(**fact),)))

    def reconcile_account(self, symbol: str) -> list[str]:
        """Read-only verification; any mismatch blocks entries and never fabricates fills."""
        return self._service.reconcile(symbol)
