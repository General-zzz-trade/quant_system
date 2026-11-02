"""OKX execution boundary: emit only order-matched exchange trade facts.

Entries use one post-only order, then cancel the remainder. Automatic market
fallback and splitting are intentionally disabled until recovery is complete.
"""
from __future__ import annotations

import logging
import json
import os
import threading
import time
from dataclasses import replace, asdict
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any
from types import SimpleNamespace

from event.header import EventHeader
from event.types import EventType, FillEvent
from monitoring.tca import TCALogger
from shared.confirmed_execution import ConfirmedExecutor

logger = logging.getLogger(__name__)


class OkxExecutionAdapter:
    def __init__(self, adapter: Any, circuit_breaker: Any = None, *,
                 journal_path: Path | None = None, scope: str | None = None, owner: str = "") -> None:
        self._adapter, self._cb = adapter, circuit_breaker
        self.owner = owner
        self._ready = not bool(owner)  # production owners must complete recovery
        self._status_callback = lambda *_: None
        simulated = bool(getattr(getattr(adapter, "_config", None), "simulated", False))
        self._scope = scope or os.environ.get("RISK_SCOPE", f"okx-{'demo' if simulated else 'live'}-default")
        self._orders = ConfirmedExecutor(
            adapter, journal_path or Path(os.environ.get("ORDER_JOURNAL_PATH", "data/runtime/orders.sqlite")),
            self._scope, owner=owner, wait_seconds=float(os.environ.get("OKX_LIMIT_FILL_TIMEOUT_S", "3.0")),
        )
        self._parents: dict[str, Any] = {}
        self._submitted_at: dict[str, float] = {}
        self._lock = threading.RLock()
        for row in self._orders.journal.orders(owner):
            if row["parent"]:
                data = json.loads(row["parent"])
                header = data.pop("header")
                header["event_type"] = EventType(header["event_type"])
                self._parents[row["intent_id"]] = SimpleNamespace(header=EventHeader(**header), **data)
        if self._orders.unresolved():
            logger.critical("Unresolved OKX journal entries found; affected symbols remain blocked until reconciled")

    def _events(self, intent_id: str, fills: tuple[Any, ...]) -> tuple[FillEvent, ...]:
        parent = self._parents[intent_id]
        events = []
        for fill in fills:
            fill_id = f"{self._scope}:{fill.symbol}:{fill.order_id}:{fill.trade_id}"
            header = EventHeader.from_parent(parent=parent.header, event_type=EventType.FILL,
                                             version=1, source="okx")
            # Stable exchange identity also makes duplicate delivery detectable.
            header = replace(header, event_id=fill_id)
            events.append(FillEvent(
                header=header, fill_id=fill_id, order_id=parent.order_id,
                symbol=fill.symbol, qty=Decimal(str(fill.qty)),
                price=Decimal(str(fill.price)), side=fill.side,
            ))
            # Observability uses confirmed fills too. Reference is the decision
            # price, explicitly labelled so it is not confused with arrival mid.
            tca = None
            try:
                tca = TCALogger("okx", path=self._orders.path.parent / "tca_okx.jsonl")
                latency = (time.monotonic() - self._submitted_at.get(intent_id, time.monotonic())) * 1000
                tca.record_fill(symbol=fill.symbol, side=fill.side, qty=float(fill.qty),
                                ref_price=float(parent.price or 0), fill_price=float(fill.price),
                                latency_ms=latency,
                                order_id=fill.order_id, fill_id=fill_id, ref_source="decision_price")
            except Exception:
                logger.debug("TCA unavailable", exc_info=True)
            finally:
                if tca is not None:
                    tca.close()
        return tuple(events)

    def send_order(self, order_event: Any) -> tuple[FillEvent, ...]:
        with self._lock:
            return self._send_order(order_event)

    def _send_order(self, order_event: Any) -> tuple[FillEvent, ...]:
        try:
            if self.owner and order_event.header.source != self.owner:
                raise ValueError("Order belongs to another strategy")
            qty = Decimal(str(order_event.qty))
            reducing = qty == 0 or getattr(order_event, "reduce_only", False) is True
            if not reducing and (not self._ready or (self._cb is not None and not self._cb.allow_request())):
                self._status_callback(str(order_event.order_id), "blocked")
                return ()
            side = str(order_event.side).lower()
            if qty == 0:
                positions = self._adapter.get_positions(symbol=order_event.symbol)
                position = next((p for p in positions if p.symbol == order_event.symbol and not p.is_flat), None)
                if position is None:
                    self._status_callback(str(order_event.order_id), "blocked")
                    return ()
                qty = abs(Decimal(str(position.qty)))
                side = "sell" if position.qty > 0 else "buy"
                if self.owner:
                    checkpoint = self._orders.journal.checkpoint(self.owner)
                    owned = Decimal(checkpoint["position"]["qty"]) if checkpoint else Decimal(0)
                    if owned == 0 or owned * Decimal(str(position.qty)) <= 0:
                        self._status_callback(str(order_event.order_id), "blocked")
                        return ()
                    qty = min(qty, abs(owned))
            intent_id = str(order_event.order_id)
            self._parents[intent_id] = order_event
            self._submitted_at[intent_id] = time.monotonic()
            if not reducing and datetime.now(timezone.utc).weekday() >= 5:
                factor = Decimal(os.environ.get("OKX_WEEKEND_QTY_FACTOR", "0.7"))
                if not factor.is_finite() or not 0 < factor <= 1:
                    raise ValueError("OKX_WEEKEND_QTY_FACTOR must be in (0, 1]")
                qty *= factor
            header = asdict(order_event.header)
            header["event_type"] = order_event.header.event_type.value
            parent = {"header": header, "order_id": intent_id, "symbol": order_event.symbol,
                      "side": order_event.side, "price": str(order_event.price or 0),
                      "reason": getattr(order_event, "reason", None)}
            facts = self._orders.submit(intent_id=intent_id, symbol=order_event.symbol,
                                        side=side, qty=qty, reduce_only=reducing, parent=parent)
            rows = {r["intent_id"]: r for r in self._orders.journal.orders(self.owner)}
            if intent_id not in rows:
                self._status_callback(intent_id, "blocked")
            events = self._events(intent_id, facts)
            if events and self._cb is not None:
                self._cb.record_success()
            return events
        except Exception:
            if self._cb is not None:
                self._cb.record_failure()
            self._ready = False
            if not any(r["intent_id"] == str(order_event.order_id) for r in self.journal.orders(self.owner)):
                self._status_callback(str(order_event.order_id), "blocked")
            logger.exception("OKX order unresolved or blocked; no automatic retry")
            return ()

    def fill_fact(self, fill_id: str) -> dict[str, Any]:
        with self._orders._connect() as db:
            row = db.execute("SELECT payload FROM execution_fills WHERE scope=? AND fill_id=? AND owner=?",
                             (self._scope, fill_id, self.owner)).fetchone()
            if row is None:
                raise ValueError("Unknown fill for this strategy")
            return json.loads(row[0])

    @property
    def journal(self):
        return self._orders.journal

    def notify_status(self) -> None:
        for row in self.journal.orders(self.owner):
            self._status_callback(row["intent_id"], row["status"])

    def reconcile_pending(self) -> tuple[FillEvent, ...]:
        with self._lock:
            self._orders.poll()
            # Retry until the consumer commits its receipt, including after restart.
            return tuple(event for row in self.journal.fills(self.owner)
                         for event in self._events(row["intent_id"], (SimpleNamespace(**row),)))

    def reconcile_account(self, symbol: str) -> list[str]:
        """Read-only verification; a mismatch blocks entries, never fabricates fills."""
        self._ready = False
        issues = []
        try:
            positions = self._adapter.get_positions(symbol=symbol)
            nonflat = [p for p in positions if p.symbol == symbol and Decimal(str(p.qty)) != 0]
            if len(nonflat) > 1:
                issues.append(f"{symbol}: hedge-mode positions require explicit reconciliation")
            actual = sum((Decimal(str(p.qty)) for p in positions if p.symbol == symbol), Decimal(0))
            expected = Decimal(0)
            owners = {r["owner"] for r in self.journal.orders() if r["symbol"] == symbol}
            for owner in owners:
                checkpoint = self.journal.checkpoint(owner)
                if checkpoint:
                    expected += Decimal(checkpoint["position"]["qty"])
            if not actual.is_finite() or actual != expected:
                issues.append(f"{symbol}: venue qty={actual}, ledger qty={expected}")
            if any(r["symbol"] == symbol for r in self._orders.unresolved()):
                issues.append(f"{symbol}: unresolved orders or legacy journal records")
            known = {r["order_id"] for r in self.journal.orders() if r["symbol"] == symbol}
            for order in self._adapter.get_open_orders(symbol=symbol):
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
