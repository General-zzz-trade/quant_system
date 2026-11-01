"""Conservative OKX order lifecycle with durable submission reservations.

No automatic resubmission, market fallback, split orders, or invented fills.
Unresolved orders from a previous process deliberately require operator
reconciliation; deleting the journal is NOT a recovery procedure.
"""
from __future__ import annotations

import hashlib
import logging
import math
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)
TERMINAL = frozenset({"filled", "canceled", "mmp_canceled"})


@dataclass
class PendingOrder:
    intent_id: str
    symbol: str
    side: str
    client_id: str
    order_id: str = ""
    is_limit: bool = True
    seen: set[str] = field(default_factory=set)
    confirmed_qty: Decimal = Decimal("0")


class ConfirmedExecutor:
    def __init__(self, adapter: Any, journal_path: Path, scope: str, *, wait_seconds: float = 3.0) -> None:
        if not scope:
            raise ValueError("execution scope is required")
        self.adapter, self.path, self.scope = adapter, Path(journal_path), scope
        if not math.isfinite(wait_seconds) or wait_seconds < 0:
            raise ValueError("wait_seconds must be finite and nonnegative")
        self.wait_seconds = wait_seconds
        self.pending: dict[str, PendingOrder] = {}
        self._lock = threading.RLock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS submissions (
                scope TEXT, intent_id TEXT, symbol TEXT, client_id TEXT, order_id TEXT,
                settled INTEGER NOT NULL DEFAULT 0, PRIMARY KEY(scope, intent_id))""")

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(self.path, timeout=5)
        try:
            with db:
                yield db
        finally:
            db.close()

    def _reserve(self, intent_id: str, symbol: str, side: str, reducing: bool) -> PendingOrder | None:
        client_id = "qs" + hashlib.sha256(f"{self.scope}:{intent_id}".encode()).hexdigest()[:30]
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT 1 FROM submissions WHERE scope=? AND intent_id=?",
                          (self.scope, intent_id)).fetchone():
                return None
            if not reducing and db.execute(
                "SELECT 1 FROM submissions WHERE scope=? AND symbol=? AND settled=0",
                (self.scope, symbol),
            ).fetchone():
                logger.error("Entry blocked: unresolved submission for %s (scope=%s)", symbol, self.scope)
                return None
            db.execute("INSERT INTO submissions VALUES (?, ?, ?, ?, '', 0)",
                       (self.scope, intent_id, symbol, client_id))
        pending = PendingOrder(intent_id, symbol, side, client_id, is_limit=not reducing)
        self.pending[intent_id] = pending
        return pending

    def submit(self, *, intent_id: str, symbol: str, side: str, qty: Decimal,
               reduce_only: bool = False) -> tuple[Any, ...]:
        qty = Decimal(str(qty))
        side = side.lower()
        if not intent_id or not symbol or side not in ("buy", "sell") or not qty.is_finite() or qty <= 0:
            raise ValueError("invalid order intent")
        with self._lock:
            # Fetch quote before reservation: a failed quote cannot have sent an order.
            price = None
            if not reduce_only:
                ticker = self.adapter.get_ticker(symbol)
                price = Decimal(str(ticker.get("bid1Price" if side == "buy" else "ask1Price", 0)))
                if not price.is_finite() or price <= 0:
                    raise ValueError("entry quote unavailable")
            pending = self._reserve(intent_id, symbol, side, reduce_only)
            if pending is None:
                return ()
            try:
                if reduce_only:
                    response = self.adapter.send_market_order(
                        symbol, side, float(qty), reduce_only=True, client_order_id=pending.client_id,
                    )
                else:
                    response = self.adapter.send_limit_order(
                        symbol, side, float(qty), float(price), post_only=True,
                        client_order_id=pending.client_id,
                    )
                pending.order_id = str(response.get("orderId") or "")
                self._save_order_id(pending)
                if response.get("status") != "submitted":
                    logger.error("Submission not confirmed for %s; reconciliation required", intent_id)
            except Exception:
                # Even a timeout can follow a successful exchange placement.
                logger.exception("Ambiguous submission %s; will query, never resend", intent_id)
            if self.wait_seconds > 0:
                time.sleep(self.wait_seconds)
            return self._observe(pending, cancel_limit=True)

    def _save_order_id(self, pending: PendingOrder) -> None:
        with self._connect() as db:
            db.execute("UPDATE submissions SET order_id=? WHERE scope=? AND intent_id=?",
                       (pending.order_id, self.scope, pending.intent_id))

    def _observe(self, pending: PendingOrder, *, cancel_limit: bool) -> tuple[Any, ...]:
        facts: list[Any] = []
        try:
            state = self.adapter.get_order(pending.symbol, order_id=pending.order_id,
                                           client_order_id=pending.client_id)
            if state is None:
                return ()  # absence from a response is not proof of rejection/cancellation
            if (str(state.client_order_id) != pending.client_id or not state.order_id
                    or (pending.order_id and str(state.order_id) != pending.order_id)):
                raise ValueError("order lookup identity mismatch")
            pending.order_id = str(state.order_id)
            self._save_order_id(pending)
            if cancel_limit and pending.is_limit and state.state not in TERMINAL:
                # Cancellation ACK is not terminal state. Always query again.
                try:
                    self.adapter.cancel_order(pending.symbol, pending.order_id)
                except Exception:
                    logger.warning("Cancellation unconfirmed for %s", pending.order_id, exc_info=True)
                state = self.adapter.get_order(pending.symbol, order_id=pending.order_id,
                                               client_order_id=pending.client_id)
            fills = self.adapter.get_recent_fills(symbol=pending.symbol, order_id=pending.order_id, limit=100)
            for fill in fills:
                if (str(fill.order_id) != pending.order_id or fill.symbol != pending.symbol
                        or str(fill.side).lower() != pending.side):
                    continue
                trade_id = str(getattr(fill, "trade_id", "") or "")
                qty, price = Decimal(str(fill.qty)), Decimal(str(fill.price))
                if not trade_id or not qty.is_finite() or not price.is_finite() or qty <= 0 or price <= 0:
                    continue
                if trade_id in pending.seen:
                    continue
                pending.seen.add(trade_id)
                pending.confirmed_qty += qty
                facts.append(fill)
            if (state is not None and str(state.order_id) == pending.order_id
                    and str(state.client_order_id) == pending.client_id and state.state in TERMINAL):
                filled_qty = Decimal(str(state.filled_qty))
                # If history is truncated/lagging, keep the entry lock instead of
                # fabricating the missing quantity from the order's average price.
                if filled_qty.is_finite() and filled_qty >= 0 and pending.confirmed_qty == filled_qty:
                    with self._connect() as db:
                        db.execute("UPDATE submissions SET settled=1 WHERE scope=? AND intent_id=?",
                                   (self.scope, pending.intent_id))
                    self.pending.pop(pending.intent_id, None)
        except Exception:
            logger.warning("Order %s remains unresolved", pending.intent_id, exc_info=True)
        return tuple(facts)

    def poll(self) -> tuple[tuple[str, Any], ...]:
        with self._lock:
            return tuple((pending.intent_id, fill)
                         for pending in list(self.pending.values())
                         for fill in self._observe(pending, cancel_limit=True))

    def unresolved(self) -> list[dict[str, Any]]:
        with self._connect() as db:
            db.row_factory = sqlite3.Row
            return [dict(row) for row in db.execute(
                "SELECT * FROM submissions WHERE scope=? AND settled=0", (self.scope,))]
