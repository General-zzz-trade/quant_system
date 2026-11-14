"""Confirmed order lifecycle with durable submission reservations.

No automatic resubmission, market fallback, split orders, or invented fills.
Orders and fills are restored by owner. Legacy rows lacking recovery metadata
remain blocked; deleting the journal is NOT a recovery procedure.

Optional absence resolution (``not_found_grace_seconds``): a venue that
rejects any request older than its receive window can never accept an order
later than that window. Once the grace period (which must exceed the window)
has passed and the venue still reports no such order, the intent is settled
as ``not_found`` so later orders, including exits, are not blocked forever.
A definitive venue rejection (``status == "rejected"``) confirmed absent by a
lookup is settled as ``rejected`` immediately. Venues that do not opt in keep
the original never-resolve behaviour.
"""
from __future__ import annotations

import hashlib
import logging
import math
import threading
import time
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any
from types import SimpleNamespace

from shared.execution_journal import ExecutionJournal
from trading.adapters.sqlite.order_journal import ENTRY_UNRESOLVED
from trading.domain.orders import VENUE_TERMINAL, IllegalTransition, check_transition

if TYPE_CHECKING:
    from trading.application.ports import Clock, OrderGateway

logger = logging.getLogger(__name__)
TERMINAL = frozenset(status.value for status in VENUE_TERMINAL)
# Transitions outside trading.domain.orders.TRANSITIONS are logged and recorded here, never acted on
# (observation mode, migration step 5). The test suite fails any test that records one.
TRANSITION_VIOLATIONS: list[str] = []


def _record_transition(current: str | None, new: str, intent_id: str) -> None:
    try:
        check_transition(current, new)
    except IllegalTransition as exc:
        TRANSITION_VIOLATIONS.append(f"{intent_id}: {exc}")
        logger.error("Order %s: %s", intent_id, exc)


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
    created_at: float | None = None
    rejected: bool = False
    reducing: bool = False


class ConfirmedExecutor:
    def __init__(self, adapter: OrderGateway, journal_path: Path, scope: str, *,
                 wait_seconds: float = 3.0, owner: str = "", entry_order_type: str = "limit",
                 not_found_grace_seconds: float | None = None, clock: Clock = time.time,
                 entry_ttl_seconds: float | None = None) -> None:
        if not scope:
            raise ValueError("execution scope is required")
        self.adapter, self.path, self.scope = adapter, Path(journal_path), scope
        if not math.isfinite(wait_seconds) or wait_seconds < 0:
            raise ValueError("wait_seconds must be finite and nonnegative")
        self.wait_seconds = wait_seconds
        if entry_order_type not in {"limit", "market"}:
            raise ValueError("entry_order_type must be limit or market")
        self.entry_order_type = entry_order_type
        if not_found_grace_seconds is not None and (
                not math.isfinite(not_found_grace_seconds) or not_found_grace_seconds < 10):
            raise ValueError("not_found_grace_seconds must be at least 10 seconds")
        self.not_found_grace = not_found_grace_seconds
        if entry_ttl_seconds is not None and (not math.isfinite(entry_ttl_seconds) or entry_ttl_seconds <= 0):
            raise ValueError("entry_ttl_seconds must be positive")
        # None: cancel a limit entry right after the first observation (legacy).
        # Set: a resting post-only entry is canceled only once it expires.
        self.entry_ttl = entry_ttl_seconds
        self.clock = clock
        self.pending: dict[str, PendingOrder] = {}
        self._lock = threading.RLock()
        self.owner = owner
        self.journal = ExecutionJournal(self.path, scope)
        self._restore_pending()

    def _connect(self):  # raw access kept for the frozen OKX adapter only
        return self.journal.connect()

    def _restore_pending(self) -> None:
        for row in self.journal.orders(self.owner, unresolved=True):
            # Old reservations lack side/quantity. Never guess how to recover them.
            if not row["side"] or not row["qty"]:
                continue
            order_type = row["order_type"] or ("market" if row["reducing"] else "limit")
            pending = PendingOrder(row["intent_id"], row["symbol"], row["side"], row["client_id"],
                                   row["order_id"], is_limit=order_type == "limit",
                                   created_at=row.get("created_at"), rejected=row["status"] == "rejected_unconfirmed",
                                   reducing=bool(row["reducing"]))
            for fact in self.journal.intent_fills(pending.intent_id):
                pending.seen.add(fact["trade_id"])
                pending.confirmed_qty += Decimal(fact["qty"])
            self.pending[pending.intent_id] = pending

    def _reserve(self, intent_id: str, symbol: str, side: str, reducing: bool,
                 qty: Decimal, parent: dict[str, Any] | None) -> PendingOrder | None:
        client_id = "qs" + hashlib.sha256(f"{self.scope}:{intent_id}".encode()).hexdigest()[:30]
        refusal, created_at = self.journal.reserve(
            intent_id=intent_id, symbol=symbol, client_id=client_id, side=side, qty=str(qty), reducing=reducing,
            owner=self.owner, parent=parent, order_type="market" if reducing else self.entry_order_type,
            clock=self.clock)
        if refusal:
            if refusal == ENTRY_UNRESOLVED:
                logger.error("Entry blocked: unresolved submission for %s (scope=%s)", symbol, self.scope)
            return None
        pending = PendingOrder(intent_id, symbol, side, client_id,
                               is_limit=not reducing and self.entry_order_type == "limit",
                               created_at=created_at, reducing=reducing)
        self.pending[intent_id] = pending
        return pending

    def submit(self, *, intent_id: str, symbol: str, side: str, qty: Decimal,
               reduce_only: bool = False, parent: dict[str, Any] | None = None,
               limit_price: Decimal | None = None) -> tuple[Any, ...]:
        qty = Decimal(str(qty))
        side = side.lower()
        if not intent_id or not symbol or side not in ("buy", "sell") or not qty.is_finite() or qty <= 0:
            raise ValueError("invalid order intent")
        with self._lock:
            # Fetch quote before reservation: a failed quote cannot have sent an order.
            price = None
            if not reduce_only and self.entry_order_type == "limit":
                if limit_price is not None:
                    price = Decimal(str(limit_price))
                else:
                    ticker = self.adapter.get_ticker(symbol)
                    price = Decimal(str(ticker.get("bid1Price" if side == "buy" else "ask1Price", 0)))
                if not price.is_finite() or price <= 0:
                    raise ValueError("entry quote unavailable")
            pending = self._reserve(intent_id, symbol, side, reduce_only, qty, parent)
            if pending is None:
                return ()
            try:
                if reduce_only or self.entry_order_type == "market":
                    response = self.adapter.send_market_order(
                        symbol, side, float(qty), reduce_only=reduce_only, client_order_id=pending.client_id,
                    )
                else:
                    response = self.adapter.send_limit_order(
                        symbol, side, float(qty), float(price), post_only=True,
                        client_order_id=pending.client_id,
                    )
                pending.order_id = str(response.get("orderId") or "")
                self._save_order_id(pending)
                if response.get("status") == "submitted":
                    self._set_status(pending, "submitted")
                elif response.get("status") == "rejected" and self.not_found_grace is not None:
                    # Definitive venue rejection; settle only after a lookup confirms absence.
                    pending.rejected = True
                    self._set_status(pending, "rejected_unconfirmed")
                    logger.error("Venue rejected %s: %s %s", intent_id, response.get("code"), response.get("msg"))
                else:
                    self._set_status(pending, "unknown")
                    logger.error("Submission not confirmed for %s; reconciliation required", intent_id)
            except Exception:
                # Even a timeout can follow a successful exchange placement.
                logger.exception("Ambiguous submission %s; will query, never resend", intent_id)
                self._set_status(pending, "unknown")
            if self.wait_seconds > 0:
                time.sleep(self.wait_seconds)
            return self._observe(pending, cancel_limit=self._expired(pending))

    def track(self, pending: PendingOrder) -> None:
        """Follow a venue-originated order (an adopted stop) that the journal already holds."""
        with self._lock:
            self.pending[pending.intent_id] = pending

    def _save_order_id(self, pending: PendingOrder) -> None:
        self.journal.set_order_id(pending.intent_id, pending.order_id)

    def _set_status(self, pending: PendingOrder, status: str) -> None:
        previous = self.journal.update_status(pending.intent_id, status, settle=False)
        if previous is not None:
            _record_transition(previous, status, pending.intent_id)

    def _settle(self, pending: PendingOrder, status: str) -> None:
        previous = self.journal.update_status(pending.intent_id, status, settle=True)
        if previous is not None:
            _record_transition(previous, status, pending.intent_id)
        self.pending.pop(pending.intent_id, None)

    def _observe(self, pending: PendingOrder, *, cancel_limit: bool) -> tuple[Any, ...]:
        facts: list[Any] = []
        try:
            state = self.adapter.get_order(pending.symbol, order_id=pending.order_id,
                                           client_order_id=pending.client_id)
            if state is None:
                if self._absence_is_final(pending):
                    status = "rejected" if pending.rejected else "not_found"
                    self._settle(pending, status)
                    logger.warning("Order %s settled as %s: venue has no such order", pending.intent_id, status)
                    return ()
                self._set_status(pending, "rejected_unconfirmed" if pending.rejected else "unknown")
                return ()  # absence from a response is not proof of rejection/cancellation
            if (str(state.client_order_id) != pending.client_id or not state.order_id
                    or (pending.order_id and str(state.order_id) != pending.order_id)
                    or getattr(state, "symbol", pending.symbol) != pending.symbol
                    or str(getattr(state, "side", pending.side)).lower() != pending.side):
                raise ValueError("order lookup identity mismatch")
            pending.order_id = str(state.order_id)
            self._save_order_id(pending)
            if cancel_limit and pending.is_limit and self._state(state) not in TERMINAL:
                # Cancellation ACK is not terminal state. Always query again.
                try:
                    self.adapter.cancel_order(pending.symbol, pending.order_id)
                except Exception:
                    logger.warning("Cancellation unconfirmed for %s", pending.order_id, exc_info=True)
                state = self.adapter.get_order(pending.symbol, order_id=pending.order_id,
                                               client_order_id=pending.client_id)
            fills = self.adapter.get_recent_fills(symbol=pending.symbol, order_id=pending.order_id, limit=100)
            for fill in fills:
                if str(fill.order_id) != pending.order_id or fill.symbol != pending.symbol:
                    continue
                if str(fill.side).lower() != pending.side:
                    raise ValueError("exchange fill side contradicts confirmed order")
                trade_id = str(getattr(fill, "trade_id", "") or "")
                qty, price = Decimal(str(fill.qty)), Decimal(str(fill.price))
                fee = Decimal(str(getattr(fill, "fee", 0) or 0))
                if (not trade_id or not qty.is_finite() or not price.is_finite()
                        or not fee.is_finite() or qty <= 0 or price <= 0):
                    raise ValueError("invalid exchange fill fact")
                if trade_id in pending.seen:
                    fill_id = f"{self.scope}:{pending.symbol}:{pending.order_id}:{trade_id}"
                    stored = self.journal.fill(fill_id)
                    if (stored is None or Decimal(stored["qty"]) != qty or Decimal(stored["price"]) != price
                            or Decimal(stored["fee"]) != fee
                            or stored["fee_asset"] != getattr(fill, "fee_asset", None)):
                        raise ValueError("exchange trade identity has conflicting facts")
                    continue
                payload = {
                    "fill_id": f"{self.scope}:{pending.symbol}:{pending.order_id}:{trade_id}",
                    "intent_id": pending.intent_id, "symbol": pending.symbol, "order_id": pending.order_id,
                    "trade_id": trade_id, "side": pending.side, "qty": str(qty), "price": str(price),
                    "fee": str(fee),
                    "fee_asset": getattr(fill, "fee_asset", None),
                }
                # Insertion is both the durable dedup record and the outbox.
                inserted = self.journal.insert_fill(fill_id=payload["fill_id"], intent_id=pending.intent_id,
                                                    owner=self.owner, payload=payload)
                pending.seen.add(trade_id)
                pending.confirmed_qty += qty
                if inserted:
                    facts.append(SimpleNamespace(**{**payload, "qty": qty, "price": price}))
            if (state is not None and str(state.order_id) == pending.order_id
                    and str(state.client_order_id) == pending.client_id and self._state(state) in TERMINAL):
                filled_qty = Decimal(str(state.filled_qty))
                # If history is truncated/lagging, keep the entry lock instead of
                # fabricating the missing quantity from the order's average price.
                if filled_qty.is_finite() and filled_qty >= 0 and pending.confirmed_qty == filled_qty:
                    self._settle(pending, self._state(state))
        except Exception:
            logger.warning("Order %s remains unresolved", pending.intent_id, exc_info=True)
            self._set_status(pending, "unknown")
        return tuple(facts)

    def _absence_is_final(self, pending: PendingOrder) -> bool:
        """Absence is final only for opted-in venues, after a confirmed rejection
        or once the receive-window grace period has fully elapsed."""
        if self.not_found_grace is None or pending.order_id or pending.confirmed_qty or pending.seen:
            return False
        if pending.rejected:
            return True
        return pending.created_at is not None and self.clock() - pending.created_at >= self.not_found_grace

    @staticmethod
    def _state(order: Any) -> str:
        return str(getattr(order, "state", getattr(order, "status", "unknown")))

    def _expired(self, pending: PendingOrder) -> bool:
        if self.entry_ttl is None:
            return True
        return pending.created_at is None or self.clock() - pending.created_at >= self.entry_ttl

    def poll(self) -> tuple[tuple[str, Any], ...]:
        with self._lock:
            return tuple((pending.intent_id, fill)
                         for pending in list(self.pending.values())
                         for fill in self._observe(pending, cancel_limit=self._expired(pending)))

    def cancel_entries(self, symbol: str) -> tuple[tuple[str, Any], ...]:
        """Cancel resting entry remainders for ``symbol`` (before a reduction)."""
        with self._lock:
            return tuple((pending.intent_id, fill)
                         for pending in list(self.pending.values())
                         if pending.symbol == symbol and pending.is_limit and not pending.reducing
                         for fill in self._observe(pending, cancel_limit=True))

    def unresolved(self) -> list[dict[str, Any]]:
        return self.journal.orders(unresolved=True)
