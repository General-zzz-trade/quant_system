"""Order lifecycle management for market maker.

Tracks live orders, detects staleness, and submits/cancels via WS-API.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass

from .config import MarketMakerConfig

log = logging.getLogger(__name__)


@dataclass
class LiveOrder:
    """A single live limit order on the exchange."""
    client_order_id: str
    side: str              # "BUY" or "SELL"
    price: float
    qty: float
    submitted_at: float    # time.monotonic()
    exchange_order_id: str | None = None
    filled_qty: float = 0.0
    status: str = "pending"  # pending, new, partially_filled, filled, cancelled


class OrderManager:
    """Manage bid/ask limit orders on Binance via WS-API.

    Maintains at most one bid and one ask. On each quote update:
      1. Cancel stale orders (too old or too far from BBO)
      2. Submit new orders at target prices
    """

    def __init__(self, cfg: MarketMakerConfig, gateway=None) -> None:
        self._cfg = cfg
        self._gw = gateway  # BinanceWsOrderGateway or None for dry-run
        self._orders: dict[str, LiveOrder] = {}  # client_order_id → LiveOrder
        self._bid_coid: str | None = None
        self._ask_coid: str | None = None
        self._pending_cancels: set[str] = set()

    @property
    def live_bid(self) -> LiveOrder | None:
        if self._bid_coid and self._bid_coid in self._orders:
            o = self._orders[self._bid_coid]
            if o.status in ("pending", "new", "partially_filled"):
                return o
        return None

    @property
    def live_ask(self) -> LiveOrder | None:
        if self._ask_coid and self._ask_coid in self._orders:
            o = self._orders[self._ask_coid]
            if o.status in ("pending", "new", "partially_filled"):
                return o
        return None

    def update_quotes(
        self,
        target_bid: float | None,
        target_ask: float | None,
        bid_size: float,
        ask_size: float,
        best_bid: float,
        best_ask: float,
    ) -> None:
        """Cancel stale orders, submit new ones at target prices.

        Args:
            target_bid/ask: desired price (None = don't quote this side)
            bid_size/ask_size: order quantity
            best_bid/best_ask: current BBO for staleness check
        """
        now = time.monotonic()
        tick = self._cfg.tick_size
        stale_dist = self._cfg.stale_tick_distance * tick

        # ── Cancel stale bid ────────────────────────────────
        lb = self.live_bid
        if lb is not None:
            age = now - lb.submitted_at
            far_from_bbo = best_bid > 0 and abs(lb.price - best_bid) > stale_dist
            price_changed = target_bid is not None and abs(lb.price - target_bid) >= tick
            if age > self._cfg.stale_order_s or far_from_bbo or price_changed:
                self._cancel(lb.client_order_id)
                lb = None

        # ── Cancel stale ask ────────────────────────────────
        la = self.live_ask
        if la is not None:
            age = now - la.submitted_at
            far_from_bbo = best_ask > 0 and abs(la.price - best_ask) > stale_dist
            price_changed = target_ask is not None and abs(la.price - target_ask) >= tick
            if age > self._cfg.stale_order_s or far_from_bbo or price_changed:
                self._cancel(la.client_order_id)
                la = None

        # ── Submit new bid ──────────────────────────────────
        if lb is None and target_bid is not None:
            self._submit("BUY", target_bid, bid_size)

        # ── Submit new ask ──────────────────────────────────
        if la is None and target_ask is not None:
            self._submit("SELL", target_ask, ask_size)

    def cancel_all(self) -> int:
        """Cancel all live orders. Returns count of cancels sent."""
        count = 0
        for coid, order in list(self._orders.items()):
            if order.status in ("pending", "new", "partially_filled"):
                self._cancel(coid)
                count += 1
        return count

    def on_order_response(self, resp: dict) -> None:
        """Handle order submission acknowledgement from WS-API."""
        coid = resp.get("clientOrderId") or resp.get("newClientOrderId")
        if not coid or coid not in self._orders:
            return
        order = self._orders[coid]
        order.exchange_order_id = str(resp.get("orderId", ""))
        status = resp.get("status", "").upper()
        if status == "NEW":
            order.status = "new"
        elif status == "PARTIALLY_FILLED":
            order.status = "partially_filled"
        elif status == "FILLED":
            order.status = "filled"
        elif status in ("CANCELED", "CANCELLED", "EXPIRED"):
            order.status = "cancelled"

    def on_fill(self, coid: str, filled_qty: float) -> None:
        """Record a fill against an order."""
        if coid not in self._orders:
            return
        order = self._orders[coid]
        order.filled_qty += filled_qty
        if order.filled_qty >= order.qty - 1e-12:
            order.status = "filled"
            self._clear_slot(coid)

    def cleanup_done_orders(self) -> None:
        """Remove terminal orders from tracking."""
        done = [
            coid for coid, o in self._orders.items()
            if o.status in ("filled", "cancelled")
        ]
        for coid in done:
            self._clear_slot(coid)
            self._orders.pop(coid, None)
            self._pending_cancels.discard(coid)

    # ── Private ──────────────────────────────────────────────

    def _submit(self, side: str, price: float, qty: float) -> str | None:
        coid = f"mm_{uuid.uuid4().hex[:12]}"
        order = LiveOrder(
            client_order_id=coid,
            side=side,
            price=price,
            qty=qty,
            submitted_at=time.monotonic(),
        )
        self._orders[coid] = order

        if side == "BUY":
            self._bid_coid = coid
        else:
            self._ask_coid = coid

        if self._cfg.dry_run:
            log.info("DRY %s %.2f @ %.4f  [%s]", side, qty, price, coid)
            order.status = "new"
            return coid

        if self._gw is None:
            return coid

        try:
            self._gw.submit_order(
                symbol=self._cfg.symbol,
                side=side,
                order_type="LIMIT",
                quantity=str(qty),
                price=str(price),
                time_in_force="GTC",
                client_order_id=coid,
            )
            log.debug("SUBMIT %s %.4f x %.4f [%s]", side, price, qty, coid)
        except Exception:
            log.exception("Failed to submit %s order", side)
            order.status = "cancelled"
            self._clear_slot(coid)
        return coid

    def _cancel(self, coid: str) -> None:
        if coid in self._pending_cancels:
            return
        self._pending_cancels.add(coid)
        order = self._orders.get(coid)
        if order:
            order.status = "cancelled"

        self._clear_slot(coid)

        if self._cfg.dry_run or self._gw is None:
            log.debug("DRY CANCEL %s", coid)
            return

        try:
            self._gw.cancel_order(
                symbol=self._cfg.symbol,
                orig_client_order_id=coid,
            )
            log.debug("CANCEL %s", coid)
        except Exception:
            log.exception("Failed to cancel %s", coid)

    def _clear_slot(self, coid: str) -> None:
        if self._bid_coid == coid:
            self._bid_coid = None
        if self._ask_coid == coid:
            self._ask_coid = None
