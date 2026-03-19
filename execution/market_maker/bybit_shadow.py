"""Bybit shadow quoting — mirror market maker quotes on Bybit demo.

Runs in a background thread, periodically syncing quotes from the
main engine to Bybit demo account via REST for comparison.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass
class ShadowState:
    """Last mirrored quotes on Bybit."""
    bid_price: float = 0.0
    ask_price: float = 0.0
    bid_coid: str = ""
    ask_coid: str = ""
    last_sync_ts: float = 0.0
    sync_count: int = 0
    error_count: int = 0


class BybitShadow:
    """Background thread that mirrors quotes on Bybit demo.

    Uses Bybit REST API (no WS needed — latency doesn't matter
    for shadow validation).
    """

    def __init__(
        self,
        symbol: str = "ETHUSDT",
        sync_interval_s: float = 5.0,
        bybit_client=None,
    ) -> None:
        self._symbol = symbol
        self._interval = sync_interval_s
        self._client = bybit_client  # BybitClient or None
        self._state = ShadowState()
        self._target_bid: float = 0.0
        self._target_ask: float = 0.0
        self._target_size: float = 0.0
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def update_target(self, bid: float, ask: float, size: float) -> None:
        """Called from main engine thread to set new target quotes."""
        with self._lock:
            self._target_bid = bid
            self._target_ask = ask
            self._target_size = size

    def start(self) -> None:
        if self._client is None:
            log.warning("Bybit shadow: no client, skipping")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._sync_loop, daemon=True, name="bybit-shadow"
        )
        self._thread.start()
        log.info("Bybit shadow started (interval=%.1fs)", self._interval)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        log.info(
            "Bybit shadow stopped: syncs=%d errors=%d",
            self._state.sync_count,
            self._state.error_count,
        )

    def _sync_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._sync_once()
            except Exception:
                log.exception("Shadow sync error")
                self._state.error_count += 1
            self._stop_event.wait(self._interval)

    def _sync_once(self) -> None:
        with self._lock:
            bid = self._target_bid
            ask = self._target_ask
            size = self._target_size

        if bid <= 0 or ask <= 0 or size <= 0:
            return

        # Cancel existing, place new
        if self._state.bid_coid:
            self._cancel_order(self._state.bid_coid)
        if self._state.ask_coid:
            self._cancel_order(self._state.ask_coid)

        bid_coid = self._place_order("Buy", bid, size)
        ask_coid = self._place_order("Sell", ask, size)

        self._state.bid_price = bid
        self._state.ask_price = ask
        self._state.bid_coid = bid_coid or ""
        self._state.ask_coid = ask_coid or ""
        self._state.last_sync_ts = time.time()
        self._state.sync_count += 1

    def _place_order(self, side: str, price: float, qty: float) -> str | None:
        if self._client is None:
            return None
        try:
            resp = self._client.place_order(
                category="linear",
                symbol=self._symbol,
                side=side,
                orderType="Limit",
                qty=str(qty),
                price=str(price),
                timeInForce="GTC",
            )
            return resp.get("result", {}).get("orderId", "")
        except Exception:
            log.exception("Shadow place_order failed")
            self._state.error_count += 1
            return None

    def _cancel_order(self, order_id: str) -> None:
        if self._client is None or not order_id:
            return
        try:
            self._client.cancel_order(
                category="linear",
                symbol=self._symbol,
                orderId=order_id,
            )
        except Exception:
            pass  # Best-effort cancel
