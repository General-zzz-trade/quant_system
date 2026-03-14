"""OrderManager — track order lifecycle: submit → ack → fill/cancel/timeout."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class _OrderEntry:
    order_id: str
    symbol: str
    submit_time: float
    terminal: bool = False


class OrderManager:
    """Lightweight order tracking with timeout detection."""

    def __init__(self, timeout_sec: float = 30.0) -> None:
        self._timeout_sec = timeout_sec
        self._orders: dict[str, _OrderEntry] = {}

    def submit(self, order_id: str, symbol: str) -> None:
        """Register a new order. Duplicate IDs are silently ignored."""
        if order_id in self._orders:
            logger.debug("Duplicate submit ignored: %s", order_id)
            return
        self._orders[order_id] = _OrderEntry(
            order_id=order_id, symbol=symbol, submit_time=time.time(),
        )

    def on_ack(self, order_id: str, venue_id: str = "") -> None:
        """Mark order as acknowledged by venue."""
        entry = self._orders.get(order_id)
        if entry:
            logger.debug("Order ack: %s -> %s", order_id, venue_id)

    def on_fill(self, order_id: str) -> None:
        """Mark order as filled (terminal)."""
        entry = self._orders.get(order_id)
        if entry:
            entry.terminal = True

    def on_cancel(self, order_id: str) -> None:
        """Mark order as cancelled (terminal)."""
        entry = self._orders.get(order_id)
        if entry:
            entry.terminal = True

    def check_timeouts(self) -> list[str]:
        """Return IDs of non-terminal orders older than timeout_sec."""
        now = time.time()
        timed_out = []
        for oid, entry in self._orders.items():
            if not entry.terminal and (now - entry.submit_time) > self._timeout_sec:
                timed_out.append(oid)
        return timed_out

    @property
    def open_count(self) -> int:
        """Number of non-terminal orders."""
        return sum(1 for e in self._orders.values() if not e.terminal)
