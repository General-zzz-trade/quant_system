"""BinanceExecutor — order submission + user stream + fill recording.

Wraps venue client with kill switch check. Shadow mode for paper trading.
"""
from __future__ import annotations

import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)


class BinanceExecutor:
    """Send orders to Binance via WS-API or REST, receive fills."""

    def __init__(
        self,
        venue_client: Any,
        kill_switch: Any,
        use_ws: bool = True,
        shadow_mode: bool = False,
        on_fill: Callable[[dict], None] | None = None,
    ) -> None:
        self._client = venue_client
        self._kill_switch = kill_switch
        self._use_ws = use_ws
        self._shadow_mode = shadow_mode
        self._on_fill = on_fill
        self._user_stream_running = False

    def send(self, order: Any) -> dict:
        """Send order to venue. Returns result dict."""
        if self._kill_switch.is_killed():
            logger.warning("Order blocked: kill switch active")
            return {"status": "blocked_kill_switch"}

        if self._shadow_mode:
            logger.info("Shadow order: %s", order)
            return {"status": "shadow"}

        return self._client.send_order(order)

    def cancel(self, order_id: str) -> dict:
        """Cancel an order."""
        if self._shadow_mode:
            return {"status": "shadow_cancel"}
        return self._client.cancel_order(order_id)

    def start_user_stream(self, on_fill: Callable[[dict], None] | None = None) -> None:
        """Start user data stream for fill notifications."""
        if on_fill:
            self._on_fill = on_fill
        if hasattr(self._client, "start_user_stream"):
            self._client.start_user_stream(callback=self._handle_fill)
        self._user_stream_running = True
        logger.info("User stream started")

    def stop_user_stream(self) -> None:
        """Stop user data stream."""
        if hasattr(self._client, "stop_user_stream"):
            self._client.stop_user_stream()
        self._user_stream_running = False
        logger.info("User stream stopped")

    def get_positions(self) -> list:
        """Get current positions from venue."""
        if hasattr(self._client, "get_positions"):
            return self._client.get_positions()
        return []

    def get_balances(self) -> dict:
        """Get current balances from venue."""
        if hasattr(self._client, "get_balances"):
            return self._client.get_balances()
        return {}

    def _handle_fill(self, fill: dict) -> None:
        """Process incoming fill from user stream."""
        logger.debug("Fill received: %s", fill)
        if self._on_fill:
            self._on_fill(fill)
