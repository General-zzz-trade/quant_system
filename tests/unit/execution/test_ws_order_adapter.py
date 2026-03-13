"""Tests for WS-API order execution adapter."""
from __future__ import annotations

import threading
import pytest
from dataclasses import dataclass
from typing import Any, Optional

from execution.adapters.binance.ws_order_adapter import WsOrderAdapter, _PendingOrder


@dataclass
class _FakeOrderEvent:
    symbol: str = "BTCUSDT"
    side: str = "BUY"
    order_type: str = "MARKET"
    qty: float = 0.001
    price: Optional[float] = None
    time_in_force: Optional[str] = None
    reduce_only: Optional[bool] = None
    client_order_id: Optional[str] = None


class _FakeRestAdapter:
    """Simulates REST venue client."""

    def __init__(self):
        self.orders = []

    def send_order(self, order_event: Any) -> list:
        self.orders.append(order_event)
        return [{"status": "filled", "symbol": getattr(order_event, "symbol", "")}]


class TestWsOrderAdapter:

    def test_fallback_to_rest_when_not_started(self):
        """Without WS connection, should fall back to REST."""
        rest = _FakeRestAdapter()
        adapter = WsOrderAdapter(
            rest_adapter=rest,
            api_key="test",
            api_secret="test",
        )
        order = _FakeOrderEvent()
        results = adapter.send_order(order)
        assert len(results) == 1
        assert len(rest.orders) == 1

    def test_is_ws_connected_false_initially(self):
        rest = _FakeRestAdapter()
        adapter = WsOrderAdapter(
            rest_adapter=rest,
            api_key="test",
            api_secret="test",
        )
        assert adapter.is_ws_connected is False

    def test_pending_order_event_synchronization(self):
        """Test the _PendingOrder threading.Event mechanism."""
        event = threading.Event()
        pending = _PendingOrder(event=event)
        assert pending.response is None
        assert pending.error is None
        assert pending.latency_ms == 0.0

        # Simulate response
        pending.response = {"status": 200}
        pending.latency_ms = 3.5
        event.set()

        assert event.is_set()
        assert pending.response["status"] == 200

    def test_stop_is_safe_when_not_started(self):
        rest = _FakeRestAdapter()
        adapter = WsOrderAdapter(
            rest_adapter=rest,
            api_key="test",
            api_secret="test",
        )
        adapter.stop()  # Should not raise


class TestWsOrderAdapterCallbacks:

    def test_on_response_sets_pending(self):
        rest = _FakeRestAdapter()
        adapter = WsOrderAdapter(
            rest_adapter=rest,
            api_key="test",
            api_secret="test",
        )

        event = threading.Event()
        pending = _PendingOrder(event=event)
        adapter._pending["req-123"] = pending

        adapter._on_response({"id": "req-123", "_latency_ms": 4.2})
        assert event.is_set()
        assert pending.response is not None
        assert pending.latency_ms == 4.2

    def test_on_error_sets_pending(self):
        rest = _FakeRestAdapter()
        adapter = WsOrderAdapter(
            rest_adapter=rest,
            api_key="test",
            api_secret="test",
        )

        event = threading.Event()
        pending = _PendingOrder(event=event)
        adapter._pending["req-456"] = pending

        adapter._on_error("req-456", {"error": "insufficient margin"})
        assert event.is_set()
        assert pending.error is not None
