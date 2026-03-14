"""Tests for WS-API order execution adapter."""
from __future__ import annotations

import threading
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


class _FakeWsGateway:
    """Simulates a WS gateway that can be configured to timeout or error."""

    def __init__(self, *, should_timeout=False, should_error=False):
        self.is_running = True
        self._should_timeout = should_timeout
        self._should_error = should_error
        self._req_counter = 0
        self._adapter = None  # set after adapter is created

    def start(self):
        pass

    def stop(self):
        self.is_running = False

    def submit_order(self, **kwargs) -> str:
        self._req_counter += 1
        req_id = f"ws-req-{self._req_counter}"
        if self._should_error and self._adapter:
            # Schedule error callback in a separate thread
            def _fire():
                self._adapter._on_error(req_id, {"error": "ws_protocol_error"})
            t = threading.Thread(target=_fire)
            t.start()
        # If should_timeout, we simply never respond
        return req_id


class TestWsTimeoutFallback:

    def test_ws_timeout_falls_back_to_rest(self):
        """When WS doesn't respond within timeout, REST adapter is used."""
        rest = _FakeRestAdapter()
        adapter = WsOrderAdapter(
            rest_adapter=rest,
            api_key="test",
            api_secret="test",
            response_timeout_sec=0.05,  # 50ms timeout
        )

        # Install a fake WS gateway that never responds (simulates timeout)
        fake_gw = _FakeWsGateway(should_timeout=True)
        adapter._gateway = fake_gw
        adapter._started = True

        order = _FakeOrderEvent()
        results = adapter.send_order(order)

        # Verify REST fallback was called
        assert len(rest.orders) == 1
        assert len(results) == 1
        assert results[0]["status"] == "filled"

    def test_ws_error_falls_back_to_rest(self):
        """When WS returns error, REST adapter is used."""
        rest = _FakeRestAdapter()
        adapter = WsOrderAdapter(
            rest_adapter=rest,
            api_key="test",
            api_secret="test",
            response_timeout_sec=1.0,
        )

        # Install a fake WS gateway that fires error callback
        fake_gw = _FakeWsGateway(should_error=True)
        fake_gw._adapter = adapter
        adapter._gateway = fake_gw
        adapter._started = True

        order = _FakeOrderEvent()
        results = adapter.send_order(order)

        # Verify REST fallback was called after WS error
        assert len(rest.orders) == 1
        assert len(results) == 1
        assert results[0]["status"] == "filled"
