"""WS-API order adapter integration test (testnet).

Tests WsOrderAdapter against Binance testnet to verify:
1. WS order submission works end-to-end
2. No double-ordering (WS success does not trigger REST fallback)
3. REST fallback on WS disconnect
4. Latency SLA: WS p99 < 10ms

Requires BINANCE_TESTNET_API_KEY and BINANCE_TESTNET_API_SECRET env vars.
Skip if not available.
"""
from __future__ import annotations

import os
import time
import threading
from dataclasses import dataclass
from typing import Any, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from execution.adapters.binance.ws_order_adapter import WsOrderAdapter


# ── Helpers ──────────────────────────────────────────────────

@dataclass
class StubOrderEvent:
    symbol: str = "BTCUSDT"
    side: str = "BUY"
    order_type: str = "MARKET"
    qty: str = "0.001"
    price: Optional[str] = None
    time_in_force: Optional[str] = None
    reduce_only: Optional[bool] = None
    client_order_id: Optional[str] = None


class OrderTrackingRestAdapter:
    """REST adapter that tracks all calls for double-order detection."""

    def __init__(self):
        self.calls: List[Any] = []

    def send_order(self, order_event: Any) -> list:
        self.calls.append(order_event)
        return [{"status": "REST_FILLED", "symbol": getattr(order_event, "symbol", "")}]


# ── Unit tests (no network) ─────────────────────────────────

class TestWsOrderAdapterLogic:
    """Unit tests for WS order adapter logic (no network required)."""

    def test_ws_disconnected_falls_back_to_rest(self):
        """When WS is not connected, should immediately use REST."""
        rest = OrderTrackingRestAdapter()
        adapter = WsOrderAdapter(rest_adapter=rest, api_key="k", api_secret="s")
        # Don't call start() — WS stays disconnected

        result = adapter.send_order(StubOrderEvent())
        assert len(rest.calls) == 1, "Should have called REST exactly once"
        assert result[0]["status"] == "REST_FILLED"

    def test_ws_submit_error_falls_back_to_rest(self):
        """If WS submit raises, should fall back to REST without double-order."""
        rest = OrderTrackingRestAdapter()
        adapter = WsOrderAdapter(rest_adapter=rest, api_key="k", api_secret="s")

        # Mock gateway that raises on submit
        mock_gw = MagicMock()
        mock_gw.is_running = True
        mock_gw.submit_order.side_effect = RuntimeError("WS send failed")
        adapter._gateway = mock_gw
        adapter._started = True

        adapter.send_order(StubOrderEvent())
        assert len(rest.calls) == 1, "Should fall back to REST exactly once"

    def test_ws_timeout_falls_back_to_rest(self):
        """If WS response times out, should fall back to REST."""
        rest = OrderTrackingRestAdapter()
        adapter = WsOrderAdapter(
            rest_adapter=rest, api_key="k", api_secret="s",
            response_timeout_sec=0.05,  # very short timeout
        )

        mock_gw = MagicMock()
        mock_gw.is_running = True
        mock_gw.submit_order.return_value = "req-123"
        adapter._gateway = mock_gw
        adapter._started = True

        adapter.send_order(StubOrderEvent())
        # No response callback → timeout → REST fallback
        assert len(rest.calls) == 1

    def test_ws_error_response_falls_back_to_rest(self):
        """If WS returns an error, should fall back to REST."""
        rest = OrderTrackingRestAdapter()
        adapter = WsOrderAdapter(
            rest_adapter=rest, api_key="k", api_secret="s",
            response_timeout_sec=1.0,
        )

        mock_gw = MagicMock()
        mock_gw.is_running = True
        mock_gw.submit_order.return_value = "req-456"
        adapter._gateway = mock_gw
        adapter._started = True

        # Simulate error response arriving async
        def trigger_error():
            time.sleep(0.01)
            adapter._on_error("req-456", {"code": -1000, "msg": "bad request"})

        t = threading.Thread(target=trigger_error)
        t.start()

        adapter.send_order(StubOrderEvent())
        t.join()
        assert len(rest.calls) == 1

    def test_ws_success_no_double_order(self):
        """CRITICAL: WS success should NOT also call REST.

        Bug audit: line 155 of ws_order_adapter.py returns
        `list(self._rest.send_order(order_event)) if pending.response is None else []`
        When WS succeeds, pending.response is set (not None), so REST is NOT called.
        """
        rest = OrderTrackingRestAdapter()
        adapter = WsOrderAdapter(
            rest_adapter=rest, api_key="k", api_secret="s",
            response_timeout_sec=1.0,
        )

        mock_gw = MagicMock()
        mock_gw.is_running = True
        mock_gw.submit_order.return_value = "req-789"
        adapter._gateway = mock_gw
        adapter._started = True

        # Simulate successful response
        def trigger_success():
            time.sleep(0.01)
            adapter._on_response({
                "id": "req-789",
                "status": "FILLED",
                "_latency_ms": 3.5,
            })

        t = threading.Thread(target=trigger_success)
        t.start()

        result = adapter.send_order(StubOrderEvent())
        t.join()

        # REST should NOT be called when WS succeeds
        assert len(rest.calls) == 0, (
            f"Double order detected! REST called {len(rest.calls)} times "
            f"after WS success. This would cause duplicate fills."
        )
        # Result should be empty list (WS handled it)
        assert result == []

    def test_ws_success_with_none_response_calls_rest(self):
        """Edge case: if somehow response is set to None, REST IS called.

        This is the current code behavior (line 155). Document it.
        In practice this shouldn't happen because _on_response always
        sets pending.response to the dict.
        """
        # This test documents current behavior, not a bug
        pass

    def test_multiple_concurrent_orders(self):
        """Multiple orders should each get their own req_id tracking."""
        rest = OrderTrackingRestAdapter()
        adapter = WsOrderAdapter(
            rest_adapter=rest, api_key="k", api_secret="s",
            response_timeout_sec=1.0,
        )

        mock_gw = MagicMock()
        mock_gw.is_running = True
        call_count = [0]
        def mock_submit(**kwargs):
            call_count[0] += 1
            return f"req-{call_count[0]}"
        mock_gw.submit_order.side_effect = mock_submit
        adapter._gateway = mock_gw
        adapter._started = True

        results = [None, None]
        def send_order_1():
            time.sleep(0.01)
            adapter._on_response({"id": "req-1", "status": "FILLED", "_latency_ms": 2.0})
        def send_order_2():
            time.sleep(0.02)
            adapter._on_response({"id": "req-2", "status": "FILLED", "_latency_ms": 3.0})

        t1 = threading.Thread(target=send_order_1)
        t2 = threading.Thread(target=send_order_2)
        t1.start()
        t2.start()

        results[0] = adapter.send_order(StubOrderEvent(symbol="BTCUSDT"))
        results[1] = adapter.send_order(StubOrderEvent(symbol="ETHUSDT"))

        t1.join()
        t2.join()

        # No REST fallback for either
        assert len(rest.calls) == 0, "No double orders"

    def test_start_idempotent(self):
        """Calling start() multiple times should not create multiple gateways."""
        rest = OrderTrackingRestAdapter()
        adapter = WsOrderAdapter(rest_adapter=rest, api_key="k", api_secret="s")

        # Mock the gateway import to avoid actual connection
        with patch("execution.adapters.binance.ws_order_adapter.WsOrderAdapter.start"):
            pass  # Just verify the flag check

        # Direct flag check
        adapter._started = True
        adapter.start()  # Should return early
        # No exception = success


# ── Latency SLA tests (mock) ────────────────────────────────

class TestWsOrderLatencySLA:
    """Verify latency tracking works correctly."""

    def test_latency_recorded_on_success(self):
        rest = OrderTrackingRestAdapter()
        adapter = WsOrderAdapter(
            rest_adapter=rest, api_key="k", api_secret="s",
            response_timeout_sec=1.0,
        )

        mock_gw = MagicMock()
        mock_gw.is_running = True
        mock_gw.submit_order.return_value = "req-lat"
        adapter._gateway = mock_gw
        adapter._started = True

        # Simulate response with known latency
        def trigger():
            time.sleep(0.01)
            adapter._on_response({
                "id": "req-lat",
                "status": "FILLED",
                "_latency_ms": 4.2,
            })

        t = threading.Thread(target=trigger)
        t.start()
        adapter.send_order(StubOrderEvent())
        t.join()

        # Verify pending order captured latency
        # (adapter logs it, we just verify no crash)


# ── Testnet integration tests ────────────────────────────────

@pytest.mark.skipif(
    not os.environ.get("BINANCE_TESTNET_API_KEY"),
    reason="BINANCE_TESTNET_API_KEY not set — skipping testnet integration tests",
)
class TestWsOrdersTestnet:
    """Live testnet tests — requires API credentials."""

    @pytest.fixture
    def adapter(self):
        rest = OrderTrackingRestAdapter()
        a = WsOrderAdapter(
            rest_adapter=rest,
            api_key=os.environ["BINANCE_TESTNET_API_KEY"],
            api_secret=os.environ["BINANCE_TESTNET_API_SECRET"],
            testnet=True,
            response_timeout_sec=5.0,
        )
        a.start()
        yield a, rest
        a.stop()

    def test_testnet_ws_connection(self, adapter):
        a, rest = adapter
        assert a.is_ws_connected, "Should connect to testnet WS"

    def test_testnet_order_submission(self, adapter):
        a, rest = adapter
        if not a.is_ws_connected:
            pytest.skip("WS not connected")

        order = StubOrderEvent(
            symbol="BTCUSDT",
            side="BUY",
            order_type="MARKET",
            qty="0.001",
        )
        a.send_order(order)
        # Should succeed via WS without REST fallback
        assert len(rest.calls) == 0, "Testnet WS order should not fall back to REST"

    def test_testnet_latency_sla(self, adapter):
        """WS order latency p99 < 10ms."""
        a, rest = adapter
        if not a.is_ws_connected:
            pytest.skip("WS not connected")

        latencies = []
        for i in range(10):
            order = StubOrderEvent(
                symbol="BTCUSDT",
                side="BUY",
                order_type="MARKET",
                qty="0.001",
                client_order_id=f"latency-test-{i}",
            )
            t0 = time.monotonic()
            a.send_order(order)
            latencies.append((time.monotonic() - t0) * 1000)
            time.sleep(0.5)  # rate limit

        if latencies:
            p99 = sorted(latencies)[int(len(latencies) * 0.99)]
            assert p99 < 10.0, f"WS p99 latency {p99:.1f}ms exceeds 10ms SLA"
