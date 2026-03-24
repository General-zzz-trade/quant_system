"""Tests for production safety components: order TTL, pipeline metrics."""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest


# ======================================================================
# 1. Order TTL in BybitAdapter
# ======================================================================

class TestOrderTTL:

    def _make_adapter(self):
        """Create a BybitAdapter with mocked client."""
        from execution.adapters.bybit.config import BybitConfig
        from execution.adapters.bybit.adapter import BybitAdapter

        config = BybitConfig(
            api_key="test_key",
            api_secret="test_secret",
            base_url="https://api-demo.bybit.com",
        )
        adapter = BybitAdapter(config)
        adapter._client = MagicMock()
        return adapter

    def test_pending_orders_tracked_on_submit(self):
        adapter = self._make_adapter()
        adapter._client.post.return_value = {
            "retCode": 0,
            "result": {"orderId": "o1", "orderLinkId": "link1"},
        }
        result = adapter.send_market_order("BTCUSDT", "buy", 0.001)
        assert result["status"] == "submitted"
        assert "link1" in adapter._pending_orders

    def test_stale_order_cancelled(self):
        from execution.adapters.bybit import adapter as adapter_mod

        adapter = self._make_adapter()
        # Simulate an order submitted 60 seconds ago (well past TTL)
        adapter._pending_orders["stale_link"] = time.time() - 60

        adapter._client.post.return_value = {"retCode": 0, "result": {}}
        cancelled = adapter._check_order_ttls()

        assert "stale_link" in cancelled
        assert "stale_link" not in adapter._pending_orders
        # Verify cancel API was called
        adapter._client.post.assert_called_once()
        call_args = adapter._client.post.call_args
        assert call_args[0][0] == "/v5/order/cancel"

    def test_fresh_order_not_cancelled(self):
        adapter = self._make_adapter()
        adapter._pending_orders["fresh_link"] = time.time()

        cancelled = adapter._check_order_ttls()
        assert cancelled == []
        assert "fresh_link" in adapter._pending_orders

    def test_ttl_check_runs_before_new_order(self):
        adapter = self._make_adapter()
        # Plant a stale order
        adapter._pending_orders["old_link"] = time.time() - 60

        # Mock: first call cancels stale, second call submits new order
        adapter._client.post.side_effect = [
            {"retCode": 0, "result": {}},  # cancel stale
            {"retCode": 0, "result": {"orderId": "o2", "orderLinkId": "new_link"}},
        ]

        result = adapter.send_market_order("ETHUSDT", "sell", 1.0)
        assert result["status"] == "submitted"
        assert "old_link" not in adapter._pending_orders
        assert adapter._client.post.call_count == 2


