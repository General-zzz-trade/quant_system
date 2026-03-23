"""Tests for production safety components: validation, order TTL, pipeline metrics."""
from __future__ import annotations

import math
import threading
import time
from unittest.mock import MagicMock, patch

import pytest


# ======================================================================
# 1. core/validation.py
# ======================================================================

class TestValidatePrice:

    def test_valid_price(self):
        from execution.validation import validate_price
        assert validate_price(100.5, "test") == 100.5

    def test_zero_price_ok(self):
        from execution.validation import validate_price
        assert validate_price(0.0, "test") == 0.0

    def test_int_price_ok(self):
        from execution.validation import validate_price
        assert validate_price(42, "test") == 42.0

    def test_nan_raises(self):
        from execution.validation import validate_price
        with pytest.raises(ValueError, match="NaN"):
            validate_price(float("nan"), "ctx")

    def test_inf_raises(self):
        from execution.validation import validate_price
        with pytest.raises(ValueError, match="Inf"):
            validate_price(float("inf"), "ctx")

    def test_negative_raises(self):
        from execution.validation import validate_price
        with pytest.raises(ValueError, match="negative"):
            validate_price(-1.0, "ctx")

    def test_non_numeric_raises(self):
        from execution.validation import validate_price
        with pytest.raises(ValueError, match="numeric"):
            validate_price("abc", "ctx")  # type: ignore[arg-type]


class TestValidateQty:

    def test_valid_qty(self):
        from execution.validation import validate_qty
        assert validate_qty(1.5, "test") == 1.5

    def test_nan_raises(self):
        from execution.validation import validate_qty
        with pytest.raises(ValueError, match="NaN"):
            validate_qty(float("nan"), "ctx")

    def test_inf_raises(self):
        from execution.validation import validate_qty
        with pytest.raises(ValueError, match="Inf"):
            validate_qty(float("inf"), "ctx")

    def test_negative_raises(self):
        from execution.validation import validate_qty
        with pytest.raises(ValueError, match="negative"):
            validate_qty(-0.01, "ctx")

    def test_zero_qty_ok(self):
        from execution.validation import validate_qty
        assert validate_qty(0.0, "test") == 0.0


class TestValidateSignal:

    def test_valid_signals(self):
        from execution.validation import validate_signal
        assert validate_signal(-1) == -1
        assert validate_signal(0) == 0
        assert validate_signal(1) == 1

    def test_invalid_signal(self):
        from execution.validation import validate_signal
        with pytest.raises(ValueError, match="must be -1, 0, or \\+1"):
            validate_signal(2)

    def test_float_signal_rejected(self):
        from execution.validation import validate_signal
        with pytest.raises(ValueError):
            validate_signal(0.5)  # type: ignore[arg-type]


class TestSanitizeFeatures:

    def test_clean_dict_unchanged(self):
        from execution.validation import sanitize_features
        d = {"a": 1.0, "b": 2.0, "label": "x"}
        result = sanitize_features(d)
        assert result == d

    def test_nan_replaced(self):
        from execution.validation import sanitize_features
        d = {"a": float("nan"), "b": 5.0}
        result = sanitize_features(d)
        assert result["a"] == 0.0
        assert result["b"] == 5.0

    def test_inf_replaced(self):
        from execution.validation import sanitize_features
        d = {"x": float("inf"), "y": float("-inf")}
        result = sanitize_features(d)
        assert result["x"] == 0.0
        assert result["y"] == 0.0

    def test_original_not_mutated(self):
        from execution.validation import sanitize_features
        d = {"a": float("nan")}
        sanitize_features(d)
        assert math.isnan(d["a"])

    def test_non_float_values_preserved(self):
        from execution.validation import sanitize_features
        d = {"s": "hello", "i": 42, "f": 3.14}
        result = sanitize_features(d)
        assert result == d


# ======================================================================
# 2. Order TTL in BybitAdapter
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


