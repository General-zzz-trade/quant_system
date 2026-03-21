"""Safety tests for order_utils — edge cases that could cause real money loss."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from scripts.ops.order_utils import clamp_notional, reliable_close_position


class TestClampNotional:
    # Signature: clamp_notional(qty, price, symbol="", max_notional=500)

    def test_zero_price(self):
        """Zero price must not cause ZeroDivisionError."""
        result = clamp_notional(0.1, 0.0, "BTCUSDT")
        assert result == 0.1  # unchanged

    def test_negative_price(self):
        result = clamp_notional(0.1, -100.0, "BTCUSDT")
        assert result == 0.1

    def test_zero_qty(self):
        result = clamp_notional(0.0, 50000.0, "BTCUSDT")
        assert result == 0.0

    def test_normal_clamp(self):
        # qty=1.0, price=50000 → notional=50000 > MAX=500 → clamp
        result = clamp_notional(1.0, 50000.0, "BTCUSDT")
        assert result < 1.0
        assert result * 50000.0 <= 500.1

    def test_nan_price(self):
        """NaN price must not produce NaN qty."""
        result = clamp_notional(0.1, float("nan"), "BTCUSDT")
        assert result == result  # not NaN

    def test_inf_price(self):
        result = clamp_notional(0.1, float("inf"), "BTCUSDT")
        assert result == result  # not NaN

    def test_under_limit_unchanged(self):
        """Qty under limit should pass through unchanged."""
        result = clamp_notional(0.001, 50000.0, "BTCUSDT")  # notional=50 < 500
        assert result == 0.001


class TestReliableClosePosition:
    # Signature: reliable_close_position(adapter, symbol, max_retries=3, verify=True)

    def test_close_with_mock_adapter(self):
        adapter = MagicMock()
        adapter.close_position.return_value = {"retCode": 0, "result": {}}
        adapter.get_positions.return_value = []
        result = reliable_close_position(adapter, "BTCUSDT", max_retries=1, verify=False)
        assert isinstance(result, dict)

    def test_close_returns_status(self):
        adapter = MagicMock()
        adapter.close_position.return_value = {"retCode": 0}
        adapter.get_positions.return_value = []
        result = reliable_close_position(adapter, "ETHUSDT", verify=False)
        assert "status" in result
