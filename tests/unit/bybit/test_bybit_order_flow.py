"""Tests for Bybit adapter order flow — market orders, cancel, close_position."""
from __future__ import annotations

from unittest.mock import MagicMock

from execution.adapters.bybit.adapter import BybitAdapter
from execution.adapters.bybit.config import BybitConfig


def _make_adapter() -> BybitAdapter:
    cfg = BybitConfig.demo(api_key="test_key", api_secret="test_secret")
    adapter = BybitAdapter(cfg)
    adapter._connected = True
    adapter._client = MagicMock()
    return adapter


class TestSendMarketOrder:
    def test_buy_order_submitted(self):
        adapter = _make_adapter()
        adapter._client.post.return_value = {
            "retCode": 0,
            "result": {"orderId": "ord_001", "orderLinkId": "qs_BTCUSDT_b_123"},
        }
        result = adapter.send_market_order("BTCUSDT", "buy", 0.01)
        assert result["status"] == "submitted"
        assert result["orderId"] == "ord_001"

    def test_sell_order_maps_side(self):
        adapter = _make_adapter()
        adapter._client.post.return_value = {
            "retCode": 0,
            "result": {"orderId": "ord_002", "orderLinkId": ""},
        }
        adapter.send_market_order("ETHUSDT", "sell", 1.5)
        call_body = adapter._client.post.call_args[0][1]
        assert call_body["side"] == "Sell"
        assert call_body["symbol"] == "ETHUSDT"
        assert call_body["qty"] == "1.5"

    def test_reduce_only_flag(self):
        adapter = _make_adapter()
        adapter._client.post.return_value = {
            "retCode": 0,
            "result": {"orderId": "ord_003", "orderLinkId": ""},
        }
        adapter.send_market_order("BTCUSDT", "sell", 0.01, reduce_only=True)
        call_body = adapter._client.post.call_args[0][1]
        assert call_body["reduceOnly"] is True

    def test_order_link_id_generated(self):
        adapter = _make_adapter()
        adapter._client.post.return_value = {
            "retCode": 0,
            "result": {"orderId": "ord_004", "orderLinkId": "qs_BTCUSDT_b_999"},
        }
        adapter.send_market_order("BTCUSDT", "buy", 0.001)
        call_body = adapter._client.post.call_args[0][1]
        assert call_body["orderLinkId"].startswith("qs_BTCUSDT_b_")

    def test_market_order_error_response(self):
        adapter = _make_adapter()
        adapter._client.post.return_value = {
            "retCode": 10001, "retMsg": "qty too small",
        }
        result = adapter.send_market_order("BTCUSDT", "buy", 0.0001)
        assert result["status"] == "error"
        assert result["retCode"] == 10001


class TestCancelOrder:
    def test_cancel_success(self):
        adapter = _make_adapter()
        adapter._client.post.return_value = {"retCode": 0, "retMsg": "OK"}
        result = adapter.cancel_order("BTCUSDT", "ord_999")
        assert result["status"] == "canceled"
        call_body = adapter._client.post.call_args[0][1]
        assert call_body["orderId"] == "ord_999"
        assert call_body["symbol"] == "BTCUSDT"

    def test_cancel_error(self):
        adapter = _make_adapter()
        adapter._client.post.return_value = {
            "retCode": 110001, "retMsg": "order not found",
        }
        result = adapter.cancel_order("BTCUSDT", "ord_bad")
        assert result["status"] == "error"

    def test_cancel_all_with_symbol(self):
        adapter = _make_adapter()
        adapter._client.post.return_value = {"retCode": 0}
        result = adapter.cancel_all(symbol="ETHUSDT")
        assert result["status"] == "canceled"
        call_body = adapter._client.post.call_args[0][1]
        assert call_body["symbol"] == "ETHUSDT"


class TestClosePosition:
    def test_close_long_position(self):
        adapter = _make_adapter()
        # get_positions returns a long position
        adapter._client.get.return_value = {
            "retCode": 0,
            "result": {"list": [{
                "symbol": "BTCUSDT", "side": "Buy", "size": "0.05",
                "avgPrice": "70000", "markPrice": "70500",
                "unrealisedPnl": "25", "leverage": "10",
                "tradeMode": "0", "updatedTime": "0",
            }]},
        }
        adapter._client.post.return_value = {
            "retCode": 0,
            "result": {"orderId": "close_001", "orderLinkId": ""},
        }
        result = adapter.close_position("BTCUSDT")
        assert result["status"] == "submitted"
        # close long -> sell
        call_body = adapter._client.post.call_args[0][1]
        assert call_body["side"] == "Sell"
        assert call_body["qty"] == "0.05"

    def test_close_short_position(self):
        adapter = _make_adapter()
        adapter._client.get.return_value = {
            "retCode": 0,
            "result": {"list": [{
                "symbol": "ETHUSDT", "side": "Sell", "size": "2.0",
                "avgPrice": "3500", "markPrice": "3450",
                "unrealisedPnl": "100", "leverage": "5",
                "tradeMode": "1", "updatedTime": "0",
            }]},
        }
        adapter._client.post.return_value = {
            "retCode": 0,
            "result": {"orderId": "close_002", "orderLinkId": ""},
        }
        result = adapter.close_position("ETHUSDT")
        assert result["status"] == "submitted"
        call_body = adapter._client.post.call_args[0][1]
        assert call_body["side"] == "Buy"
        assert call_body["qty"] == "2.0"

    def test_close_no_position(self):
        adapter = _make_adapter()
        adapter._client.get.return_value = {
            "retCode": 0, "result": {"list": []},
        }
        result = adapter.close_position("BTCUSDT")
        assert result["status"] == "no_position"

    def test_close_flat_position_treated_as_no_position(self):
        """Positions with size=0 are filtered out by get_positions."""
        adapter = _make_adapter()
        adapter._client.get.return_value = {
            "retCode": 0,
            "result": {"list": [{
                "symbol": "BTCUSDT", "side": "None", "size": "0",
                "avgPrice": "0", "markPrice": "70000",
                "unrealisedPnl": "0", "leverage": "10",
                "tradeMode": "0", "updatedTime": "0",
            }]},
        }
        result = adapter.close_position("BTCUSDT")
        assert result["status"] == "no_position"
