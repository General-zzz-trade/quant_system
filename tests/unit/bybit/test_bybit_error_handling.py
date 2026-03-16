"""Tests for Bybit error handling — retCode errors, bad responses, missing fields."""
from __future__ import annotations

from unittest.mock import MagicMock

from execution.adapters.bybit.adapter import BybitAdapter
from execution.adapters.bybit.config import BybitConfig


def _make_adapter() -> BybitAdapter:
    cfg = BybitConfig.demo(api_key="k", api_secret="s")
    adapter = BybitAdapter(cfg)
    adapter._connected = True
    adapter._client = MagicMock()
    return adapter


class TestRetCodeErrors:
    def test_nonzero_retcode_on_market_order(self):
        adapter = _make_adapter()
        adapter._client.post.return_value = {
            "retCode": 10004, "retMsg": "sign error",
        }
        result = adapter.send_market_order("BTCUSDT", "buy", 0.001)
        assert result["status"] == "error"
        assert result["retMsg"] == "sign error"

    def test_nonzero_retcode_on_limit_order(self):
        adapter = _make_adapter()
        adapter._client.post.return_value = {
            "retCode": 10005, "retMsg": "permission denied",
        }
        result = adapter.send_limit_order("BTCUSDT", "buy", 0.01, 70000.0)
        assert result["status"] == "error"
        assert result["retCode"] == 10005

    def test_nonzero_retcode_on_cancel(self):
        adapter = _make_adapter()
        adapter._client.post.return_value = {
            "retCode": 110001, "retMsg": "Order does not exist",
        }
        result = adapter.cancel_order("BTCUSDT", "nonexistent")
        assert result["status"] == "error"

    def test_network_error_retcode(self):
        adapter = _make_adapter()
        adapter._client.post.return_value = {
            "retCode": -1, "retMsg": "Connection reset", "retryable": True,
        }
        result = adapter.send_market_order("BTCUSDT", "buy", 0.01)
        assert result["status"] == "error"


class TestEmptyResults:
    def test_list_instruments_empty_on_error(self):
        adapter = _make_adapter()
        adapter._client.get.return_value = {
            "retCode": 10001, "retMsg": "bad request",
        }
        instruments = adapter.list_instruments(symbols=["BTCUSDT"])
        assert instruments == ()

    def test_get_positions_empty_on_error(self):
        adapter = _make_adapter()
        adapter._client.get.return_value = {
            "retCode": 10001, "retMsg": "error",
        }
        positions = adapter.get_positions(symbol="BTCUSDT")
        assert positions == ()

    def test_get_open_orders_empty_on_error(self):
        adapter = _make_adapter()
        adapter._client.get.return_value = {
            "retCode": 10001, "retMsg": "error",
        }
        orders = adapter.get_open_orders(symbol="BTCUSDT")
        assert orders == ()

    def test_get_recent_fills_empty_on_error(self):
        adapter = _make_adapter()
        adapter._client.get.return_value = {
            "retCode": 10001, "retMsg": "error",
        }
        fills = adapter.get_recent_fills(symbol="BTCUSDT")
        assert fills == ()

    def test_get_ticker_empty_on_error(self):
        adapter = _make_adapter()
        adapter._client.get.return_value = {
            "retCode": 10001, "retMsg": "error",
        }
        ticker = adapter.get_ticker("BTCUSDT")
        assert ticker == {}

    def test_get_klines_empty_on_error(self):
        adapter = _make_adapter()
        adapter._client.get.return_value = {
            "retCode": 10001, "retMsg": "error",
        }
        klines = adapter.get_klines("BTCUSDT")
        assert klines == []


class TestConnectionHandling:
    def test_connect_success(self):
        adapter = _make_adapter()
        adapter._connected = False
        adapter._client.get.return_value = {
            "retCode": 0,
            "result": {"list": [{"coin": [
                {"coin": "USDT", "walletBalance": "10000", "locked": "0"},
            ]}]},
        }
        assert adapter.connect() is True
        assert adapter.is_connected() is True

    def test_connect_failure(self):
        adapter = _make_adapter()
        adapter._connected = False
        adapter._client.get.return_value = {
            "retCode": 10003, "retMsg": "invalid api key",
        }
        assert adapter.connect() is False
        assert adapter.is_connected() is False
