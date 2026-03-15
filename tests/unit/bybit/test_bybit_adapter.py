# tests/unit/bybit/test_bybit_adapter.py
"""Unit tests for Bybit adapter — uses mocks (no API key required)."""
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from execution.adapters.bybit.config import BybitConfig
from execution.adapters.bybit.mapper import (
    map_balance,
    map_fill,
    map_instrument,
    map_order,
    map_position,
)


class TestBybitConfig:
    def test_demo_factory(self):
        cfg = BybitConfig.demo(api_key="k", api_secret="s")
        assert cfg.is_demo
        assert not cfg.is_testnet
        assert "demo" in cfg.base_url

    def test_testnet_factory(self):
        cfg = BybitConfig.testnet(api_key="k", api_secret="s")
        assert cfg.is_testnet
        assert not cfg.is_demo

    def test_live_factory(self):
        cfg = BybitConfig.live(api_key="k", api_secret="s")
        assert not cfg.is_demo
        assert not cfg.is_testnet


class TestMapInstrument:
    def test_btcusdt_perp(self):
        raw = {
            "symbol": "BTCUSDT",
            "baseCoin": "BTC",
            "quoteCoin": "USDT",
            "status": "Trading",
            "lotSizeFilter": {
                "qtyStep": "0.001",
                "minOrderQty": "0.001",
                "maxOrderQty": "100",
                "minNotionalValue": "5",
            },
            "priceFilter": {"tickSize": "0.10"},
        }
        inst = map_instrument(raw)
        assert inst.venue == "bybit"
        assert inst.symbol == "BTCUSDT"
        assert inst.base_asset == "BTC"
        assert inst.quote_asset == "USDT"
        assert inst.tick_size == Decimal("0.10")
        assert inst.lot_size == Decimal("0.001")
        assert inst.min_qty == Decimal("0.001")
        assert inst.trading_enabled is True


class TestMapPosition:
    def test_long_position(self):
        raw = {
            "symbol": "BTCUSDT", "side": "Buy", "size": "0.1",
            "avgPrice": "70000", "markPrice": "70500",
            "unrealisedPnl": "50", "leverage": "10",
            "tradeMode": "0", "updatedTime": "1700000000000",
        }
        pos = map_position(raw)
        assert pos.qty == Decimal("0.1")
        assert pos.is_long
        assert pos.entry_price == Decimal("70000")

    def test_short_position(self):
        raw = {
            "symbol": "ETHUSDT", "side": "Sell", "size": "1.0",
            "avgPrice": "3500", "markPrice": "3450",
            "unrealisedPnl": "50", "leverage": "5",
            "tradeMode": "1", "updatedTime": "0",
        }
        pos = map_position(raw)
        assert pos.qty == Decimal("-1.0")
        assert pos.is_short
        assert pos.margin_type == "isolated"


class TestMapOrder:
    def test_buy_limit_new(self):
        raw = {
            "symbol": "BTCUSDT", "orderId": "abc123",
            "orderLinkId": "link1", "orderStatus": "New",
            "side": "Buy", "orderType": "Limit",
            "qty": "0.01", "price": "69000",
            "cumExecQty": "0", "avgPrice": "0",
            "timeInForce": "GTC", "createdTime": "1700000000000",
        }
        order = map_order(raw)
        assert order.venue == "bybit"
        assert order.side == "buy"
        assert order.order_type == "limit"
        assert order.status == "new"
        assert order.qty == Decimal("0.01")
        assert order.price == Decimal("69000")

    def test_filled_market_order(self):
        raw = {
            "symbol": "BTCUSDT", "orderId": "def456",
            "orderStatus": "Filled", "side": "Sell",
            "orderType": "Market", "qty": "0.05",
            "cumExecQty": "0.05", "avgPrice": "70100",
            "timeInForce": "IOC", "createdTime": "0",
        }
        order = map_order(raw)
        assert order.status == "filled"
        assert order.filled_qty == Decimal("0.05")


class TestMapFill:
    def test_basic_fill(self):
        raw = {
            "symbol": "BTCUSDT", "orderId": "o1",
            "execId": "e1", "side": "Buy",
            "execQty": "0.01", "execPrice": "70000",
            "execFee": "0.42", "feeCurrency": "USDT",
            "isMaker": "false", "execTime": "1700000000000",
        }
        fill = map_fill(raw)
        assert fill.venue == "bybit"
        assert fill.fill_id == "bybit:BTCUSDT:e1"
        assert fill.side == "buy"
        assert fill.qty == Decimal("0.01")
        assert fill.price == Decimal("70000")
        assert fill.fee == Decimal("0.42")
        assert fill.liquidity == "taker"


class TestMapWalletBalance:
    def test_multi_asset(self):
        raw = {
            "list": [{
                "coin": [
                    {"coin": "USDT", "walletBalance": "50000", "locked": "100"},
                    {"coin": "BTC", "walletBalance": "1", "locked": "0"},
                    {"coin": "ETH", "walletBalance": "0", "locked": "0"},
                ]
            }]
        }
        snap = map_balance(raw)
        assert snap.venue == "bybit"
        assert len(snap.balances) == 2  # ETH excluded (zero balance)
        usdt = snap.get("USDT")
        assert usdt is not None
        assert usdt.free == Decimal("49900")
        assert usdt.locked == Decimal("100")


class TestBybitAdapter:
    def _make_adapter(self):
        from execution.adapters.bybit.adapter import BybitAdapter
        cfg = BybitConfig.demo(api_key="test", api_secret="test")
        adapter = BybitAdapter(cfg)
        adapter._connected = True
        return adapter

    def test_send_market_order(self):
        adapter = self._make_adapter()
        adapter._client = MagicMock()
        adapter._client.post.return_value = {
            "retCode": 0,
            "result": {"orderId": "order123", "orderLinkId": ""},
        }
        result = adapter.send_market_order("BTCUSDT", "buy", 0.001)
        assert result["orderId"] == "order123"
        assert result["status"] == "submitted"
        adapter._client.post.assert_called_once()

    def test_send_market_order_error(self):
        adapter = self._make_adapter()
        adapter._client = MagicMock()
        adapter._client.post.return_value = {
            "retCode": 10001, "retMsg": "invalid qty",
        }
        result = adapter.send_market_order("BTCUSDT", "buy", 0.001)
        assert result["status"] == "error"

    def test_cancel_order(self):
        adapter = self._make_adapter()
        adapter._client = MagicMock()
        adapter._client.post.return_value = {"retCode": 0, "retMsg": "OK"}
        result = adapter.cancel_order("BTCUSDT", "order123")
        assert result["status"] == "canceled"

    def test_close_position_no_position(self):
        adapter = self._make_adapter()
        adapter._client = MagicMock()
        adapter._client.get.return_value = {
            "retCode": 0, "result": {"list": []},
        }
        result = adapter.close_position("BTCUSDT")
        assert result["status"] == "no_position"
