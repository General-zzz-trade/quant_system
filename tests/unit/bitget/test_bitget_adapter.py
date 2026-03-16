"""Tests for Bitget adapter — config, client signing, mapper."""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from execution.adapters.bitget.config import BitgetConfig
from execution.adapters.bitget.client import BitgetRestClient
from execution.adapters.bitget.mapper import (
    map_balance, map_fill, map_instrument, map_order, map_position,
)


# ── Config ───────────────────────────────────────────────────────

class TestBitgetConfig:
    def test_defaults(self):
        cfg = BitgetConfig()
        assert cfg.base_url == "https://api.bitget.com"
        assert cfg.product_type == "USDT-FUTURES"
        assert cfg.margin_coin == "USDT"

    def test_demo_factory(self):
        cfg = BitgetConfig.demo(api_key="k", api_secret="s", passphrase="p")
        assert cfg.api_key == "k"
        assert cfg.passphrase == "p"

    def test_is_demo(self):
        cfg = BitgetConfig(base_url="https://api-demo.bitget.com")
        assert cfg.is_demo

    def test_frozen(self):
        cfg = BitgetConfig()
        with pytest.raises(AttributeError):
            cfg.api_key = "new"  # type: ignore[misc]


# ── Client Signing ────────────────────────────────────────────────

class TestBitgetSigning:
    def test_sign_format(self):
        cfg = BitgetConfig(api_key="testkey", api_secret="testsecret", passphrase="testpass")
        client = BitgetRestClient(cfg)

        ts = "1700000000000"
        sig = client._sign(ts, "GET", "/api/v2/mix/account/accounts")

        # Verify: base64(hmac_sha256(secret, ts+METHOD+path))
        message = ts + "GET" + "/api/v2/mix/account/accounts"
        expected = base64.b64encode(
            hmac.new(b"testsecret", message.encode(), hashlib.sha256).digest()
        ).decode()
        assert sig == expected

    def test_sign_post_with_body(self):
        cfg = BitgetConfig(api_key="k", api_secret="s", passphrase="p")
        client = BitgetRestClient(cfg)

        body = json.dumps({"symbol": "ETHUSDT", "side": "buy"})
        ts = "1700000000000"
        sig = client._sign(ts, "POST", "/api/v2/mix/order/place-order", body)

        message = ts + "POST" + "/api/v2/mix/order/place-order" + body
        expected = base64.b64encode(
            hmac.new(b"s", message.encode(), hashlib.sha256).digest()
        ).decode()
        assert sig == expected

    def test_headers_include_passphrase(self):
        cfg = BitgetConfig(api_key="mykey", api_secret="sec", passphrase="mypass")
        client = BitgetRestClient(cfg)
        headers = client._headers("12345", "sig123")
        assert headers["ACCESS-KEY"] == "mykey"
        assert headers["ACCESS-SIGN"] == "sig123"
        assert headers["ACCESS-TIMESTAMP"] == "12345"
        assert headers["ACCESS-PASSPHRASE"] == "mypass"


# ── Mapper ────────────────────────────────────────────────────────

class TestMapInstrument:
    def test_basic(self):
        raw = {
            "symbol": "ETHUSDT",
            "baseCoin": "ETH",
            "quoteCoin": "USDT",
            "pricePlace": "2",
            "volumePlace": "4",
            "priceEndStep": "0.01",
            "minTradeNum": "0.01",
            "minTradeUSDT": "5",
            "symbolStatus": "normal",
        }
        inst = map_instrument(raw)
        assert inst.symbol == "ETHUSDT"
        assert inst.venue == "bitget"
        assert inst.base_asset == "ETH"
        assert inst.min_qty == Decimal("0.01")
        assert inst.trading_enabled is True


class TestMapBalance:
    def test_basic(self):
        raw = {"list": [{
            "marginCoin": "USDT",
            "available": "500.50",
            "locked": "10.00",
            "usdtEquity": "510.50",
        }]}
        snap = map_balance(raw)
        assert len(snap.balances) == 1
        b = snap.balances[0]
        assert b.asset == "USDT"
        assert b.free == Decimal("500.50")
        assert b.total == Decimal("510.50")


class TestMapPosition:
    def test_long(self):
        raw = {
            "symbol": "ETHUSDT",
            "holdSide": "long",
            "total": "0.5",
            "openPriceAvg": "2000",
            "markPrice": "2050",
            "leverage": "3",
            "unrealizedPL": "25",
            "marginMode": "crossed",
        }
        pos = map_position(raw)
        assert pos.qty == Decimal("0.5")
        assert pos.entry_price == Decimal("2000")

    def test_short_negative_qty(self):
        raw = {
            "symbol": "ETHUSDT",
            "holdSide": "short",
            "total": "0.3",
            "openPriceAvg": "2100",
            "markPrice": "2050",
            "leverage": "2",
            "unrealizedPL": "15",
        }
        pos = map_position(raw)
        assert pos.qty == Decimal("-0.3")


class TestMapOrder:
    def test_status_mapping(self):
        raw = {"status": "live", "symbol": "ETHUSDT", "side": "buy",
               "orderType": "limit", "size": "0.1", "price": "2000", "orderId": "123"}
        order = map_order(raw)
        assert order.status == "new"
        assert order.side == "buy"

    def test_filled(self):
        raw = {"status": "filled", "symbol": "ETHUSDT", "side": "sell",
               "size": "0.5", "priceAvg": "2100", "orderId": "456"}
        order = map_order(raw)
        assert order.status == "filled"


class TestMapFill:
    def test_basic(self):
        raw = {
            "symbol": "ETHUSDT",
            "tradeId": "t123",
            "orderId": "o456",
            "side": "buy",
            "size": "0.1",
            "price": "2000",
            "fee": "-0.08",
            "tradeScope": "taker",
        }
        fill = map_fill(raw)
        assert fill.venue == "bitget"
        assert fill.symbol == "ETHUSDT"
        assert fill.qty == Decimal("0.1")
        assert fill.liquidity == "taker"
