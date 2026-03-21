# tests/unit/execution/test_hyperliquid_adapter.py
"""Tests for Hyperliquid venue adapter."""
from __future__ import annotations

import json
import time
from decimal import Decimal
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from execution.adapters.hyperliquid.config import HyperliquidConfig
from execution.adapters.hyperliquid.client import HyperliquidRestClient
from execution.adapters.hyperliquid.mapper import (
    coin_to_symbol,
    map_balance,
    map_fill,
    map_instrument,
    map_order,
    map_orderbook,
    map_position,
    normalize_coin,
)
from execution.adapters.hyperliquid.adapter import HyperliquidAdapter


# =====================================================================
# Config tests
# =====================================================================

class TestHyperliquidConfig:
    def test_mainnet_factory(self):
        cfg = HyperliquidConfig.mainnet()
        assert cfg.base_url == "https://api.hyperliquid.xyz"
        assert cfg.is_mainnet is True
        assert cfg.is_testnet is False
        assert cfg.has_private_key is False

    def test_testnet_factory(self):
        cfg = HyperliquidConfig.testnet()
        assert cfg.base_url == "https://api.hyperliquid-testnet.xyz"
        assert cfg.is_testnet is True
        assert cfg.is_mainnet is False

    def test_mainnet_with_key(self):
        cfg = HyperliquidConfig.mainnet(private_key="0x" + "ab" * 32)
        assert cfg.has_private_key is True
        assert cfg.private_key == "0x" + "ab" * 32

    def test_default_config(self):
        cfg = HyperliquidConfig()
        assert cfg.base_url == "https://api.hyperliquid.xyz"
        assert cfg.private_key == ""
        assert cfg.wallet_address == ""

    def test_frozen(self):
        cfg = HyperliquidConfig.mainnet()
        with pytest.raises(AttributeError):
            cfg.base_url = "https://other.com"  # type: ignore[misc]


# =====================================================================
# Mapper: normalize_coin / coin_to_symbol
# =====================================================================

class TestCoinNormalization:
    def test_normalize_coin_plain(self):
        assert normalize_coin("BTC") == "BTC"
        assert normalize_coin("ETH") == "ETH"

    def test_normalize_coin_usdt(self):
        assert normalize_coin("BTCUSDT") == "BTC"
        assert normalize_coin("ETHUSDT") == "ETH"
        assert normalize_coin("SOLUSDT") == "SOL"

    def test_normalize_coin_lowercase(self):
        assert normalize_coin("btcusdt") == "BTC"
        assert normalize_coin("ethusdt") == "ETH"

    def test_normalize_coin_usd_suffix(self):
        assert normalize_coin("BTCUSD") == "BTC"

    def test_normalize_coin_perp_suffix(self):
        assert normalize_coin("BTCPERP") == "BTC"

    def test_coin_to_symbol(self):
        assert coin_to_symbol("BTC") == "BTCUSDT"
        assert coin_to_symbol("ETH") == "ETHUSDT"
        assert coin_to_symbol("SOL") == "SOLUSDT"

    def test_coin_to_symbol_already_usdt(self):
        assert coin_to_symbol("BTCUSDT") == "BTCUSDT"


# =====================================================================
# Mapper: map_instrument
# =====================================================================

class TestMapInstrument:
    def test_basic(self):
        asset_info = {"name": "BTC", "szDecimals": 5, "maxLeverage": 50}
        inst = map_instrument(asset_info, 0)
        assert inst.venue == "hyperliquid"
        assert inst.symbol == "BTCUSDT"
        assert inst.base_asset == "BTC"
        assert inst.quote_asset == "USDC"
        assert inst.qty_precision == 5
        assert inst.lot_size == Decimal("0.00001")
        assert inst.contract_type == "perpetual"
        assert inst.trading_enabled is True

    def test_with_asset_ctx(self):
        asset_info = {"name": "ETH", "szDecimals": 4, "maxLeverage": 50}
        asset_ctx = {"markPx": "3500.25", "funding": "0.0001"}
        inst = map_instrument(asset_info, 1, asset_ctx)
        assert inst.symbol == "ETHUSDT"
        assert inst.price_precision == 2
        assert inst.tick_size == Decimal("0.01")

    def test_zero_sz_decimals(self):
        asset_info = {"name": "DOGE", "szDecimals": 0, "maxLeverage": 20}
        inst = map_instrument(asset_info, 5)
        assert inst.lot_size == Decimal("1")
        assert inst.min_qty == Decimal("1")


# =====================================================================
# Mapper: map_position
# =====================================================================

class TestMapPosition:
    def test_long_position(self):
        pos = {
            "coin": "BTC",
            "szi": "0.5",
            "entryPx": "68000.0",
            "liquidationPx": "60000.0",
            "unrealizedPnl": "500.0",
            "leverage": {"type": "cross", "value": 10},
        }
        vp = map_position(pos)
        assert vp.venue == "hyperliquid"
        assert vp.symbol == "BTCUSDT"
        assert vp.qty == Decimal("0.5")
        assert vp.is_long is True
        assert vp.entry_price == Decimal("68000.0")
        assert vp.liquidation_price == Decimal("60000.0")
        assert vp.leverage == 10
        assert vp.margin_type == "cross"

    def test_short_position(self):
        pos = {
            "coin": "ETH",
            "szi": "-2.0",
            "entryPx": "3500.0",
            "unrealizedPnl": "-100.0",
            "leverage": {"type": "isolated", "value": 5},
        }
        vp = map_position(pos)
        assert vp.qty == Decimal("-2.0")
        assert vp.is_short is True
        assert vp.margin_type == "isolated"

    def test_flat_position(self):
        pos = {"coin": "SOL", "szi": "0", "entryPx": None}
        vp = map_position(pos)
        assert vp.is_flat is True


# =====================================================================
# Mapper: map_order
# =====================================================================

class TestMapOrder:
    def test_open_limit_buy(self):
        order = {
            "coin": "BTC",
            "oid": 12345,
            "side": "B",
            "limitPx": "67000.0",
            "sz": "0.1",
            "orderType": {"limit": {"tif": "Gtc"}},
            "timestamp": 1700000000000,
        }
        co = map_order(order)
        assert co.venue == "hyperliquid"
        assert co.symbol == "BTCUSDT"
        assert co.order_id == "12345"
        assert co.side == "buy"
        assert co.order_type == "limit"
        assert co.tif == "gtc"
        assert co.qty == Decimal("0.1")
        assert co.price == Decimal("67000.0")
        assert co.status == "new"

    def test_sell_order(self):
        order = {
            "coin": "ETH",
            "oid": 99999,
            "side": "A",
            "limitPx": "3600.0",
            "sz": "1.5",
            "orderType": {"limit": {"tif": "Ioc"}},
        }
        co = map_order(order)
        assert co.side == "sell"
        assert co.tif == "ioc"

    def test_order_with_filled_sz(self):
        order = {
            "coin": "SOL",
            "oid": 555,
            "side": "B",
            "limitPx": "150.0",
            "sz": "10.0",
            "filledSz": "5.0",
            "avgPx": "149.5",
        }
        co = map_order(order)
        assert co.filled_qty == Decimal("5.0")
        assert co.avg_price == Decimal("149.5")


# =====================================================================
# Mapper: map_fill
# =====================================================================

class TestMapFill:
    def test_basic_fill(self):
        fill = {
            "coin": "BTC",
            "oid": 12345,
            "tid": 67890,
            "side": "B",
            "px": "68000.0",
            "sz": "0.01",
            "fee": "0.68",
            "time": 1700000000000,
            "crossed": True,
        }
        cf = map_fill(fill)
        assert cf.venue == "hyperliquid"
        assert cf.symbol == "BTCUSDT"
        assert cf.order_id == "12345"
        assert cf.trade_id == "67890"
        assert cf.side == "buy"
        assert cf.qty == Decimal("0.01")
        assert cf.price == Decimal("68000.0")
        assert cf.fee == Decimal("0.68")
        assert cf.fee_asset == "USDC"
        assert cf.liquidity == "taker"
        assert cf.ts_ms == 1700000000000
        assert cf.fill_id == "hyperliquid:BTCUSDT:67890"
        assert cf.payload_digest  # Non-empty

    def test_maker_fill(self):
        fill = {
            "coin": "ETH",
            "oid": 100,
            "tid": 200,
            "side": "A",
            "px": "3500.0",
            "sz": "1.0",
            "fee": "-0.35",  # Negative fee = maker rebate
            "time": 1700000000000,
            "crossed": False,
        }
        cf = map_fill(fill)
        assert cf.side == "sell"
        assert cf.liquidity == "maker"
        assert cf.fee == Decimal("0.35")  # Absolute value

    def test_fill_with_dir(self):
        fill = {
            "coin": "SOL",
            "oid": 300,
            "tid": 400,
            "dir": "Open Long",
            "px": "150.0",
            "sz": "10.0",
            "fee": "0.15",
            "time": 1700000000000,
        }
        cf = map_fill(fill)
        assert cf.side == "buy"


# =====================================================================
# Mapper: map_balance
# =====================================================================

class TestMapBalance:
    def test_margin_summary(self):
        state = {
            "marginSummary": {
                "accountValue": "10000.0",
                "totalMarginUsed": "3000.0",
            },
        }
        bs = map_balance(state)
        assert bs.venue == "hyperliquid"
        assert len(bs.balances) == 1
        b = bs.balances[0]
        assert b.asset == "USDC"
        assert b.free == Decimal("7000")
        assert b.locked == Decimal("3000.0")
        assert b.total == Decimal("10000")

    def test_cross_margin_summary(self):
        state = {
            "crossMarginSummary": {
                "accountValue": "5000.0",
                "totalMarginUsed": "1000.0",
            },
        }
        bs = map_balance(state)
        assert len(bs.balances) == 1
        b = bs.balances[0]
        assert b.free == Decimal("4000")

    def test_empty_state(self):
        bs = map_balance({})
        assert len(bs.balances) == 0


# =====================================================================
# Mapper: map_orderbook
# =====================================================================

class TestMapOrderbook:
    def test_basic_book(self):
        data = {
            "coin": "BTC",
            "levels": [
                [
                    {"px": "67990.0", "sz": "1.5", "n": 3},
                    {"px": "67980.0", "sz": "2.0", "n": 5},
                ],
                [
                    {"px": "68000.0", "sz": "0.8", "n": 2},
                    {"px": "68010.0", "sz": "1.2", "n": 4},
                ],
            ],
        }
        book = map_orderbook(data)
        assert len(book["bids"]) == 2
        assert len(book["asks"]) == 2
        assert book["bids"][0][0] == 67990.0
        assert book["bids"][0][1] == 1.5
        assert book["asks"][0][0] == 68000.0
        assert book["coin"] == "BTC"

    def test_empty_book(self):
        data = {"levels": [[], []]}
        book = map_orderbook(data)
        assert book["bids"] == []
        assert book["asks"] == []


# =====================================================================
# Client: info_request
# =====================================================================

class TestHyperliquidClient:
    def _make_client(self):
        cfg = HyperliquidConfig.mainnet()
        return HyperliquidRestClient(cfg)

    @patch("execution.adapters.hyperliquid.client.urlopen")
    def test_info_request_sends_correct_payload(self, mock_urlopen):
        resp = MagicMock()
        resp.read.return_value = b'{"universe": []}'
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = resp

        client = self._make_client()
        result = client.info_request({"type": "meta"})

        # Verify URL
        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        assert req.full_url == "https://api.hyperliquid.xyz/info"
        assert req.method == "POST"
        # Verify body
        body = json.loads(req.data.decode())
        assert body == {"type": "meta"}

    @patch("execution.adapters.hyperliquid.client.urlopen")
    def test_info_request_parses_response(self, mock_urlopen):
        resp = MagicMock()
        resp.read.return_value = json.dumps([
            {"universe": [{"name": "BTC", "szDecimals": 5}]},
            [{"funding": "0.0001", "markPx": "70000.0"}],
        ]).encode()
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = resp

        client = self._make_client()
        result = client.info_request({"type": "metaAndAssetCtxs"})

        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["universe"][0]["name"] == "BTC"

    @patch("execution.adapters.hyperliquid.client.urlopen")
    def test_info_request_handles_http_error(self, mock_urlopen):
        from urllib.error import HTTPError
        error = HTTPError(
            "https://api.hyperliquid.xyz/info", 429,
            "Too Many Requests", {}, MagicMock(read=MagicMock(return_value=b"rate limited")),
        )
        error.read = MagicMock(return_value=b"rate limited")
        mock_urlopen.side_effect = error

        client = self._make_client()
        result = client.info_request({"type": "meta"})
        assert result["status"] == "error"
        assert result["code"] == 429
        assert result["retryable"] is True


# =====================================================================
# Adapter: unit tests with mocked client
# =====================================================================

class TestHyperliquidAdapter:
    def _make_adapter(self):
        cfg = HyperliquidConfig.mainnet()
        adapter = HyperliquidAdapter(cfg)
        # Pre-populate asset index cache
        adapter._universe = [
            {"name": "BTC", "szDecimals": 5, "maxLeverage": 50},
            {"name": "ETH", "szDecimals": 4, "maxLeverage": 50},
            {"name": "SOL", "szDecimals": 2, "maxLeverage": 20},
        ]
        adapter._asset_ctxs = [
            {"markPx": "70000.0", "funding": "0.0001", "openInterest": "1000", "dayNtlVlm": "500000000"},
            {"markPx": "3500.0", "funding": "0.00005", "openInterest": "5000", "dayNtlVlm": "200000000"},
            {"markPx": "150.0", "funding": "0.0002", "openInterest": "20000", "dayNtlVlm": "50000000"},
        ]
        adapter._asset_index = {"BTC": 0, "ETH": 1, "SOL": 2}
        return adapter

    def test_list_instruments(self):
        adapter = self._make_adapter()
        instruments = adapter.list_instruments()
        assert len(instruments) == 3
        assert instruments[0].symbol == "BTCUSDT"
        assert instruments[1].symbol == "ETHUSDT"
        assert instruments[2].symbol == "SOLUSDT"

    def test_list_instruments_filtered(self):
        adapter = self._make_adapter()
        instruments = adapter.list_instruments(symbols=["BTCUSDT", "ETHUSDT"])
        assert len(instruments) == 2
        symbols = {i.symbol for i in instruments}
        assert "BTCUSDT" in symbols
        assert "ETHUSDT" in symbols
        assert "SOLUSDT" not in symbols

    def test_list_instruments_by_coin_name(self):
        adapter = self._make_adapter()
        instruments = adapter.list_instruments(symbols=["BTC"])
        assert len(instruments) == 1
        assert instruments[0].symbol == "BTCUSDT"

    def test_get_asset_index(self):
        adapter = self._make_adapter()
        assert adapter._get_asset_index("BTC") == 0
        assert adapter._get_asset_index("BTCUSDT") == 0
        assert adapter._get_asset_index("ETH") == 1
        assert adapter._get_asset_index("ETHUSDT") == 1

    def test_get_asset_index_unknown(self):
        adapter = self._make_adapter()
        # Mock _refresh_meta to not change anything
        adapter._refresh_meta = lambda: True  # type: ignore[assignment]
        with pytest.raises(ValueError, match="Unknown coin"):
            adapter._get_asset_index("XYZUSDT")

    def test_get_ticker(self):
        adapter = self._make_adapter()
        adapter._client.info_request = MagicMock(return_value={
            "coin": "BTC",
            "levels": [
                [{"px": "69990.0", "sz": "1.0", "n": 2}],
                [{"px": "70010.0", "sz": "0.5", "n": 1}],
            ],
        })
        ticker = adapter.get_ticker("BTC")
        assert ticker["symbol"] == "BTCUSDT"
        assert ticker["bid1Price"] == 69990.0
        assert ticker["ask1Price"] == 70010.0
        assert ticker["lastPrice"] == 70000.0  # mid price

    def test_get_orderbook(self):
        adapter = self._make_adapter()
        adapter._client.info_request = MagicMock(return_value={
            "coin": "ETH",
            "levels": [
                [{"px": "3499.0", "sz": "5.0", "n": 3}],
                [{"px": "3501.0", "sz": "3.0", "n": 2}],
            ],
        })
        book = adapter.get_orderbook("ETH")
        assert len(book["bids"]) == 1
        assert len(book["asks"]) == 1
        assert book["bids"][0][0] == 3499.0

    def test_get_positions_no_wallet(self):
        cfg = HyperliquidConfig.mainnet()
        adapter = HyperliquidAdapter(cfg)
        assert adapter.get_positions() == ()

    def test_get_balances_no_wallet(self):
        cfg = HyperliquidConfig.mainnet()
        adapter = HyperliquidAdapter(cfg)
        bal = adapter.get_balances()
        assert len(bal.balances) == 0

    def test_get_open_orders_no_wallet(self):
        cfg = HyperliquidConfig.mainnet()
        adapter = HyperliquidAdapter(cfg)
        assert adapter.get_open_orders() == ()

    def test_get_recent_fills_no_wallet(self):
        cfg = HyperliquidConfig.mainnet()
        adapter = HyperliquidAdapter(cfg)
        assert adapter.get_recent_fills() == ()

    def test_get_funding_rates(self):
        adapter = self._make_adapter()
        rates = adapter.get_funding_rates()
        assert len(rates) == 3
        assert rates[0]["symbol"] == "BTCUSDT"
        assert rates[0]["fundingRate"] == 0.0001
        assert rates[1]["symbol"] == "ETHUSDT"

    def test_close_position_no_position(self):
        adapter = self._make_adapter()
        adapter._config = HyperliquidConfig.mainnet(
            wallet_address="0x" + "00" * 20,
        )
        adapter._client.info_request = MagicMock(return_value={
            "assetPositions": [],
            "marginSummary": {"accountValue": "1000", "totalMarginUsed": "0"},
        })
        result = adapter.close_position("BTC")
        assert result["status"] == "no_position"

    def test_venue_attribute(self):
        cfg = HyperliquidConfig.mainnet()
        adapter = HyperliquidAdapter(cfg)
        assert adapter.venue == "hyperliquid"


# =====================================================================
# Send order tests (require private key — skipped)
# =====================================================================

@pytest.mark.skip(reason="needs private key")
class TestHyperliquidAdapterTrading:
    def test_send_market_order(self):
        pass

    def test_send_limit_order(self):
        pass

    def test_cancel_order(self):
        pass

    def test_set_leverage(self):
        pass


# =====================================================================
# Real API connectivity test (public endpoint)
# =====================================================================

class TestHyperliquidRealAPI:
    """Tests that hit the real Hyperliquid API (public info endpoint).

    These verify actual connectivity and response format.
    """

    def test_real_meta(self):
        """Fetch real market metadata from Hyperliquid."""
        cfg = HyperliquidConfig.mainnet()
        client = HyperliquidRestClient(cfg)
        data = client.info_request({"type": "meta"})
        assert isinstance(data, dict), f"Expected dict, got {type(data)}: {data}"
        assert "universe" in data
        universe = data["universe"]
        assert len(universe) > 0, "Universe should have at least one asset"

        # BTC should always be present
        names = [a["name"] for a in universe]
        assert "BTC" in names, f"BTC not in universe: {names[:10]}"
        assert "ETH" in names, f"ETH not in universe: {names[:10]}"

        # Check structure
        btc = next(a for a in universe if a["name"] == "BTC")
        assert "szDecimals" in btc
        assert isinstance(btc["szDecimals"], int)

    def test_real_meta_and_asset_ctxs(self):
        """Fetch real metadata with asset contexts."""
        cfg = HyperliquidConfig.mainnet()
        client = HyperliquidRestClient(cfg)
        data = client.info_request({"type": "metaAndAssetCtxs"})
        assert isinstance(data, list), f"Expected list, got {type(data)}"
        assert len(data) >= 2
        meta = data[0]
        ctxs = data[1]
        assert "universe" in meta
        assert len(ctxs) > 0
        # BTC context should have mark price
        btc_idx = next(i for i, a in enumerate(meta["universe"]) if a["name"] == "BTC")
        btc_ctx = ctxs[btc_idx]
        assert "markPx" in btc_ctx
        assert float(btc_ctx["markPx"]) > 0

    def test_real_l2_book(self):
        """Fetch real BTC orderbook."""
        cfg = HyperliquidConfig.mainnet()
        client = HyperliquidRestClient(cfg)
        data = client.info_request({"type": "l2Book", "coin": "BTC"})
        assert isinstance(data, dict), f"Expected dict, got {type(data)}"
        assert "levels" in data
        levels = data["levels"]
        assert len(levels) == 2  # bids, asks
        # Should have some levels
        bids = levels[0]
        asks = levels[1]
        assert len(bids) > 0, "No bids in BTC orderbook"
        assert len(asks) > 0, "No asks in BTC orderbook"
        # Verify price format
        assert "px" in bids[0]
        assert "sz" in bids[0]

    def test_real_adapter_list_instruments(self):
        """Test full adapter list_instruments with real API."""
        cfg = HyperliquidConfig.mainnet()
        adapter = HyperliquidAdapter(cfg)
        instruments = adapter.list_instruments()
        assert len(instruments) > 0

        # Find BTC
        btc = [i for i in instruments if i.symbol == "BTCUSDT"]
        assert len(btc) == 1, "BTCUSDT should be listed"
        assert btc[0].base_asset == "BTC"
        assert btc[0].contract_type == "perpetual"
        assert btc[0].venue == "hyperliquid"
        assert btc[0].trading_enabled is True

    def test_real_adapter_get_ticker(self):
        """Test get_ticker with real API."""
        cfg = HyperliquidConfig.mainnet()
        adapter = HyperliquidAdapter(cfg)
        # Need meta first for funding rate
        adapter._refresh_meta()
        ticker = adapter.get_ticker("BTC")
        assert ticker, "Ticker should not be empty"
        assert ticker["symbol"] == "BTCUSDT"
        assert ticker["lastPrice"] > 0
        assert ticker["bid1Price"] > 0
        assert ticker["ask1Price"] > 0
        # Bid should be <= ask
        assert ticker["bid1Price"] <= ticker["ask1Price"]

    def test_real_adapter_connect_readonly(self):
        """Test connect in read-only mode (no private key)."""
        cfg = HyperliquidConfig.mainnet()
        adapter = HyperliquidAdapter(cfg)
        assert adapter.connect() is True
        assert adapter.is_connected() is True
