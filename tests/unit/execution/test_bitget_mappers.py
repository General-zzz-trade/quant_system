# tests/unit/execution/test_bitget_mappers.py
"""Tests for all Bitget mappers — sample JSON comparison tests."""
from __future__ import annotations

from decimal import Decimal
import pytest

from execution.adapters.bitget.mapper_order import BitgetOrderMapper
from execution.adapters.bitget.mapper_fill import BitgetFillMapper
from execution.adapters.bitget.mapper_balance import map_balance, map_balances
from execution.adapters.bitget.mapper_position import map_position, map_positions
from execution.adapters.bitget.mapper_instrument import map_instrument, map_instruments


# ------------------------------------------------------------------
# Order mapper
# ------------------------------------------------------------------
class TestBitgetOrderMapper:
    def setup_method(self):
        self.mapper = BitgetOrderMapper()

    def test_rest_order_basic(self):
        raw = {
            "orderId": "1234567890",
            "clientOid": "my-client-1",
            "symbol": "BTCUSDT",
            "side": "buy",
            "orderType": "limit",
            "status": "new",
            "size": "0.01",
            "price": "50000.5",
            "baseVolume": "0",
            "priceAvg": "0",
            "force": "gtc",
            "cTime": "1709000000000",
        }
        order = self.mapper.map_order(raw)
        assert order.venue == "bitget"
        assert order.symbol == "BTCUSDT"
        assert order.order_id == "1234567890"
        assert order.client_order_id == "my-client-1"
        assert order.side == "buy"
        assert order.status == "new"
        assert order.order_type == "limit"
        assert order.tif == "gtc"
        assert order.qty == Decimal("0.01")
        assert order.price == Decimal("50000.5")
        assert order.filled_qty == Decimal("0")
        assert order.ts_ms == 1709000000000
        assert order.order_key == "bitget:BTCUSDT:order:1234567890"
        assert order.payload_digest  # non-empty hash
        assert order.raw is raw

    def test_rest_order_filled(self):
        raw = {
            "orderId": "9876543210",
            "clientOid": "",
            "symbol": "ETHUSDT",
            "side": "sell",
            "orderType": "market",
            "status": "filled",
            "size": "1.5",
            "price": "0",
            "baseVolume": "1.5",
            "priceAvg": "3200.25",
            "force": "ioc",
            "cTime": "1709000100000",
        }
        order = self.mapper.map_order(raw)
        assert order.status == "filled"
        assert order.side == "sell"
        assert order.filled_qty == Decimal("1.5")
        assert order.avg_price == Decimal("3200.25")
        assert order.client_order_id is None  # empty string → None

    def test_rest_order_cancelled(self):
        raw = {
            "orderId": "111",
            "symbol": "BTCUSDT",
            "side": "buy",
            "orderType": "limit",
            "status": "cancelled",
            "size": "0.5",
            "price": "40000",
            "baseVolume": "0",
            "priceAvg": "0",
            "cTime": "1709000200000",
        }
        order = self.mapper.map_order(raw)
        assert order.status == "canceled"  # normalized to American spelling

    def test_ws_order_format(self):
        raw = {
            "instId": "BTCUSDT",
            "ordId": "555",
            "clOrdId": "ws-client-1",
            "side": "buy",
            "ordType": "limit",
            "status": "partially_filled",
            "sz": "2.0",
            "px": "45000",
            "accBaseVolume": "0.5",
            "avgPx": "44999.5",
            "force": "gtc",
            "uTime": "1709000300000",
        }
        order = self.mapper.map_order(raw)
        assert order.symbol == "BTCUSDT"
        assert order.order_id == "555"
        assert order.client_order_id == "ws-client-1"
        assert order.status == "partially_filled"
        assert order.filled_qty == Decimal("0.5")
        assert order.avg_price == Decimal("44999.5")

    def test_missing_order_id_raises(self):
        raw = {"symbol": "BTCUSDT", "side": "buy"}
        with pytest.raises(ValueError):
            self.mapper.map_order(raw)

    def test_missing_symbol_raises(self):
        raw = {"orderId": "123", "side": "buy", "orderType": "limit", "status": "new", "size": "1", "cTime": "1"}
        with pytest.raises(ValueError, match="symbol"):
            self.mapper.map_order({**raw, "symbol": None})

    def test_invalid_side_raises(self):
        raw = {
            "orderId": "123", "symbol": "BTCUSDT", "side": "invalid",
            "orderType": "limit", "status": "new", "size": "1", "cTime": "1",
        }
        with pytest.raises(ValueError, match="unsupported side"):
            self.mapper.map_order(raw)

    def test_zero_qty_raises(self):
        raw = {
            "orderId": "123", "symbol": "BTCUSDT", "side": "buy",
            "orderType": "limit", "status": "new", "size": "0",
            "price": "100", "cTime": "1",
        }
        with pytest.raises(ValueError, match="qty must be >0"):
            self.mapper.map_order(raw)

    def test_status_normalization(self):
        """Bitget-specific statuses should map correctly."""
        base = {
            "orderId": "1", "symbol": "BTCUSDT", "side": "buy",
            "orderType": "limit", "size": "1", "cTime": "1",
        }
        for bitget_status, expected in [
            ("live", "new"),
            ("init", "new"),
            ("partial_fill", "partially_filled"),
            ("full_fill", "filled"),
            ("cancelled", "canceled"),
        ]:
            order = self.mapper.map_order({**base, "status": bitget_status})
            assert order.status == expected, f"{bitget_status} → {order.status}"


# ------------------------------------------------------------------
# Fill mapper
# ------------------------------------------------------------------
class TestBitgetFillMapper:
    def setup_method(self):
        self.mapper = BitgetFillMapper()

    def test_rest_fill_basic(self):
        raw = {
            "tradeId": "T123456",
            "orderId": "O789",
            "symbol": "BTCUSDT",
            "side": "buy",
            "baseVolume": "0.5",
            "price": "50000",
            "fee": "-0.025",
            "feeCoin": "USDT",
            "cTime": "1709000000000",
            "tradeScope": "taker",
        }
        fill = self.mapper.map_fill(raw)
        assert fill.venue == "bitget"
        assert fill.symbol == "BTCUSDT"
        assert fill.order_id == "O789"
        assert fill.trade_id == "T123456"
        assert fill.fill_id == "bitget:BTCUSDT:T123456"
        assert fill.side == "buy"
        assert fill.qty == Decimal("0.5")
        assert fill.price == Decimal("50000")
        assert fill.fee == Decimal("0.025")  # abs value
        assert fill.fee_asset == "USDT"
        assert fill.liquidity == "taker"
        assert fill.ts_ms == 1709000000000
        assert fill.payload_digest  # non-empty
        assert fill.raw is raw

    def test_fill_with_fee_detail(self):
        raw = {
            "tradeId": "T999",
            "orderId": "O111",
            "symbol": "ETHUSDT",
            "side": "sell",
            "baseVolume": "2.0",
            "price": "3200",
            "feeDetail": [{"totalFee": "-1.6", "feeCoin": "USDT"}],
            "cTime": "1709000100000",
        }
        fill = self.mapper.map_fill(raw)
        assert fill.fee == Decimal("1.6")
        assert fill.fee_asset == "USDT"

    def test_fill_maker_liquidity(self):
        raw = {
            "tradeId": "T888",
            "orderId": "O222",
            "symbol": "BTCUSDT",
            "side": "buy",
            "baseVolume": "1",
            "price": "50000",
            "fee": "0.01",
            "feeCoin": "USDT",
            "cTime": "1709000200000",
            "tradeScope": "maker",
        }
        fill = self.mapper.map_fill(raw)
        assert fill.liquidity == "maker"

    def test_fill_missing_trade_id_raises(self):
        raw = {
            "orderId": "O1",
            "symbol": "BTCUSDT",
            "side": "buy",
            "baseVolume": "1",
            "price": "50000",
            "cTime": "1",
        }
        with pytest.raises(ValueError):
            self.mapper.map_fill(raw)

    def test_fill_zero_qty_raises(self):
        raw = {
            "tradeId": "T1",
            "orderId": "O1",
            "symbol": "BTCUSDT",
            "side": "buy",
            "baseVolume": "0",
            "price": "50000",
            "cTime": "1",
        }
        with pytest.raises(ValueError, match="qty must be >0"):
            self.mapper.map_fill(raw)


# ------------------------------------------------------------------
# Balance mapper
# ------------------------------------------------------------------
class TestBitgetBalanceMapper:
    def test_single_balance(self):
        raw = {
            "marginCoin": "usdt",
            "available": "1000.50",
            "locked": "200.25",
            "accountEquity": "1200.75",
        }
        bal = map_balance(raw)
        assert bal.venue == "bitget"
        assert bal.asset == "USDT"
        assert bal.free == Decimal("1000.50")
        assert bal.locked == Decimal("200.25")
        assert bal.total == Decimal("1200.75")

    def test_balance_fallback_total(self):
        """If accountEquity is 0, total = free + locked."""
        raw = {
            "marginCoin": "BTC",
            "available": "0.5",
            "locked": "0.1",
            "accountEquity": "0",
        }
        bal = map_balance(raw)
        assert bal.total == Decimal("0.6")

    def test_batch_balances(self):
        raws = [
            {"marginCoin": "USDT", "available": "100", "locked": "10", "accountEquity": "110"},
            {"marginCoin": "BTC", "available": "0.5", "locked": "0", "accountEquity": "0.5"},
        ]
        balances = map_balances(raws)
        assert len(balances) == 2
        assert balances[0].asset == "USDT"
        assert balances[1].asset == "BTC"


# ------------------------------------------------------------------
# Position mapper
# ------------------------------------------------------------------
class TestBitgetPositionMapper:
    def test_long_position(self):
        raw = {
            "symbol": "BTCUSDT",
            "holdSide": "long",
            "total": "0.5",
            "openPriceAvg": "50000",
            "unrealizedPL": "250.5",
            "leverage": "10",
            "marginMode": "crossed",
        }
        pos = map_position(raw)
        assert pos.venue == "bitget"
        assert pos.symbol == "BTCUSDT"
        assert pos.qty == Decimal("0.5")  # positive for long
        assert pos.entry_price == Decimal("50000")
        assert pos.unrealized_pnl == Decimal("250.5")
        assert pos.leverage == 10
        assert pos.margin_type == "crossed"

    def test_short_position(self):
        raw = {
            "symbol": "ETHUSDT",
            "holdSide": "short",
            "total": "2.0",
            "openPriceAvg": "3200",
            "unrealizedPL": "-50",
            "leverage": "5",
            "marginMode": "isolated",
        }
        pos = map_position(raw)
        assert pos.qty == Decimal("-2.0")  # negative for short
        assert pos.is_short
        assert pos.margin_type == "isolated"

    def test_zero_position_filtered(self):
        raws = [
            {"symbol": "BTCUSDT", "holdSide": "long", "total": "0", "leverage": "1"},
            {"symbol": "ETHUSDT", "holdSide": "long", "total": "1.0", "leverage": "5"},
        ]
        positions = map_positions(raws)
        assert len(positions) == 1
        assert positions[0].symbol == "ETHUSDT"


# ------------------------------------------------------------------
# Instrument mapper
# ------------------------------------------------------------------
class TestBitgetInstrumentMapper:
    def test_basic_contract(self):
        raw = {
            "symbol": "BTCUSDT",
            "baseCoin": "BTC",
            "quoteCoin": "USDT",
            "pricePrecision": "2",
            "quantityPrecision": "4",
            "priceEndStep": "0.01",
            "sizeMultiplier": "0.0001",
            "minTradeNum": "0.001",
            "maxSymbolOpenNum": "100",
            "minTradeUSDT": "5",
        }
        inst = map_instrument(raw)
        assert inst.venue == "bitget"
        assert inst.symbol == "BTCUSDT"
        assert inst.base_asset == "BTC"
        assert inst.quote_asset == "USDT"
        assert inst.price_precision == 2
        assert inst.qty_precision == 4
        assert inst.tick_size == Decimal("0.01")
        assert inst.lot_size == Decimal("0.0001")
        assert inst.min_qty == Decimal("0.001")
        assert inst.max_qty == Decimal("100")
        assert inst.min_notional == Decimal("5")
        assert inst.contract_type == "perpetual"
        assert inst.margin_asset == "USDT"

    def test_missing_optional_fields(self):
        raw = {
            "symbol": "ETHUSDT",
            "baseCoin": "ETH",
            "quoteCoin": "USDT",
        }
        inst = map_instrument(raw)
        assert inst.symbol == "ETHUSDT"
        assert inst.price_precision == 2  # default
        assert inst.qty_precision == 3    # default

    def test_batch_instruments(self):
        raws = [
            {"symbol": "BTCUSDT", "baseCoin": "BTC", "quoteCoin": "USDT"},
            {"symbol": "ETHUSDT", "baseCoin": "ETH", "quoteCoin": "USDT"},
        ]
        instruments = map_instruments(raws)
        assert len(instruments) == 2
