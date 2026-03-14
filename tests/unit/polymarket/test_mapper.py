"""Tests for Polymarket fill and order mappers."""
from decimal import Decimal

from execution.adapters.polymarket.mapper import fill_from_polymarket, order_from_polymarket
from execution.models.fills import CanonicalFill
from execution.models.orders import CanonicalOrder


class TestFillMapper:
    def test_basic_fill(self):
        raw = {
            "id": "trade123",
            "order_id": "order456",
            "market": "btc-100k",
            "asset_id": "t1",
            "side": "BUY",
            "size": "10",
            "price": "0.65",
            "fee": "0.01",
            "timestamp": 1700000000000,
        }
        fill = fill_from_polymarket(raw, venue="polymarket")
        assert isinstance(fill, CanonicalFill)
        assert fill.venue == "polymarket"
        assert fill.trade_id == "trade123"
        assert fill.order_id == "order456"
        assert fill.side == "buy"
        assert fill.qty == Decimal("10")
        assert fill.price == Decimal("0.65")
        assert fill.fee == Decimal("0.01")
        assert fill.ts_ms == 1700000000000
        assert fill.fill_id  # non-empty
        assert fill.raw is raw

    def test_fill_sell_side(self):
        raw = {
            "id": "t1",
            "order_id": "o1",
            "market": "some-market",
            "asset_id": "tok1",
            "side": "SELL",
            "size": "5",
            "price": "0.80",
            "timestamp": 1700000001000,
        }
        fill = fill_from_polymarket(raw, venue="polymarket")
        assert fill.side == "sell"
        assert fill.fee == Decimal("0")

    def test_fill_symbol_from_market(self):
        raw = {
            "id": "t2",
            "order_id": "o2",
            "market": "eth-above-5k",
            "asset_id": "tok2",
            "side": "buy",
            "size": "1",
            "price": "0.50",
            "timestamp": 1700000002000,
        }
        fill = fill_from_polymarket(raw, venue="polymarket")
        assert "eth-above-5k" in fill.symbol.lower() or "tok2" in fill.symbol


class TestOrderMapper:
    def test_basic_order(self):
        raw = {
            "id": "order789",
            "market": "btc-100k",
            "asset_id": "t1",
            "side": "BUY",
            "status": "LIVE",
            "type": "GTC",
            "original_size": "20",
            "price": "0.65",
            "size_matched": "5",
            "timestamp": 1700000000000,
        }
        order = order_from_polymarket(raw, venue="polymarket")
        assert isinstance(order, CanonicalOrder)
        assert order.venue == "polymarket"
        assert order.order_id == "order789"
        assert order.side == "buy"
        assert order.status == "new"  # LIVE -> new
        assert order.qty == Decimal("20")
        assert order.price == Decimal("0.65")
        assert order.filled_qty == Decimal("5")
        assert order.raw is raw

    def test_order_canceled_status(self):
        raw = {
            "id": "order_c",
            "market": "m",
            "asset_id": "t",
            "side": "sell",
            "status": "CANCELED",
            "type": "GTC",
            "original_size": "10",
            "price": "0.70",
            "size_matched": "0",
            "timestamp": 1700000001000,
        }
        order = order_from_polymarket(raw, venue="polymarket")
        assert order.status == "canceled"

    def test_order_matched_status(self):
        raw = {
            "id": "order_m",
            "market": "m",
            "asset_id": "t",
            "side": "buy",
            "status": "MATCHED",
            "type": "GTC",
            "original_size": "10",
            "price": "0.55",
            "size_matched": "10",
            "timestamp": 1700000002000,
        }
        order = order_from_polymarket(raw, venue="polymarket")
        assert order.status == "filled"

    def test_order_type_mapping(self):
        raw = {
            "id": "o1",
            "market": "m",
            "asset_id": "t",
            "side": "buy",
            "status": "LIVE",
            "type": "FOK",
            "original_size": "5",
            "price": "0.60",
            "size_matched": "0",
            "timestamp": 1700000003000,
        }
        order = order_from_polymarket(raw, venue="polymarket")
        assert order.order_type == "limit"
        assert order.tif == "fok"
