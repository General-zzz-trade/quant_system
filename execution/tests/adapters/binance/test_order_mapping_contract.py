# execution/tests/adapters/binance/test_order_mapping_contract.py
from __future__ import annotations

from decimal import Decimal

from execution.adapters.binance.mapper_order import BinanceOrderMapper


def test_binance_spot_execution_report_new_maps_to_canonical_order() -> None:
    raw = {
        "e": "executionReport",
        "E": 1700000000999,
        "s": "btcusdt",
        "S": "BUY",
        "i": 111,
        "c": "cli-001",
        "X": "NEW",
        "o": "LIMIT",
        "f": "GTC",
        "q": "0.10",
        "p": "43000",
        "z": "0",
        "T": 1700000001001,
    }
    o = BinanceOrderMapper().map_order(raw)

    assert o.venue == "binance"
    assert o.symbol == "BTCUSDT"
    assert o.order_id == "111"
    assert o.client_order_id == "cli-001"
    assert o.status == "new"
    assert o.side == "buy"
    assert o.order_type == "limit"
    assert o.tif == "gtc"
    assert o.qty == Decimal("0.10")
    assert o.price == Decimal("43000")
    assert o.filled_qty == Decimal("0")
    assert o.order_key == "binance:BTCUSDT:order:111"
    assert o.payload_digest and len(o.payload_digest) == 64


def test_binance_futures_order_trade_update_partially_filled_maps() -> None:
    raw = {
        "e": "ORDER_TRADE_UPDATE",
        "E": 1700000002000,
        "o": {
            "s": "ETHUSDT",
            "S": "SELL",
            "i": 999,
            "c": "cli-002",
            "X": "PARTIALLY_FILLED",
            "o": "MARKET",
            "f": "IOC",
            "q": "2",
            "p": "0",
            "z": "0.4",
            "ap": "2301.25",
            "T": 1700000002123,
        },
    }
    o = BinanceOrderMapper().map_order(raw)

    assert o.symbol == "ETHUSDT"
    assert o.side == "sell"
    assert o.status == "partially_filled"
    assert o.order_type == "market"
    assert o.tif == "ioc"
    assert o.qty == Decimal("2")
    assert o.filled_qty == Decimal("0.4")
    assert o.avg_price == Decimal("2301.25")
