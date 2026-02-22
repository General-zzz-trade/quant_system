# tests/adapters/binance/test_fill_mapping_contract.py
from __future__ import annotations

from decimal import Decimal

from execution.adapters.binance.mapper_fill import BinanceFillMapper


def test_binance_futures_ws_order_trade_update_maps_to_canonical_fill() -> None:
    raw = {
        "e": "ORDER_TRADE_UPDATE",
        "E": 1700000000000,
        "o": {
            "s": "btcusdt",
            "S": "BUY",
            "i": 987654321,
            "c": "x-001",
            "t": 123456789,
            "l": "0.001",
            "L": "43000.0",
            "n": "0.12",
            "N": "USDT",
            "T": 1700000000123,
            "m": False,
        },
    }
    f = BinanceFillMapper().map_fill(raw)

    assert f.venue == "binance"
    assert f.symbol == "BTCUSDT"
    assert f.side == "buy"
    assert f.order_id == "987654321"
    assert f.trade_id == "123456789"
    assert f.fill_id == "binance:BTCUSDT:123456789"
    assert f.qty == Decimal("0.001")
    assert f.price == Decimal("43000.0")
    assert f.fee == Decimal("0.12")
    assert f.fee_asset == "USDT"
    assert f.ts_ms == 1700000000123
    assert f.payload_digest and len(f.payload_digest) == 64


def test_binance_spot_ws_execution_report_maps_to_canonical_fill() -> None:
    raw = {
        "e": "executionReport",
        "E": 1700000000999,
        "s": "ETHUSDT",
        "S": "SELL",
        "i": 111,
        "c": "x-002",
        "t": 222,
        "l": "0.5",
        "L": "2300.5",
        "n": "0.01",
        "N": "USDT",
        "T": 1700000001001,
        "m": True,
    }
    f = BinanceFillMapper().map_fill(raw)
    assert f.symbol == "ETHUSDT"
    assert f.side == "sell"
    assert f.liquidity == "maker"


def test_binance_rest_trade_maps_to_canonical_fill() -> None:
    raw = {
        "symbol": "BNBUSDT",
        "id": 9001,
        "orderId": 8001,
        "side": "BUY",
        "qty": "2",
        "price": "300",
        "commission": "0.02",
        "commissionAsset": "USDT",
        "time": 1700000002000,
        "isMaker": False,
    }
    f = BinanceFillMapper().map_fill(raw)
    assert f.fill_id == "binance:BNBUSDT:9001"
    assert f.qty == Decimal("2")
    assert f.price == Decimal("300")
    assert f.liquidity == "taker"
