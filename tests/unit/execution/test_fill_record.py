# tests/unit/execution/test_fill_record.py
"""Tests for unified fill recording — replaces 3 ad-hoc dict builders."""
from decimal import Decimal
from types import SimpleNamespace


from execution.models.fills import CanonicalFill, fill_to_record


class TestCanonicalFillToRecord:
    def test_all_fields_present(self):
        fill = CanonicalFill(
            venue="binance", symbol="BTCUSDT",
            order_id="1", trade_id="2", fill_id="binance:BTCUSDT:2",
            side="buy", qty=Decimal("0.1"), price=Decimal("70000"),
            fee=Decimal("0.07"), fee_asset="USDT", liquidity="taker",
            ts_ms=1700000000000, payload_digest="abc123",
        )
        rec = fill.to_record()
        assert rec["venue"] == "binance"
        assert rec["symbol"] == "BTCUSDT"
        assert rec["order_id"] == "1"
        assert rec["trade_id"] == "2"
        assert rec["fill_id"] == "binance:BTCUSDT:2"
        assert rec["side"] == "buy"
        assert rec["qty"] == "0.1"
        assert rec["price"] == "70000"
        assert rec["fee"] == "0.07"
        assert rec["fee_asset"] == "USDT"
        assert rec["liquidity"] == "taker"
        assert rec["ts_ms"] == "1700000000000"
        assert rec["payload_digest"] == "abc123"

    def test_no_data_loss(self):
        """Verify to_record preserves all fields — unlike old ad-hoc dicts."""
        fill = CanonicalFill(
            venue="ib", symbol="AAPL",
            order_id="99", trade_id="t1", fill_id="ib:t1",
            side="sell", qty=Decimal("100"), price=Decimal("155.50"),
            fee=Decimal("1.0"), fee_asset="USD", liquidity="maker",
            ts_ms=1700000000000,
        )
        rec = fill.to_record()
        # Old ad-hoc dict only had 5 fields; new record has 13
        assert len(rec) == 13


class TestFillToRecord:
    def test_canonical_fill_uses_to_record(self):
        fill = CanonicalFill(
            venue="binance", symbol="BTCUSDT",
            order_id="1", trade_id="2", fill_id="f:2",
            side="buy", qty=Decimal("0.1"), price=Decimal("70000"),
        )
        rec = fill_to_record(fill)
        assert rec["venue"] == "binance"
        assert rec["qty"] == "0.1"

    def test_duck_typed_fill(self):
        """Works with SimpleNamespace or any fill-like object."""
        fill = SimpleNamespace(
            venue="binance", symbol="ETHUSDT",
            order_id="5", trade_id="10", fill_id="f:10",
            side="sell", qty=Decimal("1.0"), price=Decimal("3500"),
            fee=Decimal("0.35"), ts_ms=1700000000000,
        )
        rec = fill_to_record(fill)
        assert rec["symbol"] == "ETHUSDT"
        assert rec["side"] == "sell"
        assert rec["fee"] == "0.35"

    def test_missing_fields_default_to_empty(self):
        fill = SimpleNamespace(symbol="BTCUSDT", side="buy")
        rec = fill_to_record(fill)
        assert rec["symbol"] == "BTCUSDT"
        assert rec["venue"] == ""
        assert rec["qty"] == ""
