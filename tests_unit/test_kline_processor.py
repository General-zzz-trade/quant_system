"""Unit tests for KlineProcessor — Binance WS kline JSON to MarketEvent."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from decimal import Decimal

import pytest

from execution.adapters.binance.kline_processor import KlineProcessor


def _kline_msg(symbol="BTCUSDT", closed=True, **overrides):
    k = {
        "t": 1704067200000,  # 2024-01-01 00:00:00 UTC
        "o": "42000.50", "h": "42100.00", "l": "41900.00",
        "c": "42050.75", "v": "123.456",
        "x": closed,
    }
    k.update(overrides)
    return json.dumps({"e": "kline", "s": symbol, "k": k})


def _combined_msg(symbol="BTCUSDT", closed=True):
    k = {
        "t": 1704067200000, "o": "42000.50", "h": "42100.00",
        "l": "41900.00", "c": "42050.75", "v": "123.456", "x": closed,
    }
    return json.dumps({
        "stream": f"{symbol.lower()}@kline_1m",
        "data": {"e": "kline", "s": symbol, "k": k},
    })


class TestKlineProcessor:

    def test_closed_kline_produces_event(self):
        p = KlineProcessor()
        ev = p.process_raw(_kline_msg(closed=True))
        assert ev is not None
        assert ev.symbol == "BTCUSDT"

    def test_unclosed_kline_skipped(self):
        p = KlineProcessor(only_closed=True)
        ev = p.process_raw(_kline_msg(closed=False))
        assert ev is None

    def test_unclosed_kline_when_only_closed_false(self):
        p = KlineProcessor(only_closed=False)
        ev = p.process_raw(_kline_msg(closed=False))
        assert ev is not None

    def test_combined_stream_format(self):
        p = KlineProcessor()
        ev = p.process_raw(_combined_msg())
        assert ev is not None
        assert ev.symbol == "BTCUSDT"

    def test_invalid_json_returns_none(self):
        p = KlineProcessor()
        assert p.process_raw("not json") is None
        assert p.process_raw("") is None

    def test_non_kline_event_returns_none(self):
        msg = json.dumps({"e": "aggTrade", "s": "BTCUSDT"})
        p = KlineProcessor()
        assert p.process_raw(msg) is None

    def test_missing_kline_data_returns_none(self):
        msg = json.dumps({"e": "kline", "s": "BTCUSDT"})  # no "k"
        p = KlineProcessor()
        assert p.process_raw(msg) is None

    def test_decimal_precision(self):
        p = KlineProcessor()
        ev = p.process_raw(_kline_msg())
        assert ev.close == Decimal("42050.75")
        assert ev.open == Decimal("42000.50")
        assert ev.volume == Decimal("123.456")

    def test_timestamp_conversion(self):
        p = KlineProcessor()
        ev = p.process_raw(_kline_msg())
        expected = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        assert ev.ts == expected

    def test_missing_timestamp_returns_none(self):
        k = {"o": "1", "h": "1", "l": "1", "c": "1", "v": "1", "x": True}
        msg = json.dumps({"e": "kline", "s": "BTC", "k": k})
        p = KlineProcessor()
        assert p.process_raw(msg) is None
