# tests_unit/test_rest_kline_source.py
from __future__ import annotations

import json
from decimal import Decimal
from unittest.mock import patch, MagicMock

import pytest

from execution.adapters.binance.rest_kline_source import RestKlineSource


def _mock_kline_response():
    """Sample Binance /fapi/v1/klines response (2 klines)."""
    return json.dumps([
        [
            1704067200000, "42000.0", "42500.0", "41800.0", "42300.0",
            "100.5", 1704067259999, "4230000.0", 500, "50.2",
            "2115000.0", "0",
        ],
        [
            1704067260000, "42300.0", "42600.0", "42100.0", "42400.0",
            "80.3", 1704067319999, "3400000.0", 400, "40.1",
            "1700000.0", "0",
        ],
    ]).encode("utf-8")


class TestRestKlineSource:
    def test_fetch_recent_parses_klines(self):
        source = RestKlineSource(base_url="https://fapi.binance.com")

        mock_resp = MagicMock()
        mock_resp.read.return_value = _mock_kline_response()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("execution.adapters.binance.rest_kline_source.urlopen", return_value=mock_resp):
            result = source.fetch_recent("BTCUSDT", interval="1m", limit=2)

        assert len(result) == 2
        assert result[0]["o"] == "42000.0"
        assert result[0]["c"] == "42300.0"
        assert result[0]["x"] is True
        assert result[1]["t"] == 1704067260000

    def test_fetch_recent_handles_error(self):
        source = RestKlineSource(base_url="https://fapi.binance.com")

        with patch(
            "execution.adapters.binance.rest_kline_source.urlopen",
            side_effect=Exception("network error"),
        ):
            result = source.fetch_recent("BTCUSDT")

        assert result == []

    def test_fetch_as_events(self):
        source = RestKlineSource(base_url="https://fapi.binance.com")

        mock_resp = MagicMock()
        mock_resp.read.return_value = _mock_kline_response()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("execution.adapters.binance.rest_kline_source.urlopen", return_value=mock_resp):
            events = source.fetch_as_events("BTCUSDT", interval="1m", limit=2)

        assert len(events) == 2
        # MarketEvent should have symbol and OHLCV data
        ev = events[0]
        assert hasattr(ev, "symbol")
        assert str(ev.symbol).upper() == "BTCUSDT"

    def test_fetch_recent_skips_short_rows(self):
        source = RestKlineSource()
        bad_data = json.dumps([[1, 2, 3]]).encode("utf-8")

        mock_resp = MagicMock()
        mock_resp.read.return_value = bad_data
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("execution.adapters.binance.rest_kline_source.urlopen", return_value=mock_resp):
            result = source.fetch_recent("BTCUSDT")

        assert result == []

    def test_fetch_as_events_decimal_precision(self):
        source = RestKlineSource()

        mock_resp = MagicMock()
        mock_resp.read.return_value = _mock_kline_response()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("execution.adapters.binance.rest_kline_source.urlopen", return_value=mock_resp):
            events = source.fetch_as_events("BTCUSDT")

        assert events[0].close == Decimal("42300.0")
        assert events[0].volume == Decimal("100.5")

    def test_fetch_as_events_utc_timestamp(self):
        from datetime import datetime, timezone

        source = RestKlineSource()

        mock_resp = MagicMock()
        mock_resp.read.return_value = _mock_kline_response()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("execution.adapters.binance.rest_kline_source.urlopen", return_value=mock_resp):
            events = source.fetch_as_events("BTCUSDT")

        expected = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        assert events[0].ts == expected
