"""Tests for BinanceUmMarketStreamWsClient and KlineProcessor."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import List, Optional

import pytest

from execution.adapters.binance.kline_processor import KlineProcessor
from execution.adapters.binance.ws_market_stream_um import (
    BinanceUmMarketStreamWsClient,
    MarketStreamConfig,
)


# ============================================================
# WsTransport stub
# ============================================================


@dataclass
class FakeTransport:
    """In-memory WsTransport for tests."""
    messages: List[str] = field(default_factory=list)
    connected_url: Optional[str] = None
    closed: bool = False
    _idx: int = 0

    def connect(self, url: str) -> None:
        self.connected_url = url
        self.closed = False

    def recv(self, *, timeout_s: Optional[float] = None) -> str:
        if self._idx < len(self.messages):
            msg = self.messages[self._idx]
            self._idx += 1
            return msg
        return ""

    def close(self) -> None:
        self.closed = True
        self.connected_url = None


# ============================================================
# Sample Binance kline payloads
# ============================================================

def _make_kline_payload(
    *,
    symbol: str = "BTCUSDT",
    open: str = "50000.00",
    high: str = "50100.00",
    low: str = "49900.00",
    close: str = "50050.00",
    volume: str = "123.456",
    ts_ms: int = 1700000000000,
    closed: bool = True,
) -> str:
    return json.dumps({
        "stream": f"{symbol.lower()}@kline_1m",
        "data": {
            "e": "kline",
            "E": ts_ms + 59999,
            "s": symbol,
            "k": {
                "t": ts_ms,
                "T": ts_ms + 59999,
                "s": symbol,
                "i": "1m",
                "o": open,
                "h": high,
                "l": low,
                "c": close,
                "v": volume,
                "x": closed,
            },
        },
    })


# ============================================================
# KlineProcessor Tests
# ============================================================


class TestKlineProcessor:

    def test_closed_kline_produces_market_event(self) -> None:
        proc = KlineProcessor()
        raw = _make_kline_payload(closed=True)

        event = proc.process_raw(raw)

        assert event is not None
        assert event.symbol == "BTCUSDT"
        assert event.open == Decimal("50000.00")
        assert event.high == Decimal("50100.00")
        assert event.low == Decimal("49900.00")
        assert event.close == Decimal("50050.00")
        assert event.volume == Decimal("123.456")
        assert event.ts.tzinfo is not None

    def test_open_kline_returns_none_by_default(self) -> None:
        proc = KlineProcessor(only_closed=True)
        raw = _make_kline_payload(closed=False)

        assert proc.process_raw(raw) is None

    def test_open_kline_emitted_when_only_closed_false(self) -> None:
        proc = KlineProcessor(only_closed=False)
        raw = _make_kline_payload(closed=False)

        event = proc.process_raw(raw)
        assert event is not None
        assert event.symbol == "BTCUSDT"

    def test_invalid_json_returns_none(self) -> None:
        proc = KlineProcessor()
        assert proc.process_raw("not json") is None

    def test_non_kline_event_returns_none(self) -> None:
        proc = KlineProcessor()
        raw = json.dumps({"data": {"e": "aggTrade", "s": "BTCUSDT"}})
        assert proc.process_raw(raw) is None

    def test_missing_k_field_returns_none(self) -> None:
        proc = KlineProcessor()
        raw = json.dumps({"data": {"e": "kline", "s": "BTCUSDT"}})
        assert proc.process_raw(raw) is None

    def test_timestamp_is_utc(self) -> None:
        proc = KlineProcessor()
        raw = _make_kline_payload(ts_ms=1700000000000)

        event = proc.process_raw(raw)
        assert event is not None
        assert event.ts == datetime.fromtimestamp(1700000000, tz=timezone.utc)

    def test_direct_format_without_stream_wrapper(self) -> None:
        """Non-combined stream format (no 'stream' key)."""
        proc = KlineProcessor()
        raw = json.dumps({
            "e": "kline",
            "s": "ETHUSDT",
            "k": {
                "t": 1700000000000,
                "o": "2000.00",
                "h": "2010.00",
                "l": "1990.00",
                "c": "2005.00",
                "v": "500.0",
                "x": True,
            },
        })

        event = proc.process_raw(raw)
        assert event is not None
        assert event.symbol == "ETHUSDT"


# ============================================================
# BinanceUmMarketStreamWsClient Tests
# ============================================================


class TestBinanceUmMarketStreamWsClient:

    def test_connect_builds_correct_url(self) -> None:
        transport = FakeTransport()
        client = BinanceUmMarketStreamWsClient(
            transport=transport,
            processor=KlineProcessor(),
            streams=("btcusdt@kline_1m",),
        )

        url = client.connect()

        assert "streams=btcusdt@kline_1m" in url
        assert transport.connected_url == url

    def test_multi_stream_url(self) -> None:
        transport = FakeTransport()
        client = BinanceUmMarketStreamWsClient(
            transport=transport,
            processor=KlineProcessor(),
            streams=("btcusdt@kline_1m", "ethusdt@kline_1m"),
        )

        url = client.connect()
        assert "streams=btcusdt@kline_1m/ethusdt@kline_1m" in url

    def test_step_auto_connects(self) -> None:
        transport = FakeTransport()
        client = BinanceUmMarketStreamWsClient(
            transport=transport,
            processor=KlineProcessor(),
            streams=("btcusdt@kline_1m",),
        )

        client.step()
        assert transport.connected_url is not None

    def test_step_returns_market_event_for_closed_kline(self) -> None:
        raw = _make_kline_payload(closed=True)
        transport = FakeTransport(messages=[raw])
        client = BinanceUmMarketStreamWsClient(
            transport=transport,
            processor=KlineProcessor(),
            streams=("btcusdt@kline_1m",),
        )

        event = client.step()
        assert event is not None
        assert event.symbol == "BTCUSDT"

    def test_step_returns_none_for_timeout(self) -> None:
        transport = FakeTransport(messages=[])
        client = BinanceUmMarketStreamWsClient(
            transport=transport,
            processor=KlineProcessor(),
            streams=("btcusdt@kline_1m",),
        )

        assert client.step() is None

    def test_step_returns_none_for_open_kline(self) -> None:
        raw = _make_kline_payload(closed=False)
        transport = FakeTransport(messages=[raw])
        client = BinanceUmMarketStreamWsClient(
            transport=transport,
            processor=KlineProcessor(only_closed=True),
            streams=("btcusdt@kline_1m",),
        )

        assert client.step() is None

    def test_close_resets_connected_url(self) -> None:
        transport = FakeTransport()
        client = BinanceUmMarketStreamWsClient(
            transport=transport,
            processor=KlineProcessor(),
            streams=("btcusdt@kline_1m",),
        )
        client.connect()
        assert client._connected_url is not None

        client.close()
        assert client._connected_url is None
        assert transport.closed

    def test_multiple_steps_consume_messages(self) -> None:
        msgs = [
            _make_kline_payload(ts_ms=1700000000000, closed=True),
            _make_kline_payload(ts_ms=1700000060000, closed=True),
        ]
        transport = FakeTransport(messages=msgs)
        client = BinanceUmMarketStreamWsClient(
            transport=transport,
            processor=KlineProcessor(),
            streams=("btcusdt@kline_1m",),
        )

        e1 = client.step()
        e2 = client.step()
        e3 = client.step()

        assert e1 is not None
        assert e2 is not None
        assert e3 is None  # no more messages
        assert e1.ts != e2.ts
