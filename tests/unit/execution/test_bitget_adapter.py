# tests/unit/execution/test_bitget_adapter.py
"""Tests for Bitget WebSocket infrastructure, kline processor, market data runtime,
and full venue client."""
from __future__ import annotations

import json
import time
from decimal import Decimal
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Sequence
from unittest.mock import MagicMock, patch

import pytest

from execution.adapters.bitget.kline_processor import BitgetKlineProcessor, BitgetKlineRaw
from execution.adapters.bitget.market_data_runtime import BitgetMarketDataRuntime
from execution.adapters.bitget.ws_client import BitgetWsConfig, BitgetWsMarketStreamClient


# ── BitgetWsConfig Tests ─────────────────────────────────────


class TestBitgetWsConfig:
    def test_defaults(self):
        cfg = BitgetWsConfig()
        assert cfg.ws_url == "wss://ws.bitget.com/v2/ws/public"
        assert cfg.ping_interval == 25.0
        assert cfg.reconnect_delay == 3.0
        assert cfg.inst_type == "USDT-FUTURES"

    def test_custom_values(self):
        cfg = BitgetWsConfig(
            ws_url="wss://custom.url/ws",
            ping_interval=15.0,
            reconnect_delay=5.0,
            inst_type="COIN-FUTURES",
        )
        assert cfg.ws_url == "wss://custom.url/ws"
        assert cfg.ping_interval == 15.0
        assert cfg.inst_type == "COIN-FUTURES"

    def test_frozen(self):
        cfg = BitgetWsConfig()
        with pytest.raises(AttributeError):
            cfg.ws_url = "new_url"  # type: ignore


# ── BitgetKlineRaw Tests ─────────────────────────────────────


class TestBitgetKlineRaw:
    def test_create(self):
        raw = BitgetKlineRaw(
            symbol="BTCUSDT",
            ts_ms=1700000000000,
            open=42000.0,
            high=42500.0,
            low=41500.0,
            close=42200.0,
            volume=1234.5,
        )
        assert raw.symbol == "BTCUSDT"
        assert raw.ts_ms == 1700000000000
        assert raw.open == 42000.0
        assert raw.volume == 1234.5

    def test_frozen(self):
        raw = BitgetKlineRaw(
            symbol="BTCUSDT", ts_ms=0, open=1.0, high=1.0,
            low=1.0, close=1.0, volume=0.0,
        )
        with pytest.raises(AttributeError):
            raw.symbol = "ETHUSDT"  # type: ignore


# ── BitgetKlineProcessor Tests ───────────────────────────────


class TestBitgetKlineProcessor:
    def test_process_valid_kline(self):
        proc = BitgetKlineProcessor(source="test.kline")
        raw = BitgetKlineRaw(
            symbol="BTCUSDT",
            ts_ms=1700000000000,
            open=42000.0,
            high=42500.0,
            low=41500.0,
            close=42200.0,
            volume=1234.5,
        )
        event = proc.process(raw)
        assert event is not None
        assert event.symbol == "BTCUSDT"
        assert event.open == Decimal("42000.0")
        assert event.high == Decimal("42500.0")
        assert event.low == Decimal("41500.0")
        assert event.close == Decimal("42200.0")
        assert event.volume == Decimal("1234.5")

    def test_process_source_tag(self):
        proc = BitgetKlineProcessor(source="bitget.ws.candle")
        raw = BitgetKlineRaw(
            symbol="ETHUSDT", ts_ms=1700000000000,
            open=2000.0, high=2100.0, low=1950.0, close=2050.0, volume=500.0,
        )
        event = proc.process(raw)
        assert event is not None
        assert event.header.source == "bitget.ws.candle"

    def test_process_invalid_ohlc_returns_none(self):
        proc = BitgetKlineProcessor()
        raw = BitgetKlineRaw(
            symbol="BTCUSDT", ts_ms=1700000000000,
            open=42000.0, high=41000.0, low=41500.0, close=42200.0, volume=100.0,
        )
        event = proc.process(raw)
        assert event is None

    def test_default_source(self):
        proc = BitgetKlineProcessor()
        assert proc.source == "bitget.ws.kline"


# ── BitgetWsMarketStreamClient Tests ─────────────────────────


class TestBitgetWsMarketStreamClient:
    def test_init_defaults(self):
        client = BitgetWsMarketStreamClient(symbols=["BTCUSDT", "ETHUSDT"])
        assert client.channel == "candle1m"
        assert len(client.symbols) == 2

    def test_on_message_valid_kline(self):
        received: List[BitgetKlineRaw] = []
        client = BitgetWsMarketStreamClient(
            symbols=["BTCUSDT"],
            on_kline=lambda raw: received.append(raw),
        )
        msg = json.dumps({
            "arg": {"channel": "candle1m", "instId": "BTCUSDT", "instType": "USDT-FUTURES"},
            "data": [
                ["1700000000000", "42000", "42500", "41500", "42200", "1234.5", "51870000"],
            ],
        })
        client._on_message(None, msg)
        assert len(received) == 1
        assert received[0].symbol == "BTCUSDT"
        assert received[0].open == 42000.0
        assert received[0].close == 42200.0

    def test_on_message_multiple_candles(self):
        received: List[BitgetKlineRaw] = []
        client = BitgetWsMarketStreamClient(
            symbols=["BTCUSDT"],
            on_kline=lambda raw: received.append(raw),
        )
        msg = json.dumps({
            "arg": {"channel": "candle1m", "instId": "ETHUSDT"},
            "data": [
                ["1700000000000", "2000", "2100", "1950", "2050", "500", "1000000"],
                ["1700000060000", "2050", "2080", "2020", "2060", "300", "615000"],
            ],
        })
        client._on_message(None, msg)
        assert len(received) == 2
        assert received[1].ts_ms == 1700000060000

    def test_on_message_ignores_subscription_confirmation(self):
        received: List[BitgetKlineRaw] = []
        client = BitgetWsMarketStreamClient(
            symbols=["BTCUSDT"],
            on_kline=lambda raw: received.append(raw),
        )
        msg = json.dumps({"event": "subscribe", "arg": {"channel": "candle1m"}})
        client._on_message(None, msg)
        assert len(received) == 0

    def test_on_message_ignores_non_candle_channel(self):
        received: List[BitgetKlineRaw] = []
        client = BitgetWsMarketStreamClient(
            symbols=["BTCUSDT"],
            on_kline=lambda raw: received.append(raw),
        )
        msg = json.dumps({
            "arg": {"channel": "ticker", "instId": "BTCUSDT"},
            "data": [{"last": "42000"}],
        })
        client._on_message(None, msg)
        assert len(received) == 0

    def test_on_message_bad_json(self):
        client = BitgetWsMarketStreamClient(
            symbols=["BTCUSDT"],
            on_kline=MagicMock(),
        )
        client._on_message(None, "not json")
        client.on_kline.assert_not_called()

    def test_on_message_short_candle_data_skipped(self):
        received: List[BitgetKlineRaw] = []
        client = BitgetWsMarketStreamClient(
            symbols=["BTCUSDT"],
            on_kline=lambda raw: received.append(raw),
        )
        msg = json.dumps({
            "arg": {"channel": "candle1m", "instId": "BTCUSDT"},
            "data": [
                ["1700000000000", "42000"],  # too short
            ],
        })
        client._on_message(None, msg)
        assert len(received) == 0

    def test_on_open_sends_subscription(self):
        client = BitgetWsMarketStreamClient(
            symbols=["BTCUSDT", "ETHUSDT"],
            channel="candle5m",
            cfg=BitgetWsConfig(inst_type="USDT-FUTURES"),
        )
        mock_ws = MagicMock()
        client._on_open(mock_ws)
        mock_ws.send.assert_called_once()
        sent = json.loads(mock_ws.send.call_args[0][0])
        assert sent["op"] == "subscribe"
        assert len(sent["args"]) == 2
        assert sent["args"][0]["channel"] == "candle5m"
        assert sent["args"][0]["instId"] == "BTCUSDT"
        assert sent["args"][1]["instId"] == "ETHUSDT"
        assert sent["args"][0]["instType"] == "USDT-FUTURES"

    def test_on_kline_callback_error_does_not_crash(self):
        def bad_callback(raw: Any) -> None:
            raise ValueError("boom")

        client = BitgetWsMarketStreamClient(
            symbols=["BTCUSDT"],
            on_kline=bad_callback,
        )
        msg = json.dumps({
            "arg": {"channel": "candle1m", "instId": "BTCUSDT"},
            "data": [
                ["1700000000000", "42000", "42500", "41500", "42200", "100", "4200000"],
            ],
        })
        # Should not raise
        client._on_message(None, msg)


# ── BitgetMarketDataRuntime Tests ─────────────────────────────


class _FakeBitgetWsClient:
    """Stub for BitgetWsMarketStreamClient."""

    def __init__(self) -> None:
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True


class TestBitgetMarketDataRuntime:
    def test_subscribe_adds_handler(self):
        ws = _FakeBitgetWsClient()
        runtime = BitgetMarketDataRuntime(ws_client=ws)  # type: ignore
        handler = MagicMock()
        runtime.subscribe(handler)
        assert handler in runtime._handlers

    def test_subscribe_no_duplicates(self):
        ws = _FakeBitgetWsClient()
        runtime = BitgetMarketDataRuntime(ws_client=ws)  # type: ignore
        handler = MagicMock()
        runtime.subscribe(handler)
        runtime.subscribe(handler)
        assert len(runtime._handlers) == 1

    def test_unsubscribe_removes_handler(self):
        ws = _FakeBitgetWsClient()
        runtime = BitgetMarketDataRuntime(ws_client=ws)  # type: ignore
        handler = MagicMock()
        runtime.subscribe(handler)
        runtime.unsubscribe(handler)
        assert handler not in runtime._handlers

    def test_unsubscribe_nonexistent_no_error(self):
        ws = _FakeBitgetWsClient()
        runtime = BitgetMarketDataRuntime(ws_client=ws)  # type: ignore
        runtime.unsubscribe(MagicMock())

    def test_start_stop_lifecycle(self):
        ws = _FakeBitgetWsClient()
        runtime = BitgetMarketDataRuntime(ws_client=ws)  # type: ignore
        runtime.start()
        assert runtime._running is True
        assert ws.started is True
        runtime.stop()
        assert runtime._running is False
        assert ws.stopped is True

    def test_double_start_idempotent(self):
        ws = _FakeBitgetWsClient()
        runtime = BitgetMarketDataRuntime(ws_client=ws)  # type: ignore
        runtime.start()
        runtime.start()
        runtime.stop()

    def test_enqueue_and_dispatch(self):
        ws = _FakeBitgetWsClient()
        runtime = BitgetMarketDataRuntime(ws_client=ws)  # type: ignore

        received: List[Any] = []
        runtime.subscribe(lambda e: received.append(e))
        runtime.start()

        ev = SimpleNamespace(symbol="BTCUSDT", close=42000.0)
        runtime.enqueue(ev)

        deadline = time.monotonic() + 2.0
        while len(received) < 1 and time.monotonic() < deadline:
            time.sleep(0.01)

        runtime.stop()
        assert len(received) == 1
        assert received[0].symbol == "BTCUSDT"

    def test_multiple_events_dispatched_in_order(self):
        ws = _FakeBitgetWsClient()
        runtime = BitgetMarketDataRuntime(ws_client=ws)  # type: ignore

        received: List[Any] = []
        runtime.subscribe(lambda e: received.append(e))
        runtime.start()

        for i in range(5):
            runtime.enqueue(SimpleNamespace(idx=i))

        deadline = time.monotonic() + 2.0
        while len(received) < 5 and time.monotonic() < deadline:
            time.sleep(0.01)

        runtime.stop()
        assert [r.idx for r in received] == [0, 1, 2, 3, 4]

    def test_handler_error_does_not_crash_runtime(self):
        ws = _FakeBitgetWsClient()
        runtime = BitgetMarketDataRuntime(ws_client=ws)  # type: ignore

        good: List[Any] = []

        def bad(e: Any) -> None:
            raise ValueError("boom")

        runtime.subscribe(bad)
        runtime.subscribe(lambda e: good.append(e))
        runtime.start()

        runtime.enqueue(SimpleNamespace(x=1))
        runtime.enqueue(SimpleNamespace(x=2))

        deadline = time.monotonic() + 2.0
        while len(good) < 2 and time.monotonic() < deadline:
            time.sleep(0.01)

        runtime.stop()
        assert len(good) >= 1


# ── BitgetFuturesFullVenueClient Tests ────────────────────────


class TestBitgetFuturesFullVenueClient:
    @pytest.fixture
    def mock_rest_client(self):
        client = MagicMock()
        client.get_contracts.return_value = [
            {
                "symbol": "BTCUSDT",
                "baseCoin": "BTC",
                "quoteCoin": "USDT",
                "pricePrecision": "2",
                "volumePlace": "3",
                "priceEndStep": "0.01",
                "sizeMultiplier": "0.001",
                "minTradeNum": "0.001",
                "minTradeUSDT": "5",
            },
        ]
        client.get_accounts.return_value = [
            {
                "marginCoin": "USDT",
                "available": "10000",
                "locked": "500",
                "accountEquity": "10500",
            },
        ]
        client.get_positions.return_value = [
            {
                "symbol": "BTCUSDT",
                "total": "0.1",
                "holdSide": "long",
                "openPriceAvg": "42000",
                "unrealizedPL": "100",
                "leverage": "10",
                "marginMode": "crossed",
            },
        ]
        client.get_pending_orders.return_value = [
            {
                "orderId": "123456",
                "symbol": "BTCUSDT",
                "side": "buy",
                "status": "new",
                "orderType": "limit",
                "force": "gtc",
                "size": "0.01",
                "price": "41000",
                "baseVolume": "0",
                "priceAvg": "0",
                "cTime": "1700000000000",
            },
        ]
        client.get_fills.return_value = [
            {
                "tradeId": "t001",
                "orderId": "123456",
                "symbol": "BTCUSDT",
                "side": "buy",
                "baseVolume": "0.01",
                "price": "41000",
                "fee": "0.5",
                "feeCoin": "USDT",
                "cTime": "1700000000000",
                "tradeScope": "taker",
            },
        ]
        return client

    @pytest.fixture
    def mock_order_gateway(self):
        gw = MagicMock()
        gw.submit_order.return_value = {"orderId": "789"}
        gw.cancel_order.return_value = {"orderId": "789", "status": "cancelled"}
        return gw

    @pytest.fixture
    def venue_client(self, mock_rest_client, mock_order_gateway):
        from execution.adapters.bitget.venue_client_futures import BitgetFuturesFullVenueClient
        return BitgetFuturesFullVenueClient(
            rest_client=mock_rest_client,
            order_gateway=mock_order_gateway,
        )

    def test_list_instruments(self, venue_client):
        instruments = venue_client.list_instruments()
        assert len(instruments) == 1
        assert instruments[0].symbol == "BTCUSDT"
        assert instruments[0].venue == "bitget"
        assert instruments[0].base_asset == "BTC"

    def test_get_balances(self, venue_client):
        snapshot = venue_client.get_balances()
        assert snapshot.venue == "bitget"
        assert len(snapshot.balances) == 1
        bal = snapshot.balances[0]
        assert bal.asset == "USDT"
        assert bal.free == Decimal("10000")
        assert bal.locked == Decimal("500")

    def test_get_positions(self, venue_client):
        positions = venue_client.get_positions()
        assert len(positions) == 1
        pos = positions[0]
        assert pos.symbol == "BTCUSDT"
        assert pos.qty == Decimal("0.1")
        assert pos.entry_price == Decimal("42000")

    def test_get_open_orders(self, venue_client):
        orders = venue_client.get_open_orders()
        assert len(orders) == 1
        order = orders[0]
        assert order.symbol == "BTCUSDT"
        assert order.side == "buy"
        assert order.order_id == "123456"
        assert order.status == "new"

    def test_get_recent_fills(self, venue_client):
        fills = venue_client.get_recent_fills(symbol="BTCUSDT")
        assert len(fills) == 1
        fill = fills[0]
        assert fill.symbol == "BTCUSDT"
        assert fill.trade_id == "t001"
        assert fill.side == "buy"
        assert fill.qty == Decimal("0.01")

    def test_get_recent_fills_no_symbol_returns_empty(self, venue_client):
        fills = venue_client.get_recent_fills()
        assert fills == ()

    def test_get_recent_fills_since_ms_filter(self, venue_client):
        fills = venue_client.get_recent_fills(symbol="BTCUSDT", since_ms=1700000000001)
        assert len(fills) == 0

    def test_submit_order_delegates(self, venue_client, mock_order_gateway):
        cmd = SimpleNamespace(symbol="BTCUSDT", side="buy", qty=0.01)
        result = venue_client.submit_order(cmd)
        mock_order_gateway.submit_order.assert_called_once_with(cmd)
        assert result["orderId"] == "789"

    def test_cancel_order_delegates(self, venue_client, mock_order_gateway):
        cmd = SimpleNamespace(symbol="BTCUSDT", order_id="789")
        result = venue_client.cancel_order(cmd)
        mock_order_gateway.cancel_order.assert_called_once_with(cmd)

    def test_submit_order_retryable_error(self, venue_client, mock_order_gateway):
        from execution.adapters.bitget.rest import BitgetRetryableError
        from execution.bridge.execution_bridge import RetryableVenueError
        mock_order_gateway.submit_order.side_effect = BitgetRetryableError("rate limit")
        with pytest.raises(RetryableVenueError):
            venue_client.submit_order(SimpleNamespace())

    def test_submit_order_non_retryable_error(self, venue_client, mock_order_gateway):
        from execution.adapters.bitget.rest import BitgetNonRetryableError
        from execution.bridge.execution_bridge import NonRetryableVenueError
        mock_order_gateway.submit_order.side_effect = BitgetNonRetryableError("bad param")
        with pytest.raises(NonRetryableVenueError):
            venue_client.submit_order(SimpleNamespace())

    def test_venue_field(self, venue_client):
        assert venue_client.venue == "bitget"
