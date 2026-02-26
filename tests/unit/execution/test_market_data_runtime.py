# tests/unit/execution/test_market_data_runtime.py
"""Tests for BinanceMarketDataRuntime."""
from __future__ import annotations

import threading
import time
from types import SimpleNamespace
from typing import Any, List, Optional
from unittest.mock import MagicMock

import pytest

from execution.adapters.binance.market_data_runtime import BinanceMarketDataRuntime


# ── Stubs ────────────────────────────────────────────────────

class _FakeWsClient:
    """Simulates BinanceUmMarketStreamWsClient with controllable step()."""

    def __init__(self, events: List[Any] | None = None):
        self._events = list(events or [])
        self._idx = 0
        self._closed = False

    def step(self) -> Optional[Any]:
        if self._closed or self._idx >= len(self._events):
            time.sleep(0.01)  # avoid tight loop
            return None
        ev = self._events[self._idx]
        self._idx += 1
        return ev

    def close(self) -> None:
        self._closed = True

    def connect(self) -> str:
        return "wss://fake"


def _market_event(symbol: str = "BTCUSDT") -> SimpleNamespace:
    return SimpleNamespace(
        event_type="market",
        symbol=symbol,
        close=40000.0,
    )


# ── Tests ────────────────────────────────────────────────────

class TestSubscribeUnsubscribe:
    def test_subscribe_adds_handler(self):
        ws = _FakeWsClient()
        runtime = BinanceMarketDataRuntime(ws_client=ws)
        handler = MagicMock()
        runtime.subscribe(handler)
        assert handler in runtime._handlers

    def test_subscribe_no_duplicates(self):
        ws = _FakeWsClient()
        runtime = BinanceMarketDataRuntime(ws_client=ws)
        handler = MagicMock()
        runtime.subscribe(handler)
        runtime.subscribe(handler)
        assert len(runtime._handlers) == 1

    def test_unsubscribe_removes_handler(self):
        ws = _FakeWsClient()
        runtime = BinanceMarketDataRuntime(ws_client=ws)
        handler = MagicMock()
        runtime.subscribe(handler)
        runtime.unsubscribe(handler)
        assert handler not in runtime._handlers

    def test_unsubscribe_nonexistent_no_error(self):
        ws = _FakeWsClient()
        runtime = BinanceMarketDataRuntime(ws_client=ws)
        runtime.unsubscribe(MagicMock())  # should not raise


class TestEventDelivery:
    def test_events_delivered_to_handler(self):
        ev1 = _market_event("BTCUSDT")
        ev2 = _market_event("ETHUSDT")
        ws = _FakeWsClient(events=[ev1, ev2])
        runtime = BinanceMarketDataRuntime(ws_client=ws)

        received: List[Any] = []
        runtime.subscribe(lambda e: received.append(e))
        runtime.start()

        # Wait for delivery
        deadline = time.monotonic() + 2.0
        while len(received) < 2 and time.monotonic() < deadline:
            time.sleep(0.01)

        runtime.stop()
        assert len(received) == 2
        assert received[0].symbol == "BTCUSDT"
        assert received[1].symbol == "ETHUSDT"

    def test_multiple_handlers(self):
        ev = _market_event()
        ws = _FakeWsClient(events=[ev])
        runtime = BinanceMarketDataRuntime(ws_client=ws)

        r1: List[Any] = []
        r2: List[Any] = []
        runtime.subscribe(lambda e: r1.append(e))
        runtime.subscribe(lambda e: r2.append(e))
        runtime.start()

        deadline = time.monotonic() + 2.0
        while (len(r1) < 1 or len(r2) < 1) and time.monotonic() < deadline:
            time.sleep(0.01)

        runtime.stop()
        assert len(r1) >= 1
        assert len(r2) >= 1

    def test_none_events_not_delivered(self):
        ws = _FakeWsClient(events=[])  # step() always returns None
        runtime = BinanceMarketDataRuntime(ws_client=ws)

        received: List[Any] = []
        runtime.subscribe(lambda e: received.append(e))
        runtime.start()
        time.sleep(0.1)
        runtime.stop()
        assert len(received) == 0


class TestLifecycle:
    def test_start_stop(self):
        ws = _FakeWsClient()
        runtime = BinanceMarketDataRuntime(ws_client=ws)
        runtime.start()
        assert runtime._running is True
        runtime.stop()
        assert runtime._running is False

    def test_double_start_idempotent(self):
        ws = _FakeWsClient()
        runtime = BinanceMarketDataRuntime(ws_client=ws)
        runtime.start()
        runtime.start()  # should not create second thread
        runtime.stop()

    def test_handler_error_does_not_crash_runtime(self):
        ev = _market_event()
        ws = _FakeWsClient(events=[ev, _market_event("ETH")])
        runtime = BinanceMarketDataRuntime(ws_client=ws)

        good_received: List[Any] = []

        def bad_handler(e: Any) -> None:
            raise ValueError("boom")

        runtime.subscribe(bad_handler)
        runtime.subscribe(lambda e: good_received.append(e))
        runtime.start()

        deadline = time.monotonic() + 2.0
        while len(good_received) < 1 and time.monotonic() < deadline:
            time.sleep(0.01)

        runtime.stop()
        assert len(good_received) >= 1  # runtime survived the error
