"""Tests for data collectors."""
from __future__ import annotations

import time
import threading
from datetime import datetime, timezone
from decimal import Decimal
from types import SimpleNamespace
from typing import Any, Callable, Dict, List

import pytest

from data.backends.base import Tick
from data.collectors.tick_collector import TickCollector
from data.collectors.funding_collector import FundingCollector


class FakeTickStore:
    """Mock TickStore that records write calls."""

    def __init__(self) -> None:
        self.written: List[tuple[str, list[Tick]]] = []

    def write_ticks(self, symbol: str, ticks: list[Tick]) -> None:
        self.written.append((symbol, list(ticks)))

    def read_ticks(
        self,
        symbol: str,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[Tick]:
        return []

    def count(self, symbol: str) -> int:
        return sum(len(t) for s, t in self.written if s == symbol)


def _make_trade_msg(symbol: str = "BTCUSDT", price: str = "42000", qty: str = "0.1") -> Dict[str, Any]:
    return {
        "s": symbol,
        "p": price,
        "q": qty,
        "T": int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000),
        "m": False,
        "a": "100001",
    }


class TestTickCollector:
    def test_buffer_fills_and_flushes_on_size(self) -> None:
        store = FakeTickStore()
        registered_cb: list[Callable] = []

        def mock_ws(cb: Callable) -> None:
            registered_cb.append(cb)

        collector = TickCollector(
            store=store,
            symbols=["BTCUSDT"],
            ws_callback=mock_ws,
            flush_size=10,
            flush_interval=999,  # won't trigger in this test
        )
        collector.start()

        assert len(registered_cb) == 1
        handler = registered_cb[0]

        for i in range(10):
            handler(_make_trade_msg(price=str(42000 + i)))

        # Buffer should have flushed at size=10
        collector.stop()

        total_ticks = sum(len(t) for _, t in store.written)
        assert total_ticks == 10

    def test_flush_on_stop(self) -> None:
        store = FakeTickStore()
        registered_cb: list[Callable] = []

        def mock_ws(cb: Callable) -> None:
            registered_cb.append(cb)

        collector = TickCollector(
            store=store,
            symbols=["BTCUSDT"],
            ws_callback=mock_ws,
            flush_size=1000,
            flush_interval=999,
        )
        collector.start()
        handler = registered_cb[0]

        for i in range(5):
            handler(_make_trade_msg())

        assert collector.buffer_size == 5
        collector.stop()

        total_ticks = sum(len(t) for _, t in store.written)
        assert total_ticks == 5

    def test_ignores_unsubscribed_symbols(self) -> None:
        store = FakeTickStore()
        registered_cb: list[Callable] = []

        def mock_ws(cb: Callable) -> None:
            registered_cb.append(cb)

        collector = TickCollector(
            store=store,
            symbols=["BTCUSDT"],
            ws_callback=mock_ws,
            flush_size=1000,
            flush_interval=999,
        )
        collector.start()
        handler = registered_cb[0]

        handler(_make_trade_msg(symbol="ETHUSDT"))
        assert collector.buffer_size == 0

        handler(_make_trade_msg(symbol="BTCUSDT"))
        assert collector.buffer_size == 1

        collector.stop()

    def test_is_running_property(self) -> None:
        store = FakeTickStore()
        collector = TickCollector(
            store=store,
            symbols=["BTCUSDT"],
            ws_callback=lambda cb: None,
            flush_size=100,
            flush_interval=999,
        )
        assert not collector.is_running
        collector.start()
        assert collector.is_running
        collector.stop()
        assert not collector.is_running


class TestFundingCollector:
    def test_fetch_and_store(self) -> None:
        fetched = []
        stored: list[list] = []

        sample_data = [{"symbol": "BTCUSDT", "rate": "0.0001"}]

        def mock_fetch() -> list[dict]:
            fetched.append(1)
            return sample_data

        def mock_store(data: list[dict]) -> None:
            stored.append(data)

        collector = FundingCollector(
            fetch_fn=mock_fetch,
            store_fn=mock_store,
            interval=999,  # long interval so only the initial run_now fires
        )
        collector.start()
        # Give the initial fetch a moment
        time.sleep(0.1)
        collector.stop()

        assert len(fetched) >= 1
        assert len(stored) >= 1
        assert stored[0] == sample_data

    def test_is_running(self) -> None:
        collector = FundingCollector(
            fetch_fn=lambda: [],
            store_fn=lambda d: None,
            interval=999,
        )
        assert not collector.is_running
        collector.start()
        assert collector.is_running
        collector.stop()
        assert not collector.is_running

    def test_last_active_ts_updated(self) -> None:
        collector = FundingCollector(
            fetch_fn=lambda: [{"rate": "0.01"}],
            store_fn=lambda d: None,
            interval=999,
        )
        assert collector.last_active_ts is None
        collector.start()
        time.sleep(0.1)
        collector.stop()
        assert collector.last_active_ts is not None

    def test_fetch_exception_handled(self) -> None:
        call_count = []

        def bad_fetch() -> list[dict]:
            call_count.append(1)
            raise RuntimeError("connection failed")

        collector = FundingCollector(
            fetch_fn=bad_fetch,
            store_fn=lambda d: None,
            interval=999,
        )
        collector.start()
        time.sleep(0.1)
        collector.stop()

        # Should not crash, and fetch was attempted
        assert len(call_count) >= 1
