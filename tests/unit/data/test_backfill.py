"""Tests for HistoricalBackfiller — mock-based verification of rate limiting and batching."""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, List, Sequence
from unittest.mock import MagicMock


from data.collectors.backfill import BackfillConfig, HistoricalBackfiller
from data.store import Bar


T0 = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)


def _make_raw_kline(ts: datetime, close: float = 100.0) -> list:
    """Create a raw kline list in exchange format."""
    ms = int(ts.timestamp() * 1000)
    return [ms, "100.0", "105.0", "95.0", str(close), "1000.0"]


class FakeBarStore:
    """In-memory bar store for testing."""

    def __init__(self) -> None:
        self.bars: dict[str, list[Bar]] = {}
        self.write_count = 0

    def read_bars(
        self,
        symbol: str,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> List[Bar]:
        return self.bars.get(symbol, [])

    def write_bars(self, symbol: str, bars: Sequence[Bar]) -> None:
        if symbol not in self.bars:
            self.bars[symbol] = []
        self.bars[symbol].extend(bars)
        self.write_count += 1


class TestHistoricalBackfiller:
    def test_backfill_writes_bars(self) -> None:
        klines = [_make_raw_kline(T0 + timedelta(hours=i)) for i in range(5)]
        fetch = MagicMock(side_effect=[klines, []])
        store = FakeBarStore()
        config = BackfillConfig(
            symbols=("BTCUSDT",),
            interval="1h",
            max_requests_per_minute=60000,  # effectively no throttle
            batch_size=1000,
        )
        backfiller = HistoricalBackfiller(
            fetch_klines=fetch, bar_store=store, config=config
        )
        count = backfiller.backfill("BTCUSDT", T0, T0 + timedelta(hours=4))
        assert count == 5
        assert len(store.bars["BTCUSDT"]) == 5

    def test_backfill_skips_existing(self) -> None:
        store = FakeBarStore()
        # Pre-populate hour 0
        existing_bar = Bar(
            ts=T0, open=Decimal("100"), high=Decimal("105"),
            low=Decimal("95"), close=Decimal("100"),
            volume=Decimal("1000"), symbol="BTCUSDT",
        )
        store.bars["BTCUSDT"] = [existing_bar]

        klines = [_make_raw_kline(T0 + timedelta(hours=i)) for i in range(3)]
        fetch = MagicMock(side_effect=[klines, []])
        config = BackfillConfig(
            symbols=("BTCUSDT",),
            interval="1h",
            max_requests_per_minute=60000,
            batch_size=1000,
        )
        backfiller = HistoricalBackfiller(
            fetch_klines=fetch, bar_store=store, config=config
        )
        count = backfiller.backfill("BTCUSDT", T0, T0 + timedelta(hours=2))
        # hour 0 already exists, so only hours 1 and 2 should be new
        assert count == 2

    def test_backfill_all(self) -> None:
        klines = [_make_raw_kline(T0)]
        fetch = MagicMock(side_effect=[klines, [], klines, []])
        store = FakeBarStore()
        config = BackfillConfig(
            symbols=("BTCUSDT", "ETHUSDT"),
            interval="1h",
            max_requests_per_minute=60000,
            batch_size=1000,
        )
        backfiller = HistoricalBackfiller(
            fetch_klines=fetch, bar_store=store, config=config
        )
        results = backfiller.backfill_all(T0, T0)
        assert "BTCUSDT" in results
        assert "ETHUSDT" in results

    def test_rate_limiting(self) -> None:
        """Verify that rate limiting introduces delay between requests."""
        klines = [_make_raw_kline(T0)]
        call_times: list[float] = []

        def tracking_fetch(*args: Any) -> list:
            call_times.append(time.monotonic())
            if len(call_times) <= 1:
                return klines
            return []

        store = FakeBarStore()
        config = BackfillConfig(
            symbols=("BTCUSDT",),
            interval="1h",
            max_requests_per_minute=600,  # 10 req/sec → 0.1s gap
            batch_size=1,
        )
        backfiller = HistoricalBackfiller(
            fetch_klines=tracking_fetch, bar_store=store, config=config
        )
        backfiller.backfill("BTCUSDT", T0, T0 + timedelta(hours=1))
        # With 600 rpm = 10/sec, gap should be ~0.1s
        if len(call_times) >= 2:
            gap = call_times[1] - call_times[0]
            assert gap >= 0.05  # at least 50ms between calls

    def test_empty_fetch_returns_zero(self) -> None:
        fetch = MagicMock(return_value=[])
        store = FakeBarStore()
        config = BackfillConfig(
            symbols=("BTCUSDT",),
            interval="1h",
            max_requests_per_minute=60000,
            batch_size=1000,
        )
        backfiller = HistoricalBackfiller(
            fetch_klines=fetch, bar_store=store, config=config
        )
        count = backfiller.backfill("BTCUSDT", T0, T0 + timedelta(hours=1))
        assert count == 0
