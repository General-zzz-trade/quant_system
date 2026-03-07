"""Tests for data.store.TimeSeriesStore."""
from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from data.store import Bar, TimeSeriesStore


def _bar(hour: int = 0, close: str = "100.00", symbol: str = "BTCUSDT") -> Bar:
    return Bar(
        ts=datetime(2024, 1, 1, hour, 0, 0, tzinfo=timezone.utc),
        open=Decimal("99.00"),
        high=Decimal("101.00"),
        low=Decimal("98.00"),
        close=Decimal(close),
        volume=Decimal("1000"),
        symbol=symbol,
        exchange="test",
    )


class TestTimeSeriesStore:

    def test_write_and_read_roundtrip(self, tmp_path: Path) -> None:
        store = TimeSeriesStore(tmp_path / "ts")
        bars = [_bar(hour=i) for i in range(5)]

        store.write_bars("BTCUSDT", bars)
        result = store.read_bars("BTCUSDT")

        assert len(result) == 5
        assert result[0].close == Decimal("100.00")
        assert result[0].symbol == "BTCUSDT"

    def test_read_empty_returns_empty(self, tmp_path: Path) -> None:
        store = TimeSeriesStore(tmp_path / "ts")
        assert store.read_bars("NONEXIST") == []

    def test_write_empty_raises(self, tmp_path: Path) -> None:
        store = TimeSeriesStore(tmp_path / "ts")
        with pytest.raises(ValueError, match="empty"):
            store.write_bars("X", [])

    def test_time_range_filter(self, tmp_path: Path) -> None:
        store = TimeSeriesStore(tmp_path / "ts")
        bars = [_bar(hour=i) for i in range(10)]
        store.write_bars("BTCUSDT", bars)

        start = datetime(2024, 1, 1, 3, 0, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 1, 6, 0, 0, tzinfo=timezone.utc)
        result = store.read_bars("BTCUSDT", start=start, end=end)

        assert len(result) == 4  # hours 3, 4, 5, 6

    def test_deduplicates_on_append(self, tmp_path: Path) -> None:
        store = TimeSeriesStore(tmp_path / "ts")
        store.write_bars("BTCUSDT", [_bar(hour=0, close="100"), _bar(hour=1, close="101")])
        store.write_bars("BTCUSDT", [_bar(hour=1, close="102"), _bar(hour=2, close="103")])

        result = store.read_bars("BTCUSDT")
        assert len(result) == 3
        # hour=1 should be the updated value
        hour1 = [b for b in result if b.ts.hour == 1][0]
        assert hour1.close == Decimal("102")

    def test_list_symbols(self, tmp_path: Path) -> None:
        store = TimeSeriesStore(tmp_path / "ts")
        store.write_bars("BTCUSDT", [_bar(symbol="BTCUSDT")])
        store.write_bars("ETHUSDT", [_bar(symbol="ETHUSDT")])

        symbols = store.list_symbols()
        assert "BTCUSDT" in symbols
        assert "ETHUSDT" in symbols

    def test_preserves_timezone(self, tmp_path: Path) -> None:
        store = TimeSeriesStore(tmp_path / "ts")
        store.write_bars("BTCUSDT", [_bar()])
        result = store.read_bars("BTCUSDT")
        assert result[0].ts.tzinfo is not None
