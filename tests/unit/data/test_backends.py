"""Tests for data storage backends."""
from __future__ import annotations

import tempfile
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from data.backends.base import Tick
from data.backends.parquet_backend import ParquetBarStore
from data.store import Bar


def _make_bars(n: int, symbol: str = "BTCUSDT", start: datetime | None = None) -> list[Bar]:
    base = start or datetime(2024, 1, 1, tzinfo=timezone.utc)
    bars = []
    for i in range(n):
        ts = base + timedelta(minutes=i)
        price = Decimal("40000") + Decimal(i)
        bars.append(
            Bar(
                ts=ts,
                open=price,
                high=price + Decimal("10"),
                low=price - Decimal("10"),
                close=price + Decimal("5"),
                volume=Decimal("100") + Decimal(i),
                symbol=symbol,
                exchange="binance",
            )
        )
    return bars


class TestParquetBarStore:
    def test_write_and_read_bars(self, tmp_path: Path) -> None:
        store = ParquetBarStore(tmp_path / "bars")
        bars = _make_bars(100)

        store.write_bars("BTCUSDT", bars)
        result = store.read_bars("BTCUSDT")

        assert len(result) == 100
        assert result[0].ts == bars[0].ts
        assert result[-1].ts == bars[-1].ts
        assert result[0].open == Decimal("40000")
        assert result[99].close == Decimal("40099") + Decimal("5")

    def test_read_with_time_filter(self, tmp_path: Path) -> None:
        store = ParquetBarStore(tmp_path / "bars")
        bars = _make_bars(100)
        store.write_bars("BTCUSDT", bars)

        start = datetime(2024, 1, 1, 0, 10, tzinfo=timezone.utc)
        end = datetime(2024, 1, 1, 0, 20, tzinfo=timezone.utc)
        result = store.read_bars("BTCUSDT", start=start, end=end)

        assert len(result) == 11  # inclusive on both ends
        assert result[0].ts >= start
        assert result[-1].ts <= end

    def test_symbols(self, tmp_path: Path) -> None:
        store = ParquetBarStore(tmp_path / "bars")
        store.write_bars("BTCUSDT", _make_bars(5, symbol="BTCUSDT"))
        store.write_bars("ETHUSDT", _make_bars(5, symbol="ETHUSDT"))

        syms = store.symbols()
        assert "BTCUSDT" in syms
        assert "ETHUSDT" in syms

    def test_date_range(self, tmp_path: Path) -> None:
        store = ParquetBarStore(tmp_path / "bars")
        bars = _make_bars(50)
        store.write_bars("BTCUSDT", bars)

        dr = store.date_range("BTCUSDT")
        assert dr is not None
        assert dr[0] == bars[0].ts
        assert dr[1] == bars[-1].ts

    def test_date_range_empty(self, tmp_path: Path) -> None:
        store = ParquetBarStore(tmp_path / "bars")
        assert store.date_range("NONEXISTENT") is None

    def test_write_empty_is_noop(self, tmp_path: Path) -> None:
        store = ParquetBarStore(tmp_path / "bars")
        store.write_bars("BTCUSDT", [])
        assert store.read_bars("BTCUSDT") == []


class TestTick:
    def test_tick_creation(self) -> None:
        ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        tick = Tick(
            ts=ts,
            symbol="BTCUSDT",
            price=Decimal("42000.50"),
            qty=Decimal("0.5"),
            side="buy",
            trade_id="123456",
        )
        assert tick.ts == ts
        assert tick.symbol == "BTCUSDT"
        assert tick.price == Decimal("42000.50")
        assert tick.qty == Decimal("0.5")
        assert tick.side == "buy"
        assert tick.trade_id == "123456"

    def test_tick_frozen(self) -> None:
        tick = Tick(
            ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
            symbol="BTCUSDT",
            price=Decimal("42000"),
            qty=Decimal("1"),
            side="sell",
        )
        with pytest.raises(AttributeError):
            tick.price = Decimal("43000")  # type: ignore[misc]

    def test_tick_default_trade_id(self) -> None:
        tick = Tick(
            ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
            symbol="ETHUSDT",
            price=Decimal("2500"),
            qty=Decimal("10"),
            side="buy",
        )
        assert tick.trade_id == ""
