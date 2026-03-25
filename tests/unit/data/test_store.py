"""Tests for data.store — TimeSeriesStore JSONL round-trip, filtering, edge cases."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

from data.store import Bar, TimeSeriesStore


T0 = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)


def _bar(
    ts: datetime,
    open_: float = 100.0,
    high: float = 105.0,
    low: float = 95.0,
    close: float = 102.0,
    volume: float | None = 1000.0,
    symbol: str = "BTCUSDT",
    exchange: str = "bybit",
) -> Bar:
    return Bar(
        ts=ts,
        open=Decimal(str(open_)),
        high=Decimal(str(high)),
        low=Decimal(str(low)),
        close=Decimal(str(close)),
        volume=Decimal(str(volume)) if volume is not None else None,
        symbol=symbol,
        exchange=exchange,
    )


class TestTimeSeriesStoreRoundTrip:
    """Write bars then read them back — values must match."""

    def test_write_read_roundtrip(self, tmp_path) -> None:
        store = TimeSeriesStore(tmp_path / "store")
        bars = [
            _bar(T0, close=100.0),
            _bar(T0 + timedelta(hours=1), close=101.5),
            _bar(T0 + timedelta(hours=2), close=99.8),
        ]
        store.write_bars("BTCUSDT", bars)
        result = store.read_bars("BTCUSDT")

        assert len(result) == 3
        assert result[0].close == Decimal("100.0")
        assert result[1].close == Decimal("101.5")
        assert result[2].close == Decimal("99.8")

    def test_roundtrip_preserves_all_fields(self, tmp_path) -> None:
        store = TimeSeriesStore(tmp_path / "store")
        original = _bar(
            T0,
            open_=50000.12,
            high=51000.99,
            low=49500.01,
            close=50800.55,
            volume=123.456,
            symbol="ETHUSDT",
            exchange="binance",
        )
        store.write_bars("ETHUSDT", [original])
        result = store.read_bars("ETHUSDT")

        assert len(result) == 1
        b = result[0]
        assert b.open == Decimal("50000.12")
        assert b.high == Decimal("51000.99")
        assert b.low == Decimal("49500.01")
        assert b.close == Decimal("50800.55")
        assert b.volume == Decimal("123.456")
        assert b.symbol == "ETHUSDT"
        assert b.exchange == "binance"

    def test_roundtrip_none_volume(self, tmp_path) -> None:
        store = TimeSeriesStore(tmp_path / "store")
        bar = _bar(T0, volume=None)
        store.write_bars("BTCUSDT", [bar])
        result = store.read_bars("BTCUSDT")

        assert len(result) == 1
        assert result[0].volume is None


class TestTimeSeriesStoreFiltering:
    """Time-range filtering on read_bars."""

    def test_filter_by_start(self, tmp_path) -> None:
        store = TimeSeriesStore(tmp_path / "store")
        bars = [
            _bar(T0),
            _bar(T0 + timedelta(hours=1)),
            _bar(T0 + timedelta(hours=2)),
        ]
        store.write_bars("BTCUSDT", bars)

        result = store.read_bars("BTCUSDT", start=T0 + timedelta(hours=1))
        assert len(result) == 2

    def test_filter_by_end(self, tmp_path) -> None:
        store = TimeSeriesStore(tmp_path / "store")
        bars = [
            _bar(T0),
            _bar(T0 + timedelta(hours=1)),
            _bar(T0 + timedelta(hours=2)),
        ]
        store.write_bars("BTCUSDT", bars)

        result = store.read_bars("BTCUSDT", end=T0 + timedelta(hours=1))
        assert len(result) == 2

    def test_filter_by_range(self, tmp_path) -> None:
        store = TimeSeriesStore(tmp_path / "store")
        bars = [
            _bar(T0 + timedelta(hours=i)) for i in range(5)
        ]
        store.write_bars("BTCUSDT", bars)

        result = store.read_bars(
            "BTCUSDT",
            start=T0 + timedelta(hours=1),
            end=T0 + timedelta(hours=3),
        )
        assert len(result) == 3


class TestTimeSeriesStoreEmpty:
    """Edge cases: empty data, missing symbols."""

    def test_read_nonexistent_symbol_returns_empty(self, tmp_path) -> None:
        store = TimeSeriesStore(tmp_path / "store")
        result = store.read_bars("NONEXISTENT")
        assert result == []

    def test_write_empty_bars(self, tmp_path) -> None:
        store = TimeSeriesStore(tmp_path / "store")
        store.write_bars("BTCUSDT", [])
        result = store.read_bars("BTCUSDT")
        assert result == []

    def test_list_symbols_empty(self, tmp_path) -> None:
        store = TimeSeriesStore(tmp_path / "store")
        assert store.list_symbols() == []


class TestTimeSeriesStoreAppend:
    """Multiple writes to the same symbol should accumulate bars."""

    def test_append_bars(self, tmp_path) -> None:
        store = TimeSeriesStore(tmp_path / "store")
        store.write_bars("BTCUSDT", [_bar(T0, close=100.0)])
        store.write_bars("BTCUSDT", [_bar(T0 + timedelta(hours=1), close=101.0)])

        result = store.read_bars("BTCUSDT")
        assert len(result) == 2
        assert result[0].close == Decimal("100.0")
        assert result[1].close == Decimal("101.0")


class TestTimeSeriesStoreListSymbols:
    """list_symbols should return all written symbols, sorted."""

    def test_list_multiple_symbols(self, tmp_path) -> None:
        store = TimeSeriesStore(tmp_path / "store")
        store.write_bars("ETHUSDT", [_bar(T0)])
        store.write_bars("BTCUSDT", [_bar(T0)])
        store.write_bars("SOLUSDT", [_bar(T0)])

        symbols = store.list_symbols()
        assert symbols == ["BTCUSDT", "ETHUSDT", "SOLUSDT"]


class TestBarDataclass:
    """Bar is a frozen dataclass with correct fields."""

    def test_frozen(self) -> None:
        bar = _bar(T0)
        try:
            bar.close = Decimal("999")  # type: ignore[misc]
            assert False, "should have raised"
        except AttributeError:
            pass

    def test_slots(self) -> None:
        bar = _bar(T0)
        assert not hasattr(bar, "__dict__")
