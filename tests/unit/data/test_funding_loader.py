"""Tests for data.loaders.funding_rate — CSV loading and filtering."""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from data.loaders.funding_rate import FundingRecord, load_funding_csv


T0 = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
T0_MS = int(T0.timestamp() * 1000)


def _write_csv(path: Path, rows: list[dict]) -> None:
    """Write a funding rate CSV file."""
    with path.open("w") as f:
        f.write("timestamp,symbol,funding_rate,mark_price\n")
        for r in rows:
            f.write(
                f"{r['timestamp']},{r['symbol']},{r['funding_rate']},"
                f"{r.get('mark_price', '')}\n"
            )


class TestLoadFundingCsv:
    """load_funding_csv round-trip and filtering."""

    def test_basic_load(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "funding.csv"
        _write_csv(csv_path, [
            {"timestamp": T0_MS, "symbol": "BTCUSDT", "funding_rate": "0.0001", "mark_price": "50000.0"},
            {"timestamp": T0_MS + 28800000, "symbol": "BTCUSDT", "funding_rate": "-0.0002", "mark_price": "49500.0"},
        ])
        records = load_funding_csv(str(csv_path))
        assert len(records) == 2
        assert records[0].funding_rate == Decimal("0.0001")
        assert records[1].funding_rate == Decimal("-0.0002")
        assert records[0].mark_price == Decimal("50000.0")

    def test_filter_by_symbol(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "funding.csv"
        _write_csv(csv_path, [
            {"timestamp": T0_MS, "symbol": "BTCUSDT", "funding_rate": "0.0001"},
            {"timestamp": T0_MS, "symbol": "ETHUSDT", "funding_rate": "0.0003"},
            {"timestamp": T0_MS + 28800000, "symbol": "BTCUSDT", "funding_rate": "0.0002"},
        ])
        records = load_funding_csv(str(csv_path), symbol="ETHUSDT")
        assert len(records) == 1
        assert records[0].symbol == "ETHUSDT"

    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        records = load_funding_csv(str(tmp_path / "nonexistent.csv"))
        assert records == []

    def test_empty_csv_returns_empty(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "funding.csv"
        csv_path.write_text("timestamp,symbol,funding_rate,mark_price\n")
        records = load_funding_csv(str(csv_path))
        assert records == []

    def test_records_sorted_by_timestamp(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "funding.csv"
        # Write in reverse order
        _write_csv(csv_path, [
            {"timestamp": T0_MS + 28800000, "symbol": "BTCUSDT", "funding_rate": "0.0002"},
            {"timestamp": T0_MS, "symbol": "BTCUSDT", "funding_rate": "0.0001"},
        ])
        records = load_funding_csv(str(csv_path))
        assert records[0].ts < records[1].ts

    def test_missing_mark_price_is_none(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "funding.csv"
        _write_csv(csv_path, [
            {"timestamp": T0_MS, "symbol": "BTCUSDT", "funding_rate": "0.0001"},
        ])
        records = load_funding_csv(str(csv_path))
        assert len(records) == 1
        assert records[0].mark_price is None

    def test_case_insensitive_symbol_filter(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "funding.csv"
        _write_csv(csv_path, [
            {"timestamp": T0_MS, "symbol": "BTCUSDT", "funding_rate": "0.0001"},
        ])
        records = load_funding_csv(str(csv_path), symbol="btcusdt")
        assert len(records) == 1

    def test_funding_record_frozen(self) -> None:
        rec = FundingRecord(
            ts=T0, symbol="BTCUSDT", funding_rate=Decimal("0.0001"), mark_price=None,
        )
        try:
            rec.funding_rate = Decimal("0.999")  # type: ignore[misc]
            assert False, "should have raised"
        except AttributeError:
            pass
