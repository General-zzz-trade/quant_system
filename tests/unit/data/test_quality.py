"""Tests for data quality validators, gap detection, and gap filling."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from data.quality.gaps import Gap, GapDetector, GapFiller, GapReport
from data.quality.validators import BarValidator, ValidationResult
from data.store import Bar


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bar(
    ts: datetime,
    open_: float = 100.0,
    high: float = 105.0,
    low: float = 95.0,
    close: float = 102.0,
    volume: float = 1000.0,
    symbol: str = "BTCUSDT",
) -> Bar:
    return Bar(
        ts=ts,
        open=Decimal(str(open_)),
        high=Decimal(str(high)),
        low=Decimal(str(low)),
        close=Decimal(str(close)),
        volume=Decimal(str(volume)),
        symbol=symbol,
    )


T0 = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)


def _make_bars(count: int, *, interval_sec: int = 3600, start: datetime = T0) -> list[Bar]:
    """Create a contiguous sequence of valid bars."""
    bars: list[Bar] = []
    for i in range(count):
        ts = start + timedelta(seconds=interval_sec * i)
        close = 100.0 + i * 0.5
        bars.append(
            _bar(ts, open_=close - 0.3, high=close + 2.0, low=close - 2.0, close=close)
        )
    return bars


# ===========================================================================
# BarValidator tests
# ===========================================================================

class TestBarValidator:
    def test_valid_bars_pass(self) -> None:
        bars = _make_bars(10)
        result = BarValidator().validate(bars)
        assert result.valid is True
        assert len(result.errors) == 0
        assert result.stats["total_bars"] == 10

    def test_empty_bars(self) -> None:
        result = BarValidator().validate([])
        assert result.valid is True
        assert "empty bar sequence" in result.warnings

    def test_invalid_ohlc_high_below_close(self) -> None:
        bars = [
            _bar(T0, open_=100.0, high=99.0, low=95.0, close=102.0),
        ]
        result = BarValidator().validate(bars)
        assert result.valid is False
        assert any("OHLC" in e and "high" in e for e in result.errors)

    def test_invalid_ohlc_low_above_open(self) -> None:
        bars = [
            _bar(T0, open_=100.0, high=105.0, low=101.0, close=102.0),
        ]
        result = BarValidator().validate(bars)
        assert result.valid is False
        assert any("OHLC" in e and "low" in e for e in result.errors)

    def test_negative_volume_fails(self) -> None:
        bars = [_bar(T0, volume=-5.0)]
        result = BarValidator().validate(bars)
        assert result.valid is False
        assert any("volume" in e.lower() for e in result.errors)

    def test_duplicate_timestamp_fails(self) -> None:
        bars = [_bar(T0), _bar(T0)]
        result = BarValidator().validate(bars)
        assert result.valid is False
        assert any("duplicate" in e for e in result.errors)

    def test_out_of_order_fails(self) -> None:
        bars = [_bar(T0 + timedelta(hours=1)), _bar(T0)]
        result = BarValidator().validate(bars)
        assert result.valid is False
        assert any("out of order" in e for e in result.errors)

    def test_large_gap_warning(self) -> None:
        bars = [_bar(T0), _bar(T0 + timedelta(hours=5))]
        result = BarValidator(max_gap_seconds=3600).validate(bars)
        assert result.valid is True
        assert any("gap" in w for w in result.warnings)

    def test_anomaly_detected(self) -> None:
        bars = _make_bars(20)
        # Inject a massive spike
        anomalous_ts = T0 + timedelta(hours=10)
        bars[10] = _bar(
            anomalous_ts,
            open_=100.0,
            high=500.0,
            low=100.0,
            close=500.0,
        )
        result = BarValidator(zscore_threshold=3.0).validate(bars)
        assert result.stats["anomalies"] > 0
        assert any("anomalous" in w for w in result.warnings)

    def test_stats_populated(self) -> None:
        bars = _make_bars(5)
        result = BarValidator().validate(bars)
        assert "total_bars" in result.stats
        assert "anomalies" in result.stats
        assert result.stats["total_bars"] == 5


# ===========================================================================
# GapDetector tests
# ===========================================================================

class TestGapDetector:
    def test_no_gaps(self) -> None:
        bars = _make_bars(5, interval_sec=3600)
        detector = GapDetector(interval_seconds=3600)
        report = detector.detect(bars, start=T0, end=T0 + timedelta(hours=4))
        assert len(report.gaps) == 0
        assert report.completeness_pct == 100.0
        assert report.total_actual == report.total_expected

    def test_detects_gap(self) -> None:
        # Create bars at hours 0, 1, 2, 5, 6  (gap at 3, 4)
        bars = [
            _bar(T0),
            _bar(T0 + timedelta(hours=1)),
            _bar(T0 + timedelta(hours=2)),
            _bar(T0 + timedelta(hours=5)),
            _bar(T0 + timedelta(hours=6)),
        ]
        detector = GapDetector(interval_seconds=3600)
        report = detector.detect(bars, start=T0, end=T0 + timedelta(hours=6))
        assert len(report.gaps) == 1
        assert report.gaps[0].expected_bars == 2
        assert report.gaps[0].reason == "missing"
        assert report.completeness_pct < 100.0

    def test_all_missing(self) -> None:
        detector = GapDetector(interval_seconds=3600)
        report = detector.detect([], start=T0, end=T0 + timedelta(hours=3))
        assert len(report.gaps) == 1
        assert report.completeness_pct == 0.0

    def test_symbol_from_bars(self) -> None:
        bars = [_bar(T0, symbol="ETHUSDT")]
        detector = GapDetector(interval_seconds=3600)
        report = detector.detect(bars, start=T0, end=T0)
        assert report.symbol == "ETHUSDT"


# ===========================================================================
# GapFiller tests
# ===========================================================================

class TestGapFiller:
    def test_forward_fill(self) -> None:
        bars = [
            _bar(T0, close=100.0),
            _bar(T0 + timedelta(hours=3), close=110.0),
        ]
        gap = Gap(
            start=T0 + timedelta(hours=1),
            end=T0 + timedelta(hours=2),
            expected_bars=2,
            reason="missing",
        )
        filler = GapFiller()
        filled = filler.fill_forward(bars, [gap])
        assert len(filled) > len(bars)
        # Filled bars should use the last known close (100.0)
        for bar in filled:
            if bar.ts > T0 and bar.ts < T0 + timedelta(hours=3):
                assert bar.close == Decimal("100.0")

    def test_linear_fill(self) -> None:
        bars = [
            _bar(T0, close=100.0),
            _bar(T0 + timedelta(hours=3), close=130.0),
        ]
        gap = Gap(
            start=T0 + timedelta(hours=1),
            end=T0 + timedelta(hours=2),
            expected_bars=2,
            reason="missing",
        )
        filler = GapFiller()
        filled = filler.fill_linear(bars, [gap])
        assert len(filled) > len(bars)
        # Filled bars should be between 100 and 130
        for bar in filled:
            if bar.ts > T0 and bar.ts < T0 + timedelta(hours=3):
                assert Decimal("100.0") <= bar.close <= Decimal("130.0")

    def test_fill_no_gaps(self) -> None:
        bars = _make_bars(3)
        filler = GapFiller()
        filled = filler.fill_forward(bars, [])
        assert len(filled) == 3

    def test_filled_bars_sorted(self) -> None:
        bars = [
            _bar(T0, close=100.0),
            _bar(T0 + timedelta(hours=4), close=120.0),
        ]
        gap = Gap(
            start=T0 + timedelta(hours=1),
            end=T0 + timedelta(hours=3),
            expected_bars=3,
            reason="missing",
        )
        filler = GapFiller()
        filled = filler.fill_forward(bars, [gap])
        timestamps = [b.ts for b in filled]
        assert timestamps == sorted(timestamps)
