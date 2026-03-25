"""Tests for data quality wiring — validates that BarValidator and GapDetector
are properly integrated into warmup and the standalone monitoring tool."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

from data.quality.gaps import GapDetector
from data.quality.validators import BarValidator
from data.store import Bar


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

T0 = datetime(2025, 6, 1, 0, 0, 0, tzinfo=timezone.utc)


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


def _make_bars(count: int, interval_sec: int = 3600) -> list[Bar]:
    bars = []
    for i in range(count):
        ts = T0 + timedelta(seconds=interval_sec * i)
        close = 100.0 + i * 0.5
        bars.append(
            _bar(ts, open_=close - 0.3, high=close + 2.0, low=close - 2.0, close=close)
        )
    return bars


# ===========================================================================
# Test: BarValidator catches OHLC violation
# ===========================================================================

class TestBarValidatorCatchesOHLCViolation:
    def test_high_below_close(self) -> None:
        """BarValidator should flag when high < close."""
        bars = [_bar(T0, open_=100.0, high=99.0, low=95.0, close=102.0)]
        result = BarValidator().validate(bars)
        assert result.valid is False
        assert result.stats["ohlc_errors"] > 0
        assert any("OHLC" in e for e in result.errors)

    def test_low_above_open(self) -> None:
        """BarValidator should flag when low > min(open, close)."""
        bars = [_bar(T0, open_=100.0, high=105.0, low=101.0, close=102.0)]
        result = BarValidator().validate(bars)
        assert result.valid is False
        assert any("low" in e for e in result.errors)

    def test_high_below_low(self) -> None:
        """BarValidator should flag when high < low."""
        bars = [_bar(T0, open_=100.0, high=90.0, low=95.0, close=92.0)]
        result = BarValidator().validate(bars)
        assert result.valid is False
        assert any("high" in e and "low" in e for e in result.errors)

    def test_valid_bars_pass(self) -> None:
        """Valid bars should produce no errors."""
        bars = _make_bars(10)
        result = BarValidator().validate(bars)
        assert result.valid is True
        assert len(result.errors) == 0


# ===========================================================================
# Test: GapDetector finds missing bars
# ===========================================================================

class TestGapDetectorFindsMissingBars:
    def test_detects_single_gap(self) -> None:
        """GapDetector should find a gap where bars are missing."""
        # Bars at hours 0, 1, 2, 5, 6 (gap at 3, 4)
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
        assert report.completeness_pct < 100.0

    def test_no_gaps_full_sequence(self) -> None:
        """Contiguous bars should have no gaps."""
        bars = _make_bars(10)
        detector = GapDetector(interval_seconds=3600)
        report = detector.detect(
            bars, start=T0, end=T0 + timedelta(hours=9)
        )
        assert len(report.gaps) == 0
        assert report.completeness_pct == 100.0

    def test_multiple_gaps(self) -> None:
        """GapDetector should find multiple disjoint gaps."""
        # Bars at hours 0, 1, 4, 5, 8 (gaps at 2-3 and 6-7)
        bars = [
            _bar(T0),
            _bar(T0 + timedelta(hours=1)),
            _bar(T0 + timedelta(hours=4)),
            _bar(T0 + timedelta(hours=5)),
            _bar(T0 + timedelta(hours=8)),
        ]
        detector = GapDetector(interval_seconds=3600)
        report = detector.detect(bars, start=T0, end=T0 + timedelta(hours=8))
        assert len(report.gaps) == 2


# ===========================================================================
# Test: Quality check on sample CSV data
# ===========================================================================

class TestQualityCheckOnSampleData:
    def test_check_file_valid_csv(self, tmp_path: Path) -> None:
        """check_file should PASS on a valid OHLCV CSV."""
        import pandas as pd

        # Create a valid CSV
        rows = []
        for i in range(20):
            ts = T0 + timedelta(hours=i)
            close = 100.0 + i * 0.5
            rows.append({
                "open_time": int(ts.timestamp() * 1000),
                "open": close - 0.3,
                "high": close + 2.0,
                "low": close - 2.0,
                "close": close,
                "volume": 1000.0 + i,
            })
        df = pd.DataFrame(rows)
        csv_path = tmp_path / "BTCUSDT_1h.csv"
        df.to_csv(csv_path, index=False)

        from monitoring.data_quality_check import check_file
        result = check_file(csv_path, symbol="BTCUSDT")
        assert result["pass"] is True
        assert result["stats"]["total_bars"] == 20
        assert result["stats"]["gaps"] == 0

    def test_check_file_with_ohlc_error(self, tmp_path: Path) -> None:
        """check_file should FAIL when OHLC constraints are violated."""
        import pandas as pd

        rows = []
        for i in range(10):
            ts = T0 + timedelta(hours=i)
            rows.append({
                "open_time": int(ts.timestamp() * 1000),
                "open": 100.0,
                "high": 105.0,
                "low": 95.0,
                "close": 102.0,
                "volume": 1000.0,
            })
        # Inject OHLC violation: high < close
        rows[5]["high"] = 90.0
        rows[5]["close"] = 102.0

        df = pd.DataFrame(rows)
        csv_path = tmp_path / "BTCUSDT_1h.csv"
        df.to_csv(csv_path, index=False)

        from monitoring.data_quality_check import check_file
        result = check_file(csv_path, symbol="BTCUSDT")
        assert result["pass"] is False
        assert result["stats"]["ohlc_errors"] > 0

    def test_check_file_with_gaps(self, tmp_path: Path) -> None:
        """check_file should detect gaps in the time series."""
        import pandas as pd

        # Create bars with a gap (skip hours 3, 4)
        rows = []
        for i in [0, 1, 2, 5, 6, 7]:
            ts = T0 + timedelta(hours=i)
            rows.append({
                "open_time": int(ts.timestamp() * 1000),
                "open": 100.0,
                "high": 105.0,
                "low": 95.0,
                "close": 102.0,
                "volume": 1000.0,
            })

        df = pd.DataFrame(rows)
        csv_path = tmp_path / "BTCUSDT_1h.csv"
        df.to_csv(csv_path, index=False)

        from monitoring.data_quality_check import check_file
        result = check_file(csv_path, symbol="BTCUSDT")
        assert result["stats"]["gaps"] > 0
        assert result["stats"]["completeness_pct"] < 100.0

    def test_check_file_empty(self, tmp_path: Path) -> None:
        """check_file should handle empty CSV gracefully."""
        csv_path = tmp_path / "empty.csv"
        csv_path.write_text("open_time,open,high,low,close,volume\n")

        from monitoring.data_quality_check import check_file
        result = check_file(csv_path, symbol="TEST")
        assert result["pass"] is True
        assert result["stats"]["rows"] == 0


# ===========================================================================
# Test: Warmup validation adapter
# ===========================================================================

class TestWarmupValidation:
    def test_validate_warmup_bars_runs_without_error(self) -> None:
        """_validate_warmup_bars should process adapter-format dicts."""
        from runner.warmup import validate_warmup_bars as _validate_warmup_bars

        bars = []
        for i in range(10):
            ts = T0 + timedelta(hours=i)
            bars.append({
                "time": str(int(ts.timestamp() * 1000)),
                "open": "100.0",
                "high": "105.0",
                "low": "95.0",
                "close": "102.0",
                "volume": "1000.0",
            })

        # Should not raise
        _validate_warmup_bars(bars, "BTCUSDT", "60")

    def test_validate_warmup_bars_empty(self) -> None:
        """_validate_warmup_bars should handle empty list."""
        from runner.warmup import validate_warmup_bars as _validate_warmup_bars
        _validate_warmup_bars([], "BTCUSDT", "60")

    def test_validate_warmup_bars_4h_interval(self) -> None:
        """_validate_warmup_bars should use 4h interval for 240."""
        from runner.warmup import validate_warmup_bars as _validate_warmup_bars

        bars = []
        for i in range(5):
            ts = T0 + timedelta(hours=4 * i)
            bars.append({
                "time": str(int(ts.timestamp() * 1000)),
                "open": "100.0",
                "high": "105.0",
                "low": "95.0",
                "close": "102.0",
                "volume": "1000.0",
            })

        # Should not raise
        _validate_warmup_bars(bars, "BTCUSDT", "240")
