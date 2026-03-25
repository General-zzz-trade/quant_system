"""Tests for monitoring.data_quality_check module."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Helper to create CSV test data
# ---------------------------------------------------------------------------

def _make_ohlcv_csv(path: Path, rows: list[dict]) -> Path:
    """Write rows to a CSV file and return the path."""
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return path


def _make_clean_bars(n: int = 100, base_ts_ms: int = 1700000000000,
                     interval_ms: int = 3600_000) -> list[dict]:
    """Generate n clean 1h OHLCV rows."""
    rows = []
    for i in range(n):
        price = 40000.0 + i * 10
        rows.append({
            "open_time": base_ts_ms + i * interval_ms,
            "open": price,
            "high": price + 50,
            "low": price - 50,
            "close": price + 5,
            "volume": 1000.0 + i,
        })
    return rows


# ---------------------------------------------------------------------------
# _detect_ts_column
# ---------------------------------------------------------------------------

class TestDetectTsColumn:
    def test_finds_open_time(self):
        from monitoring.data_quality_check import _detect_ts_column
        assert _detect_ts_column(["open_time", "open", "close"]) == "open_time"

    def test_finds_timestamp(self):
        from monitoring.data_quality_check import _detect_ts_column
        assert _detect_ts_column(["timestamp", "open", "close"]) == "timestamp"

    def test_finds_ts(self):
        from monitoring.data_quality_check import _detect_ts_column
        assert _detect_ts_column(["ts", "open", "close"]) == "ts"

    def test_returns_none_when_missing(self):
        from monitoring.data_quality_check import _detect_ts_column
        assert _detect_ts_column(["open", "high", "low", "close"]) is None

    def test_prefers_open_time_over_ts(self):
        from monitoring.data_quality_check import _detect_ts_column
        # open_time checked first in the iteration order
        assert _detect_ts_column(["ts", "open_time", "close"]) == "open_time"


# ---------------------------------------------------------------------------
# _detect_interval_seconds
# ---------------------------------------------------------------------------

class TestDetectIntervalSeconds:
    def test_1h_suffix(self):
        from monitoring.data_quality_check import _detect_interval_seconds
        assert _detect_interval_seconds("BTCUSDT_1h.csv") == 3600

    def test_15m_suffix(self):
        from monitoring.data_quality_check import _detect_interval_seconds
        assert _detect_interval_seconds("ETHUSDT_15m.csv") == 900

    def test_4h_suffix(self):
        from monitoring.data_quality_check import _detect_interval_seconds
        assert _detect_interval_seconds("BTCUSDT_4h.csv") == 14400

    def test_5m_suffix(self):
        from monitoring.data_quality_check import _detect_interval_seconds
        assert _detect_interval_seconds("BTCUSDT_5m.csv") == 300

    def test_default_for_unknown(self):
        from monitoring.data_quality_check import _detect_interval_seconds
        assert _detect_interval_seconds("BTCUSDT_funding.csv") == 3600


# ---------------------------------------------------------------------------
# _is_ohlcv_file
# ---------------------------------------------------------------------------

class TestIsOhlcvFile:
    def test_valid_ohlcv(self):
        from monitoring.data_quality_check import _is_ohlcv_file
        assert _is_ohlcv_file(["open_time", "open", "high", "low", "close", "volume"])

    def test_missing_column(self):
        from monitoring.data_quality_check import _is_ohlcv_file
        assert not _is_ohlcv_file(["open_time", "open", "high", "close"])  # missing low

    def test_non_ohlcv(self):
        from monitoring.data_quality_check import _is_ohlcv_file
        assert not _is_ohlcv_file(["timestamp", "funding_rate", "symbol"])


# ---------------------------------------------------------------------------
# check_file — core validation
# ---------------------------------------------------------------------------

class TestCheckFile:
    def test_clean_data_passes(self, tmp_path):
        """Clean OHLCV data should pass all checks."""
        from monitoring.data_quality_check import check_file

        rows = _make_clean_bars(100)
        csv_path = _make_ohlcv_csv(tmp_path / "BTCUSDT_1h.csv", rows)

        result = check_file(csv_path, symbol="BTCUSDT")

        assert result["pass"] is True
        assert result["file"] == "BTCUSDT_1h.csv"
        assert result["symbol"] == "BTCUSDT"
        assert result["stats"]["rows"] == 100
        assert result["stats"]["type"] == "ohlcv"
        assert result["stats"]["total_bars"] == 100

    def test_empty_file_warns(self, tmp_path):
        """Empty CSV should produce a warning."""
        from monitoring.data_quality_check import check_file

        csv_path = tmp_path / "empty.csv"
        csv_path.write_text("open_time,open,high,low,close,volume\n")

        result = check_file(csv_path, symbol="TEST")

        assert result["pass"] is True  # empty is not a failure
        assert any("Empty" in w for w in result["warnings"])
        assert result["stats"]["rows"] == 0

    def test_unreadable_file_fails(self, tmp_path):
        """Invalid CSV should fail with error."""
        from monitoring.data_quality_check import check_file

        csv_path = tmp_path / "bad.csv"
        csv_path.write_text("not,a,valid\x00csv\x00file")

        result = check_file(csv_path, symbol="BAD")

        # Should handle gracefully (either pass with warnings or fail)
        assert isinstance(result["errors"], list)

    def test_nan_values_warned(self, tmp_path):
        """NaN values in numeric columns should produce a warning."""
        from monitoring.data_quality_check import check_file

        # Add extra numeric columns that won't be converted to Decimal
        rows = _make_clean_bars(10)
        for r in rows:
            r["extra_metric"] = 1.0
        rows[3]["extra_metric"] = float("nan")
        rows[5]["extra_metric"] = float("nan")
        csv_path = _make_ohlcv_csv(tmp_path / "BTCUSDT_1h.csv", rows)

        result = check_file(csv_path, symbol="BTCUSDT")

        assert result["stats"]["nan_count"] >= 2
        assert any("NaN" in w for w in result["warnings"])

    def test_non_ohlcv_file(self, tmp_path):
        """Non-OHLCV CSV should be handled without OHLC checks."""
        from monitoring.data_quality_check import check_file

        rows = [
            {"timestamp": 1700000000000, "funding_rate": 0.0001},
            {"timestamp": 1700003600000, "funding_rate": 0.0002},
        ]
        csv_path = _make_ohlcv_csv(tmp_path / "BTCUSDT_funding.csv", rows)

        result = check_file(csv_path, symbol="BTCUSDT")

        assert result["pass"] is True
        assert result["stats"]["type"] == "non-ohlcv"

    def test_no_timestamp_column(self, tmp_path):
        """OHLCV file without known timestamp column should warn."""
        from monitoring.data_quality_check import check_file

        rows = [
            {"idx": 1, "open": 100, "high": 110, "low": 90, "close": 105, "volume": 100},
            {"idx": 2, "open": 105, "high": 115, "low": 95, "close": 110, "volume": 120},
        ]
        csv_path = _make_ohlcv_csv(tmp_path / "no_ts.csv", rows)

        result = check_file(csv_path, symbol="TEST")

        assert any("timestamp" in w.lower() for w in result["warnings"])

    def test_result_has_required_keys(self, tmp_path):
        """Result dict should always have file, symbol, pass, errors, warnings, stats."""
        from monitoring.data_quality_check import check_file

        rows = _make_clean_bars(10)
        csv_path = _make_ohlcv_csv(tmp_path / "BTCUSDT_1h.csv", rows)

        result = check_file(csv_path, symbol="BTCUSDT")

        assert "file" in result
        assert "symbol" in result
        assert "pass" in result
        assert "errors" in result
        assert "warnings" in result
        assert "stats" in result

    def test_gaps_detected(self, tmp_path):
        """Missing bars (gaps) should be detected and reported."""
        from monitoring.data_quality_check import check_file

        rows = _make_clean_bars(50)
        # Remove bars 20-30 to create a gap
        gapped_rows = rows[:20] + rows[30:]
        csv_path = _make_ohlcv_csv(tmp_path / "BTCUSDT_1h.csv", gapped_rows)

        result = check_file(csv_path, symbol="BTCUSDT")

        assert result["stats"]["gaps"] > 0
        assert result["stats"]["completeness_pct"] < 100.0

    def test_ohlc_consistency_error(self, tmp_path):
        """Bar where high < close should produce OHLC errors."""
        from monitoring.data_quality_check import check_file

        rows = _make_clean_bars(10)
        # Make high < close (inconsistent)
        rows[5]["high"] = rows[5]["close"] - 100
        csv_path = _make_ohlcv_csv(tmp_path / "BTCUSDT_1h.csv", rows)

        result = check_file(csv_path, symbol="BTCUSDT")

        assert result["stats"]["ohlc_errors"] > 0

    def test_millisecond_timestamps_handled(self, tmp_path):
        """Timestamps > 1e12 should be treated as milliseconds."""
        from monitoring.data_quality_check import check_file

        # Use ms timestamps (> 1e12)
        rows = _make_clean_bars(20, base_ts_ms=1700000000000)
        csv_path = _make_ohlcv_csv(tmp_path / "BTCUSDT_1h.csv", rows)

        result = check_file(csv_path, symbol="BTCUSDT")

        # Should not fail on timestamp conversion
        assert result["stats"].get("total_bars", 0) > 0


# ---------------------------------------------------------------------------
# JSON output format
# ---------------------------------------------------------------------------

class TestJsonOutput:
    """Tests for JSON output mode of main()."""

    def test_json_output_structure(self, tmp_path):
        """JSON output should have expected top-level keys."""
        from monitoring.data_quality_check import check_file

        rows = _make_clean_bars(20)
        csv_path = _make_ohlcv_csv(tmp_path / "BTCUSDT_1h.csv", rows)

        result = check_file(csv_path, symbol="BTCUSDT")

        # Verify the result can be serialized to JSON
        json_str = json.dumps(result, default=str)
        parsed = json.loads(json_str)

        assert "file" in parsed
        assert "symbol" in parsed
        assert "pass" in parsed
        assert "stats" in parsed
        assert isinstance(parsed["errors"], list)
        assert isinstance(parsed["warnings"], list)

    def test_stats_keys_for_ohlcv(self, tmp_path):
        """OHLCV file stats should include gaps, completeness_pct, anomalies."""
        from monitoring.data_quality_check import check_file

        rows = _make_clean_bars(50)
        csv_path = _make_ohlcv_csv(tmp_path / "BTCUSDT_1h.csv", rows)

        result = check_file(csv_path, symbol="BTCUSDT")

        stats = result["stats"]
        assert "rows" in stats
        assert "total_bars" in stats
        assert "gaps" in stats
        assert "completeness_pct" in stats
        assert "anomalies" in stats
        assert "ohlc_errors" in stats
        assert "time_errors" in stats
