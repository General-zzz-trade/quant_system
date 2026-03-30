"""Tests for _load_schedule handling mixed-format CSV files.

Regression test for the bug where newer ls_ratio/OI CSV rows switched from
2-column (timestamp, value) to 5-column (timestamp, symbol, value, long%, short%),
causing DictReader to put the symbol string into the value column.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from features.batch_feature_engine import _load_schedule


def _write_csv(content: str) -> Path:
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
    f.write(content)
    f.close()
    return Path(f.name)


class TestLoadScheduleMixedFormat:
    """_load_schedule should handle CSV format transitions gracefully."""

    def test_pure_2col(self):
        path = _write_csv(
            "timestamp,long_short_ratio\n"
            "1000,1.5\n"
            "2000,2.0\n"
        )
        result = _load_schedule(path, "timestamp", "long_short_ratio")
        assert result == {1000: 1.5, 2000: 2.0}

    def test_mixed_2col_then_5col(self):
        """Older 2-col rows followed by newer 5-col rows with symbol."""
        path = _write_csv(
            "timestamp,long_short_ratio\n"
            "1000,1.5\n"
            "2000,2.0\n"
            "3000,BTCUSDT,2.686,0.7287,0.2713\n"
            "4000,BTCUSDT,2.632,0.7247,0.2753\n"
        )
        result = _load_schedule(path, "timestamp", "long_short_ratio")
        assert result[1000] == 1.5
        assert result[2000] == 2.0
        # 5-col rows: val_col gets 'BTCUSDT', overflow[0] = '2.686'
        assert abs(result[3000] - 2.686) < 0.001
        assert abs(result[4000] - 2.632) < 0.001
        assert len(result) == 4

    def test_all_5col(self):
        """All rows in new 5-column format."""
        path = _write_csv(
            "timestamp,long_short_ratio\n"
            "1000,ETHUSDT,2.4188,0.7075,0.2925\n"
            "2000,ETHUSDT,2.3829,0.7044,0.2956\n"
        )
        result = _load_schedule(path, "timestamp", "long_short_ratio")
        assert abs(result[1000] - 2.4188) < 0.001
        assert abs(result[2000] - 2.3829) < 0.001

    def test_missing_file(self):
        result = _load_schedule(Path("/nonexistent.csv"), "ts", "val")
        assert result == {}

    def test_funding_rate_unaffected(self):
        """Normal 2-col CSV (like funding) should still work."""
        path = _write_csv(
            "timestamp,funding_rate\n"
            "1000,0.0001\n"
            "2000,-0.0002\n"
        )
        result = _load_schedule(path, "timestamp", "funding_rate")
        assert abs(result[1000] - 0.0001) < 1e-6
        assert abs(result[2000] - (-0.0002)) < 1e-6
