"""Daily data must not be visible on the day it summarizes (no look-ahead)."""
from __future__ import annotations

import math

import numpy as np

from features.batch_features_data_loaders import _daily_available
from features.batch_features_extra import _causal_align

H, DAY = 3_600_000, 86_400_000
D0 = 1_700_006_400_000 // DAY * DAY


def test_daily_row_is_usable_only_from_next_day():
    bars = np.array([D0 + 12 * H, D0 + 23 * H, D0 + DAY - H, D0 + DAY])  # opens
    out = _daily_available(bars, np.array([D0, D0 + DAY]), np.array([1.0, 2.0]))
    # Bar opening D0+23h closes at D0+24h = when day D0 becomes available.
    assert math.isnan(out[0]) and out[1] == 1.0 and out[2] == 1.0 and out[3] == 1.0


def test_causal_align_never_interpolates_or_backfills():
    obs = np.array([D0 + 10 * H, D0 + 20 * H])
    bars = np.array([D0, D0 + 9 * H, D0 + 14 * H, D0 + 19 * H])
    out = _causal_align(bars, obs, np.array([5.0, 9.0]))
    assert math.isnan(out[0])          # before the first observation: unknown, not back-filled
    assert out[1] == 5.0               # closes exactly at the observation time
    assert out[2] == 5.0               # no blend toward the next value
    assert out[3] == 9.0


def test_cursor_date_rows_shift_to_next_day(tmp_path):
    from runner.builders.alpha_builder import CsvCursor
    path = tmp_path / "spy.csv"
    path.write_text("date,close\n2023-11-15,450\n2023-11-16,452\n")
    cursor = CsvCursor(path, "date", "close", ts_unit="date")
    day = 1_700_006_400_000  # 2023-11-15 00:00 UTC
    assert math.isnan(cursor.get(day + 20 * H))   # same day: not yet known
    assert cursor.get(day + DAY) == 450.0
    assert cursor.get(day + 2 * DAY) == 452.0
