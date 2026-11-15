"""The overlay reads only the close of each complete group; ``group_closes`` gives exactly those rows of
``aggregate`` without pandas groupby, which was most of the runtime's per-bar cost (2026-11-15 speed work).

Exactness matters: the strategy's weights are computed from these closes, research and the runtime call the
same function, and the paper runtime must keep reproducing the research simulation bar by bar.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading.domain.market import HOUR, aggregate, group_closes


def _frame(rng, start_hour, hours, drop):
    """Sorted, unique hourly bars from ``start_hour`` with ``drop`` of them missing at random places."""
    open_hours = np.arange(start_hour, start_hour + hours)
    keep = np.ones(hours, dtype=bool)
    keep[rng.choice(hours, size=drop, replace=False)] = False
    open_time = open_hours[keep].astype(np.int64) * HOUR
    close = rng.lognormal(10, 0.05, size=len(open_time))
    return pd.DataFrame({"open_time": open_time, "open": close, "high": close, "low": close, "close": close,
                         "volume": rng.random(len(open_time))}).assign(available_at=lambda f: f["open_time"] + HOUR)


@pytest.mark.parametrize("hours", [4, 24])
@pytest.mark.parametrize("seed", range(6))
def test_group_closes_are_exactly_the_closes_of_aggregate(hours, seed):
    rng = np.random.default_rng(seed)
    frame = _frame(rng, start_hour=int(rng.integers(400_000, 500_000)), hours=int(rng.integers(30, 900)),
                   drop=int(rng.integers(0, 6)))
    expected = aggregate(frame, hours)[["available_at", "close"]]
    pd.testing.assert_frame_equal(group_closes(frame, hours), expected, check_exact=True)


def test_partial_and_misaligned_groups_are_left_out():
    rng = np.random.default_rng(1)
    frame = _frame(rng, start_hour=24 * 20_000 + 3, hours=24 * 3, drop=0)      # starts 3 h into a day
    got = group_closes(frame, 24)
    assert list(got["available_at"]) == [24 * 20_002 * HOUR, 24 * 20_003 * HOUR]  # day 1 partial, day 4 partial
    assert got["close"].iloc[0] == frame.loc[frame["open_time"] == (24 * 20_002 - 1) * HOUR, "close"].item()


def test_an_empty_frame_has_no_groups():
    frame = _frame(np.random.default_rng(0), start_hour=0, hours=1, drop=1)
    assert len(group_closes(frame, 24)) == 0 and list(group_closes(frame, 24).columns) == ["available_at", "close"]


def test_groups_whose_bars_do_not_start_on_the_hour_are_left_out_as_aggregate_does():
    frame = _frame(np.random.default_rng(2), start_hour=24 * 20_000, hours=24 * 3, drop=0)
    frame["open_time"] += HOUR // 2                                     # every bar 30 min off the hour
    assert len(aggregate(frame, 24)) == 0
    pd.testing.assert_frame_equal(group_closes(frame, 24), aggregate(frame, 24)[["available_at", "close"]],
                                  check_exact=True, check_index_type=False)
