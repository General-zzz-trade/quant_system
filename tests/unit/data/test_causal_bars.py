"""Closed-bar causality and input validation in ``data.quality.causal_bars``.

Restored 2026-11-15 from the tests deleted with the ML offline replay in M4 (they tested kept code; M4 audit
finding 1). Assertions are unchanged; the deleted fixture module was inlined, and the unclosed-bar test
builds its CSV directly and also checks that a closed file loads.
"""
from __future__ import annotations

import csv
import math
from dataclasses import replace
from decimal import Decimal as D

import pytest

from data.quality.causal_bars import HOUR, Bar, aggregate_closed, align_closed_signals, load_bars, validate_bars


def synthetic_bars(count=1200):
    """Same series as the deleted ``runner/offline/fixtures.py``; no market evidence."""
    def price(i):
        return D(str(round(40000 + 2000 * math.sin(i / 50) + 500 * math.sin(i / 9), 2)))
    return [Bar(1704067200000 + i * HOUR, price(i), max(price(i), price(i + 1)) * D("1.005"),
                min(price(i), price(i + 1)) * D(".995"), price(i + 1), D(100)) for i in range(count)]


def test_four_hour_signal_visible_only_after_close():
    import pandas as pd
    source = pd.to_datetime(["2024-01-01T08:00Z", "2024-01-01T12:00Z"])
    targets = pd.to_datetime(["2024-01-01T09:00Z", "2024-01-01T11:59Z", "2024-01-01T12:00Z", "2024-01-01T15:00Z"])
    assert list(align_closed_signals([1, -1], source, targets)) == [0, 0, 1, 1]


def test_complete_four_hour_groups_only():
    bars = synthetic_bars(11)[1:]
    result = aggregate_closed(bars)
    assert len(result) == 1
    assert result[0].open_time == bars[3].open_time
    assert result[0].available_at == bars[6].available_at
    assert result[0].volume == 400


@pytest.mark.parametrize("damage", ["gap", "duplicate", "reverse", "ohlc", "nan", "negative", "unaligned"])
def test_bad_input_rejected(damage):
    bars = synthetic_bars(4)
    if damage == "gap":
        del bars[1]
    elif damage == "duplicate":
        bars[1] = bars[0]
    elif damage == "reverse":
        bars.reverse()
    elif damage == "ohlc":
        bars[0] = replace(bars[0], high=bars[0].low)
    elif damage == "nan":
        bars[0] = replace(bars[0], close=D("NaN"))
    elif damage == "negative":
        bars[0] = replace(bars[0], volume=D(-1))
    else:
        bars[0] = replace(bars[0], open_time=bars[0].open_time + 1)
    with pytest.raises(ValueError):
        validate_bars(bars)


def test_csv_rejects_unclosed_bar(tmp_path):
    bars = synthetic_bars(30)
    path = tmp_path / "bars.csv"
    with path.open("w", newline="") as stream:
        rows = [b.record() for b in bars]
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    assert load_bars(path, as_of_ms=bars[-1].available_at) == bars
    with pytest.raises(ValueError, match="unclosed"):
        load_bars(path, as_of_ms=bars[-1].available_at - 1)
