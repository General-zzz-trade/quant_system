"""Hourly bars -> causal 4h / daily aggregates, as pandas frames.

Every row carries ``available_at`` (ms): the close time at which the bar is
known. Aggregates keep only complete UTC-aligned groups, so a daily bar is
visible from 00:00 UTC of the next day and a 4h bar from its 4h boundary.
"""
from __future__ import annotations

import csv
from decimal import Decimal as D
from pathlib import Path

from data.quality.causal_bars import load_bars
from trading.domain.market import DAY, HOUR, aggregate, frame_from_bars, funding_series

__all__ = ["DAY", "HOUR", "aggregate", "frame_from_bars", "funding_series", "load_funding", "load_symbol"]


def load_funding(path):
    """{settlement ms: (rate, mark price)} from a binance_public funding CSV (moved unchanged from
    ``runner.offline.report`` when the ML offline replay was deleted, M4 2026-11-15)."""
    result = {}
    with Path(path).open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            at, rate, mark = int(row["funding_time"]), D(row["rate"]), D(row["mark_price"])
            observed = int(row.get("observed_funding_time") or at)
            if not 0 <= observed - at <= 1000:
                raise ValueError("funding timestamp normalization exceeds the declared 1-second tolerance")
            if at in result or at % (8 * HOUR) or not rate.is_finite() or not mark.is_finite() or mark <= 0:
                raise ValueError("invalid/duplicate funding record; this runner requires UTC 8h settlements")
            result[at] = (rate, mark)
    return result


def load_symbol(data_dir: Path, symbol: str, *, start_ms: int | None = None):
    """(bars, funding dict, hourly frame) from the binance_public download."""
    bars = load_bars(Path(data_dir) / f"{symbol}_1h.csv", start_ms=start_ms)
    funding = load_funding(Path(data_dir) / f"{symbol}_funding.csv")
    return bars, funding, frame_from_bars(bars)
