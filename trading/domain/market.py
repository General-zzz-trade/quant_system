"""Hourly bars -> causal 4h / daily aggregates, as pandas frames.

Every row carries ``available_at`` (ms): the close time at which the bar is
known. Aggregates keep only complete UTC-aligned groups, so a daily bar is
visible from 00:00 UTC of the next day and a 4h bar from its 4h boundary.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal
from typing import Protocol

import numpy as np
import pandas as pd

HOUR = 3_600_000
DAY = 24 * HOUR


class BarLike(Protocol):
    """A closed hourly bar (data.quality.causal_bars.Bar satisfies it)."""

    @property
    def open_time(self) -> int: ...

    @property
    def open(self) -> Decimal: ...

    @property
    def high(self) -> Decimal: ...

    @property
    def low(self) -> Decimal: ...

    @property
    def close(self) -> Decimal: ...

    @property
    def volume(self) -> Decimal: ...


def frame_from_bars(bars: Sequence[BarLike]) -> pd.DataFrame:
    return pd.DataFrame({
        "open_time": np.fromiter((b.open_time for b in bars), dtype=np.int64, count=len(bars)),
        "open": np.fromiter((float(b.open) for b in bars), dtype=float, count=len(bars)),
        "high": np.fromiter((float(b.high) for b in bars), dtype=float, count=len(bars)),
        "low": np.fromiter((float(b.low) for b in bars), dtype=float, count=len(bars)),
        "close": np.fromiter((float(b.close) for b in bars), dtype=float, count=len(bars)),
        "volume": np.fromiter((float(b.volume) for b in bars), dtype=float, count=len(bars)),
    }).assign(available_at=lambda f: f["open_time"] + HOUR)


def aggregate(frame: pd.DataFrame, hours: int) -> pd.DataFrame:
    """Complete UTC-aligned groups of ``hours`` consecutive hourly bars only."""
    span = hours * HOUR
    group = frame["open_time"] // span
    agg = frame.groupby(group, sort=True).agg(
        open_time=("open_time", "first"), open=("open", "first"), high=("high", "max"),
        low=("low", "min"), close=("close", "last"), volume=("volume", "sum"), count=("open_time", "size"))
    agg = agg[(agg["count"] == hours) & (agg["open_time"] % span == 0)].drop(columns="count")
    agg["available_at"] = agg["open_time"] + span
    return agg.reset_index(drop=True)


def funding_series(funding: Mapping[int, tuple[Decimal, Decimal]]) -> pd.Series:
    """Settled funding rates indexed by settlement time (ms)."""
    times = sorted(funding)
    return pd.Series([float(funding[t][0]) for t in times], index=pd.Index(times, dtype=np.int64), dtype=float)
