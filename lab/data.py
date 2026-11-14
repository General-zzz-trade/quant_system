"""Hourly bars -> causal 4h / daily aggregates, as pandas frames.

Every row carries ``available_at`` (ms): the close time at which the bar is
known. Aggregates keep only complete UTC-aligned groups, so a daily bar is
visible from 00:00 UTC of the next day and a 4h bar from its 4h boundary.
"""
from __future__ import annotations

from pathlib import Path

from data.quality.causal_bars import load_bars
from trading.domain.market import DAY, HOUR, aggregate, frame_from_bars, funding_series

__all__ = ["DAY", "HOUR", "aggregate", "frame_from_bars", "funding_series", "load_symbol"]


def load_symbol(data_dir: Path, symbol: str, *, start_ms: int | None = None):
    """(bars, funding dict, hourly frame) from the binance_public download."""
    from runner.offline.report import load_funding
    bars = load_bars(Path(data_dir) / f"{symbol}_1h.csv", start_ms=start_ms)
    funding = load_funding(Path(data_dir) / f"{symbol}_funding.csv")
    return bars, funding, frame_from_bars(bars)
