"""Strict UTC bar input and closed-bar aggregation for offline experiments."""
from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from decimal import Decimal
from pathlib import Path

HOUR = 3_600_000


@dataclass(frozen=True)
class Bar:
    open_time: int
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    interval_ms: int = HOUR

    @property
    def available_at(self) -> int:
        return self.open_time + self.interval_ms

    def record(self) -> dict:
        return {k: str(v) if isinstance(v, Decimal) else v for k, v in asdict(self).items()}


def validate_bars(bars: list[Bar], interval_ms: int = HOUR) -> None:
    if not bars:
        raise ValueError("empty dataset")
    previous = None
    for i, bar in enumerate(bars):
        if bar.interval_ms != interval_ms or bar.open_time < 0 or bar.open_time % interval_ms:
            raise ValueError(f"bar {i}: UTC interval alignment")
        if previous is not None and bar.open_time != previous + interval_ms:
            raise ValueError(f"bar {i}: gap, duplicate or out-of-order timestamp")
        values = (bar.open, bar.high, bar.low, bar.close, bar.volume)
        if not all(v.is_finite() for v in values):
            raise ValueError(f"bar {i}: non-finite value")
        if min(values[:4]) <= 0 or bar.volume < 0:
            raise ValueError(f"bar {i}: non-positive price or negative volume")
        if bar.low > min(bar.open, bar.close) or bar.high < max(bar.open, bar.close) or bar.low > bar.high:
            raise ValueError(f"bar {i}: invalid OHLC")
        previous = bar.open_time


def load_bars(path: Path, *, as_of_ms: int | None = None) -> list[Bar]:
    """CSV timestamps are UTC epoch milliseconds. Never sort/fill broken input."""
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    bars = [Bar(int(r["open_time"]), *(Decimal(r[k]) for k in ("open", "high", "low", "close", "volume")))
            for r in rows]
    validate_bars(bars)
    if as_of_ms is not None and bars[-1].available_at > as_of_ms:
        raise ValueError("dataset includes an unclosed bar")
    return bars


def aggregate_closed(bars: list[Bar], hours: int = 4) -> list[Bar]:
    """Only complete UTC-aligned groups survive; edge fragments are discarded."""
    validate_bars(bars)
    if hours < 1:
        raise ValueError("hours must be positive")
    groups: dict[int, list[Bar]] = {}
    interval = hours * HOUR
    for bar in bars:
        groups.setdefault(bar.open_time // interval * interval, []).append(bar)
    output = []
    for start, group in groups.items():
        if len(group) != hours or group[0].open_time != start:
            continue
        output.append(Bar(start, group[0].open, max(b.high for b in group), min(b.low for b in group),
                          group[-1].close, sum((b.volume for b in group), Decimal(0)), interval))
    return output


def align_closed_signals(signals, source_open_times, target_available_times, *, hours=4):
    """Legacy pandas research helper: signals become visible at source CLOSE."""
    import pandas as pd
    source = pd.DatetimeIndex(pd.to_datetime(source_open_times, utc=True)) + pd.Timedelta(hours=hours)
    target = pd.DatetimeIndex(pd.to_datetime(target_available_times, utc=True))
    if (not source.is_monotonic_increasing or source.has_duplicates
            or not target.is_monotonic_increasing or target.has_duplicates):
        raise ValueError("signal timestamps must be sorted and unique")
    return pd.Series(signals, index=source).reindex(target, method="ffill").fillna(0).to_numpy()


def fingerprint(value) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def dataset_manifest(bars: list[Bar], *, synthetic: bool = False) -> dict:
    validate_bars(bars)
    return {"sha256": fingerprint([b.record() for b in bars]), "rows": len(bars),
            "start_ms": bars[0].open_time, "end_ms": bars[-1].available_at,
            "timestamp_semantics": "UTC open_time milliseconds; available at open_time + 1h",
            "synthetic": synthetic}
