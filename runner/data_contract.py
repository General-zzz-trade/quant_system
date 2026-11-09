"""Freshness and causal boundaries for the existing batch prediction inputs."""
from __future__ import annotations

import csv
import math
from pathlib import Path
import time
from typing import Any


def inspect_batch_inputs(symbol: str, root: Path, *, as_of_ms: int | None = None,
                         now_ms: int | None = None) -> dict[str, Any]:
    """Validate files used by batch_feature_engine; never modify input data.

    MarketEvent.ts and the training CSV use candle OPEN time. A closed hourly
    candle must be the most recent closed hour; OI/LS observations must be at or
    before that time. No fresh cache can vouch for an unrelated stale CSV.
    """
    now = int(time.time() * 1000) if now_ms is None else now_ms
    at = now // 3_600_000 * 3_600_000 - 3_600_000 if as_of_ms is None else as_of_ms
    result: dict[str, Any] = {"ready": False, "as_of_ms": at, "reason": "", "signature": []}
    try:
        if (type(at) is not int or at <= 0 or at % 3_600_000
                or not 3_600_000 <= now - at <= 3_900_000):
            raise ValueError("Batch candle must be the latest closed hourly candle")
        specifications = (
            (f"{symbol}_1h.csv", "open_time", "close", 0),
            (f"{symbol}_open_interest.csv", "timestamp", "sum_open_interest", 3_900_000),
            (f"{symbol}_ls_ratio.csv", "timestamp", "long_short_ratio", 3_900_000),
        )
        for filename, timestamp_key, value_key, max_age in specifications:
            path = root / filename
            if path.is_symlink() or not path.is_file():
                raise ValueError("Batch input is missing or is a symbolic link: " + filename)
            before = path.stat()
            points = {}
            with path.open(newline="") as stream:
                for row in csv.DictReader(stream):
                    timestamp, value = float(row[timestamp_key]), float(row[value_key])
                    if not math.isfinite(timestamp) or not timestamp.is_integer() or timestamp <= 0:
                        raise ValueError("Invalid batch observation timestamp")
                    timestamp = int(timestamp)
                    if timestamp > at:
                        continue
                    if not math.isfinite(value) or value <= 0:
                        raise ValueError("Invalid batch observation value")
                    if timestamp in points and points[timestamp] != value:
                        raise ValueError("Conflicting batch observations at the same timestamp")
                    points[timestamp] = value
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError("Batch input changed during validation")
            if not points or not 0 <= at - max(points) <= max_age:
                raise ValueError("Batch input has no fresh causal observation: " + filename)
            result["signature"].append((filename, after.st_size, after.st_mtime_ns))
        result["ready"] = True
    except (OSError, ValueError, KeyError, TypeError, OverflowError) as exc:
        result["reason"] = str(exc)
    return result
