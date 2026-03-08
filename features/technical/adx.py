"""Average Directional Index (ADX) feature — delegates to Rust."""
from __future__ import annotations

from typing import Optional

from _quant_hotpath import rust_adx
from features.types import Bars, FeatureSeries


def adx(bars: Bars, window: int = 14) -> FeatureSeries:
    """Compute ADX (Average Directional Index).

    Uses Wilder's smoothing via Rust kernel.
    Returns None for the first 2*window bars (warmup period).
    """
    n = len(bars)
    if n < 2 or window <= 0:
        return [None] * n

    highs = [float(b.high) for b in bars]
    lows = [float(b.low) for b in bars]
    closes = [float(b.close) for b in bars]

    raw = rust_adx(highs, lows, closes, window)
    # Convert NaN to None for FeatureSeries compatibility
    result: FeatureSeries = []
    for v in raw:
        if v != v:  # NaN check
            result.append(None)
        else:
            result.append(v)
    return result
