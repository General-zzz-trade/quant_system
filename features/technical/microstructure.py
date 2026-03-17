"""Microstructure features — VWAP, order flow imbalance, volatility cone.

All functions follow the FeatureSeries pattern: return List[Optional[float]].
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

from features.types import Bar, FeatureSeries

from _quant_hotpath import (
    cpp_vwap as _cpp_vwap,
    cpp_order_flow_imbalance as _cpp_ofi,
    cpp_price_impact as _cpp_price_impact,
)


def vwap(bars: Sequence[Bar], window: int = 20) -> FeatureSeries:
    """Volume-weighted average price over a rolling window."""
    closes = [float(b.close) for b in bars]
    volumes = [float(b.volume) if b.volume is not None else 0.0 for b in bars]
    return list(_cpp_vwap(closes, volumes, window))


def order_flow_imbalance(bars: Sequence[Bar], window: int = 10) -> FeatureSeries:
    """Order flow imbalance proxy using bar direction and volume."""
    opens = [float(b.open) for b in bars]
    closes = [float(b.close) for b in bars]
    volumes = [float(b.volume) if b.volume is not None else 0.0 for b in bars]
    return list(_cpp_ofi(opens, closes, volumes, window))


def volatility_cone(
    returns: Sequence[Optional[float]],
    windows: Sequence[int] = (5, 10, 20, 60),
) -> dict[int, FeatureSeries]:
    """Realized volatility at multiple time horizons."""
    result: dict[int, FeatureSeries] = {}
    for w in windows:
        series: FeatureSeries = []
        for i in range(len(returns)):
            if i < w - 1 or returns[i] is None:
                series.append(None)
                continue

            window_rets = []
            for j in range(i - w + 1, i + 1):
                if returns[j] is not None:
                    window_rets.append(returns[j])

            if len(window_rets) < w // 2:
                series.append(None)
                continue

            mean = sum(r for r in window_rets if r is not None) / len(window_rets)
            var = sum((r - mean) ** 2 for r in window_rets if r is not None) / max(len(window_rets) - 1, 1)
            annual_vol = math.sqrt(var * 252)
            series.append(annual_vol)

        result[w] = series
    return result


def price_impact(bars: Sequence[Bar], window: int = 20) -> FeatureSeries:
    """Kyle's lambda proxy — price change per unit volume."""
    closes = [float(b.close) for b in bars]
    volumes = [float(b.volume) if b.volume is not None else 0.0 for b in bars]
    return list(_cpp_price_impact(closes, volumes, window))
