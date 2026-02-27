"""Microstructure features — VWAP, order flow imbalance, volatility cone.

All functions follow the FeatureSeries pattern: return List[Optional[float]].
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence

from features.types import Bar, FeatureSeries

try:
    from features._quant_rolling import (
        cpp_vwap as _cpp_vwap,
        cpp_order_flow_imbalance as _cpp_ofi,
        cpp_rolling_volatility as _cpp_rolling_vol,
        cpp_price_impact as _cpp_price_impact,
    )
    _USING_CPP = True
except ImportError:
    _USING_CPP = False


def vwap(bars: Sequence[Bar], window: int = 20) -> FeatureSeries:
    """Volume-weighted average price over a rolling window.

    VWAP = sum(close * volume) / sum(volume) over window.
    """
    if _USING_CPP:
        closes = [float(b.close) for b in bars]
        volumes = [float(b.volume) if b.volume is not None else 0.0 for b in bars]
        return _cpp_vwap(closes, volumes, window)

    result: FeatureSeries = []
    for i in range(len(bars)):
        if i < window - 1:
            result.append(None)
            continue

        total_pv = 0.0
        total_v = 0.0
        for j in range(i - window + 1, i + 1):
            b = bars[j]
            v = float(b.volume) if b.volume is not None else 0.0
            c = float(b.close)
            total_pv += c * v
            total_v += v

        if total_v > 0:
            result.append(total_pv / total_v)
        else:
            result.append(None)
    return result


def order_flow_imbalance(bars: Sequence[Bar], window: int = 10) -> FeatureSeries:
    """Order flow imbalance proxy using bar direction and volume.

    OFI = sum(signed_volume) / sum(abs_volume) over window.
    Positive = buying pressure, negative = selling pressure.
    Range: [-1, 1].
    """
    if _USING_CPP:
        opens = [float(b.open) for b in bars]
        closes = [float(b.close) for b in bars]
        volumes = [float(b.volume) if b.volume is not None else 0.0 for b in bars]
        return _cpp_ofi(opens, closes, volumes, window)

    result: FeatureSeries = []

    signed_volumes: List[float] = []
    for b in bars:
        v = float(b.volume) if b.volume is not None else 0.0
        direction = 1.0 if float(b.close) >= float(b.open) else -1.0
        signed_volumes.append(direction * v)

    for i in range(len(bars)):
        if i < window - 1:
            result.append(None)
            continue

        window_sv = signed_volumes[i - window + 1: i + 1]
        total_abs = sum(abs(sv) for sv in window_sv)
        if total_abs > 0:
            result.append(sum(window_sv) / total_abs)
        else:
            result.append(0.0)
    return result


def volatility_cone(
    returns: Sequence[Optional[float]],
    windows: Sequence[int] = (5, 10, 20, 60),
) -> dict[int, FeatureSeries]:
    """Realized volatility at multiple time horizons.

    Returns a dict of window -> annualized vol series.
    Useful for comparing current vol to historical vol distribution.
    """
    valid = [r for r in returns if r is not None]

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

            mean = sum(window_rets) / len(window_rets)
            var = sum((r - mean) ** 2 for r in window_rets) / max(len(window_rets) - 1, 1)
            annual_vol = math.sqrt(var * 252)
            series.append(annual_vol)

        result[w] = series
    return result


def price_impact(bars: Sequence[Bar], window: int = 20) -> FeatureSeries:
    """Kyle's lambda proxy — price change per unit volume.

    Measures market impact: higher values = less liquid.
    """
    if _USING_CPP:
        closes = [float(b.close) for b in bars]
        volumes = [float(b.volume) if b.volume is not None else 0.0 for b in bars]
        return _cpp_price_impact(closes, volumes, window)

    result: FeatureSeries = []
    for i in range(len(bars)):
        if i < window:
            result.append(None)
            continue

        price_changes = []
        volumes = []
        for j in range(i - window + 1, i + 1):
            if j > 0:
                dc = abs(float(bars[j].close) - float(bars[j-1].close))
                v = float(bars[j].volume) if bars[j].volume is not None else 0.0
                price_changes.append(dc)
                volumes.append(v)

        total_v = sum(volumes)
        if total_v > 0:
            result.append(sum(price_changes) / total_v)
        else:
            result.append(None)
    return result
