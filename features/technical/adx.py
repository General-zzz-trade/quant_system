"""Average Directional Index (ADX) feature."""
from __future__ import annotations

from typing import List, Optional, Sequence

from features.types import Bars, FeatureSeries


def adx(bars: Bars, window: int = 14) -> FeatureSeries:
    """Compute ADX (Average Directional Index).

    Uses Wilder's smoothing: smoothed = prev - (prev/window) + current.
    Returns None for the first 2*window bars (warmup period).
    """
    n = len(bars)
    if n < 2 or window <= 0:
        return [None] * n

    # Step 1: compute +DM, -DM, TR for each bar
    plus_dm: List[float] = [0.0]
    minus_dm: List[float] = [0.0]
    tr_list: List[float] = [bars[0].high - bars[0].low]

    for i in range(1, n):
        h, l, pc = bars[i].high, bars[i].low, bars[i - 1].close
        up_move = h - bars[i - 1].high
        down_move = bars[i - 1].low - l
        plus_dm.append(up_move if up_move > down_move and up_move > 0 else 0.0)
        minus_dm.append(down_move if down_move > up_move and down_move > 0 else 0.0)
        tr_list.append(max(h - l, abs(h - pc), abs(l - pc)))

    # Step 2: Wilder's smoothing for +DM, -DM, TR
    out: FeatureSeries = [None] * n
    if n <= window:
        return out

    sm_plus = sum(plus_dm[:window])
    sm_minus = sum(minus_dm[:window])
    sm_tr = sum(tr_list[:window])

    # Step 3: compute +DI, -DI, DX
    dx_values: List[float] = []

    def _di_dx(sp: float, sm: float, st: float) -> Optional[float]:
        if st == 0:
            return None
        plus_di = 100.0 * sp / st
        minus_di = 100.0 * sm / st
        denom = plus_di + minus_di
        if denom == 0:
            return 0.0
        return 100.0 * abs(plus_di - minus_di) / denom

    dx = _di_dx(sm_plus, sm_minus, sm_tr)
    if dx is not None:
        dx_values.append(dx)

    for i in range(window, n):
        sm_plus = sm_plus - sm_plus / window + plus_dm[i]
        sm_minus = sm_minus - sm_minus / window + minus_dm[i]
        sm_tr = sm_tr - sm_tr / window + tr_list[i]
        dx = _di_dx(sm_plus, sm_minus, sm_tr)
        if dx is not None:
            dx_values.append(dx)

    # Step 4: ADX = smoothed average of DX over window
    if len(dx_values) < window:
        return out

    adx_val = sum(dx_values[:window]) / window
    # First ADX value at index 2*window - 1
    start_idx = 2 * window - 1
    if start_idx < n:
        out[start_idx] = adx_val

    for j in range(window, len(dx_values)):
        adx_val = (adx_val * (window - 1) + dx_values[j]) / window
        idx = window + j
        if idx < n:
            out[idx] = adx_val

    return out
