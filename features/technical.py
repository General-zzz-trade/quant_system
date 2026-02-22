from __future__ import annotations

from math import log, sqrt
from typing import List, Optional, Sequence

from .rolling import RollingWindow
from .types import Bar, Bars, FeatureSeries


def _closes(bars: Bars) -> List[float]:
    return [float(b.close) for b in bars]


def _highs(bars: Bars) -> List[float]:
    return [float(b.high) for b in bars]


def _lows(bars: Bars) -> List[float]:
    return [float(b.low) for b in bars]


def sma(values: Sequence[float], window: int) -> FeatureSeries:
    rw = RollingWindow(window)
    out: FeatureSeries = []
    for x in values:
        rw.push(float(x))
        out.append(rw.mean if rw.full else None)
    return out


def ema(values: Sequence[float], window: int) -> FeatureSeries:
    if window <= 0:
        raise ValueError("window must be positive")

    alpha = 2.0 / (window + 1.0)
    out: FeatureSeries = []
    prev: Optional[float] = None
    for i, x0 in enumerate(values):
        x = float(x0)
        if prev is None:
            prev = x
        else:
            prev = alpha * x + (1.0 - alpha) * prev
        out.append(prev)
    return out


def returns(values: Sequence[float], *, log_ret: bool = False) -> FeatureSeries:
    out: FeatureSeries = [None] * len(values)
    for i in range(1, len(values)):
        p0 = float(values[i - 1])
        p1 = float(values[i])
        if p0 == 0.0:
            out[i] = None
            continue
        r = p1 / p0
        out[i] = log(r) if log_ret else (r - 1.0)
    return out


def log_returns(values: Sequence[float]) -> FeatureSeries:
    return returns(values, log_ret=True)


def volatility(ret_series: Sequence[Optional[float]], window: int) -> FeatureSeries:
    rw = RollingWindow(window)
    out: FeatureSeries = []
    for r in ret_series:
        if r is None:
            rw.push(0.0)
        else:
            rw.push(float(r))
        out.append(rw.std if rw.full else None)
    return out


def rsi(values: Sequence[float], window: int = 14) -> FeatureSeries:
    if window <= 0:
        raise ValueError("window must be positive")

    out: FeatureSeries = [None] * len(values)
    avg_gain: Optional[float] = None
    avg_loss: Optional[float] = None

    for i in range(1, len(values)):
        change = float(values[i]) - float(values[i - 1])
        gain = max(change, 0.0)
        loss = max(-change, 0.0)

        if i < window:
            # warmup
            if avg_gain is None:
                avg_gain = 0.0
                avg_loss = 0.0
            avg_gain += gain
            avg_loss += loss
            continue

        if i == window:
            assert avg_gain is not None and avg_loss is not None
            avg_gain = avg_gain / window
            avg_loss = avg_loss / window
        else:
            assert avg_gain is not None and avg_loss is not None
            avg_gain = (avg_gain * (window - 1) + gain) / window
            avg_loss = (avg_loss * (window - 1) + loss) / window

        if avg_loss == 0.0:
            out[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            out[i] = 100.0 - (100.0 / (1.0 + rs))

    return out


def atr(bars: Bars, window: int = 14) -> FeatureSeries:
    if window <= 0:
        raise ValueError("window must be positive")

    highs = _highs(bars)
    lows = _lows(bars)
    closes = _closes(bars)

    trs: List[float] = []
    for i in range(len(bars)):
        if i == 0:
            tr = highs[i] - lows[i]
        else:
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )
        trs.append(tr)

    # Wilder's smoothing
    out: FeatureSeries = [None] * len(trs)
    atr_prev: Optional[float] = None

    for i, tr in enumerate(trs):
        if i < window:
            if atr_prev is None:
                atr_prev = 0.0
            atr_prev += tr
            continue
        if i == window:
            assert atr_prev is not None
            atr_prev = atr_prev / window
            out[i] = atr_prev
            continue
        assert atr_prev is not None
        atr_prev = (atr_prev * (window - 1) + tr) / window
        out[i] = atr_prev

    return out
