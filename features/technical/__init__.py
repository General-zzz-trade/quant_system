from __future__ import annotations

from math import log, sqrt
from typing import List, Optional, Sequence

from features.rolling import RollingWindow
from features.types import Bar, Bars, FeatureSeries

try:
    from features._quant_rolling import (
        cpp_sma,
        cpp_ema,
        cpp_returns,
        cpp_volatility,
        cpp_rsi,
        cpp_macd,
        cpp_bollinger_bands,
        cpp_atr,
        cpp_vwap,
        cpp_order_flow_imbalance,
        cpp_rolling_volatility,
        cpp_price_impact,
        cpp_ols,
    )
    _USING_CPP = True
except ImportError:
    _USING_CPP = False


def _closes(bars: Bars) -> List[float]:
    return [float(b.close) for b in bars]


def _highs(bars: Bars) -> List[float]:
    return [float(b.high) for b in bars]


def _lows(bars: Bars) -> List[float]:
    return [float(b.low) for b in bars]


def sma(values: Sequence[float], window: int) -> FeatureSeries:
    if _USING_CPP:
        return cpp_sma([float(v) for v in values], window)  # type: ignore[no-any-return]
    rw = RollingWindow(window)
    out: FeatureSeries = []
    for x in values:
        rw.push(float(x))
        out.append(rw.mean if rw.full else None)
    return out


def ema(values: Sequence[float], window: int) -> FeatureSeries:
    if _USING_CPP:
        return cpp_ema([float(v) for v in values], window)  # type: ignore[no-any-return]
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
    if _USING_CPP:
        return cpp_returns([float(v) for v in values], log_ret)  # type: ignore[no-any-return]
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
    if _USING_CPP:
        return cpp_volatility(list(ret_series), window)  # type: ignore[no-any-return]
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
    if _USING_CPP:
        return cpp_rsi([float(v) for v in values], window)  # type: ignore[no-any-return]
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
            if avg_gain is None or avg_loss is None:
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


def macd(
    values: Sequence[float],
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[FeatureSeries, FeatureSeries, FeatureSeries]:
    if _USING_CPP:
        return cpp_macd([float(v) for v in values], fast, slow, signal)  # type: ignore[no-any-return]
    if fast <= 0 or slow <= 0 or signal <= 0:
        raise ValueError("MACD windows must be positive")
    if fast >= slow:
        raise ValueError("fast window must be smaller than slow window")

    fast_ema = ema(values, fast)
    slow_ema = ema(values, slow)

    macd_line: FeatureSeries = []
    for f, s in zip(fast_ema, slow_ema):
        if f is None or s is None:
            macd_line.append(None)
        else:
            macd_line.append(f - s)

    macd_values = [v if v is not None else 0.0 for v in macd_line]
    signal_raw = ema(macd_values, signal)

    first_valid = next((i for i, v in enumerate(macd_line) if v is not None), len(macd_line))
    signal_line: FeatureSeries = []
    for i, s in enumerate(signal_raw):
        if i < first_valid + signal - 1:
            signal_line.append(None)
        else:
            signal_line.append(s)

    histogram: FeatureSeries = []
    for m, s in zip(macd_line, signal_line):
        if m is None or s is None:
            histogram.append(None)
        else:
            histogram.append(m - s)

    return macd_line, signal_line, histogram


def bollinger_bands(
    values: Sequence[float],
    window: int = 20,
    num_std: float = 2.0,
) -> tuple[FeatureSeries, FeatureSeries, FeatureSeries]:
    if _USING_CPP:
        return cpp_bollinger_bands([float(v) for v in values], window, num_std)  # type: ignore[no-any-return]
    if window <= 0:
        raise ValueError("window must be positive")
    if num_std <= 0:
        raise ValueError("num_std must be positive")

    rw = RollingWindow(window)
    upper: FeatureSeries = []
    middle: FeatureSeries = []
    lower: FeatureSeries = []

    for x in values:
        rw.push(float(x))
        if not rw.full:
            upper.append(None)
            middle.append(None)
            lower.append(None)
        else:
            mid = rw.mean
            std = rw.std
            if mid is None or std is None:
                upper.append(None)
                middle.append(None)
                lower.append(None)
            else:
                upper.append(mid + num_std * std)
                middle.append(mid)
                lower.append(mid - num_std * std)

    return upper, middle, lower


def atr(bars: Bars, window: int = 14) -> FeatureSeries:
    if _USING_CPP:
        highs = _highs(bars)
        lows = _lows(bars)
        closes = _closes(bars)
        return cpp_atr(highs, lows, closes, window)  # type: ignore[no-any-return]
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
