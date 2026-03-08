from __future__ import annotations

from typing import List, Optional, Sequence

from features.rolling import RollingWindow
from features.types import Bar, Bars, FeatureSeries

from _quant_hotpath import (
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


def _closes(bars: Bars) -> List[float]:
    return [float(b.close) for b in bars]


def _highs(bars: Bars) -> List[float]:
    return [float(b.high) for b in bars]


def _lows(bars: Bars) -> List[float]:
    return [float(b.low) for b in bars]


def sma(values: Sequence[float], window: int) -> FeatureSeries:
    return cpp_sma([float(v) for v in values], window)  # type: ignore[no-any-return]


def ema(values: Sequence[float], window: int) -> FeatureSeries:
    return cpp_ema([float(v) for v in values], window)  # type: ignore[no-any-return]


def returns(values: Sequence[float], *, log_ret: bool = False) -> FeatureSeries:
    return cpp_returns([float(v) for v in values], log_ret)  # type: ignore[no-any-return]


def log_returns(values: Sequence[float]) -> FeatureSeries:
    return returns(values, log_ret=True)


def volatility(ret_series: Sequence[Optional[float]], window: int) -> FeatureSeries:
    return cpp_volatility(list(ret_series), window)  # type: ignore[no-any-return]


def rsi(values: Sequence[float], window: int = 14) -> FeatureSeries:
    return cpp_rsi([float(v) for v in values], window)  # type: ignore[no-any-return]


def macd(
    values: Sequence[float],
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[FeatureSeries, FeatureSeries, FeatureSeries]:
    return cpp_macd([float(v) for v in values], fast, slow, signal)  # type: ignore[no-any-return]


def bollinger_bands(
    values: Sequence[float],
    window: int = 20,
    num_std: float = 2.0,
) -> tuple[FeatureSeries, FeatureSeries, FeatureSeries]:
    return cpp_bollinger_bands([float(v) for v in values], window, num_std)  # type: ignore[no-any-return]


def atr(bars: Bars, window: int = 14) -> FeatureSeries:
    highs = _highs(bars)
    lows = _lows(bars)
    closes = _closes(bars)
    return cpp_atr(highs, lows, closes, window)  # type: ignore[no-any-return]
