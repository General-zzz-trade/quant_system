"""Feature extraction utilities.

This module is intentionally lightweight and dependency-free.
It provides a small set of technical features and a simple in-memory store.
"""

from .types import Bar, Bars, FeatureName, FeatureSeries
from .technical import (
    sma,
    ema,
    rsi,
    atr,
    returns,
    log_returns,
    volatility,
)
__all__ = [
    "Bar",
    "Bars",
    "FeatureName",
    "FeatureSeries",
    "sma",
    "ema",
    "rsi",
    "atr",
    "returns",
    "log_returns",
    "volatility",
]
