"""Incremental tracker wrappers and helper functions for enriched_computer.

Extracted from enriched_computer.py to keep file sizes manageable.
Re-exported from enriched_computer.py for backward compatibility.
"""
from __future__ import annotations

from math import sqrt
from typing import Deque, Dict, Optional

from _quant_hotpath import PyAdxTracker, PyAtrTracker, PyEmaTracker, PyRsiTracker


class _EMA:
    """Incremental EMA tracker -- Rust-backed via PyEmaTracker."""

    __slots__ = ("_inner",)

    def __init__(self, span: int) -> None:
        self._inner = PyEmaTracker(span)

    def push(self, x: float) -> None:
        self._inner.push(x)

    @property
    def value(self) -> Optional[float]:
        return self._inner.value()

    @property
    def ready(self) -> bool:
        return self._inner.ready()


class _RSITracker:
    """Incremental RSI -- Rust-backed via PyRsiTracker."""

    __slots__ = ("_inner",)

    def __init__(self, period: int) -> None:
        self._inner = PyRsiTracker(period)

    def push(self, close: float) -> None:
        self._inner.push(close)

    @property
    def value(self) -> Optional[float]:
        return self._inner.value()


class _ATRTracker:
    """Incremental ATR -- Rust-backed via PyAtrTracker."""

    __slots__ = ("_inner",)

    def __init__(self, period: int) -> None:
        self._inner = PyAtrTracker(period)

    def push(self, high: float, low: float, close: float) -> None:
        self._inner.push(high, low, close)

    @property
    def value(self) -> Optional[float]:
        return self._inner.value()

    def normalized(self, close: float) -> float:
        return self._inner.normalized(close)


class _ADXTracker:
    """Incremental ADX -- Rust-backed via PyAdxTracker."""

    __slots__ = ("_inner",)

    def __init__(self, period: int = 14) -> None:
        self._inner = PyAdxTracker(period)

    def push(self, high: float, low: float, close: float) -> None:
        self._inner.push(high, low, close)

    @property
    def value(self) -> Optional[float]:
        return self._inner.value()


def _symbol_aliases(symbol: str) -> tuple[str, ...]:
    """Return normalized aliases for a symbol (e.g. ETHUSDT -> [ETHUSDT, ETH])."""
    upper = str(symbol).upper()
    aliases = [upper]
    if upper.endswith("USDT"):
        aliases.append(upper[:-4])
    else:
        aliases.append(f"{upper}USDT")
    return tuple(dict.fromkeys(aliases))


def _resolve_multi_dominance_pairs(symbol: str) -> tuple[tuple[str, str], ...]:
    """Resolve dominance pair config for a symbol."""
    from features.enriched_computer import _MULTI_DOMINANCE_PAIRS
    for alias in _symbol_aliases(symbol):
        pairs = _MULTI_DOMINANCE_PAIRS.get(alias)
        if pairs:
            return pairs
    return ()


def _build_multi_dominance_ratios(
    symbol: str,
    close: float,
    reference_closes: Optional[Dict[str, float]],
) -> Dict[str, float]:
    """Build dominance ratio dict for multi-pair features."""
    if close <= 0 or not reference_closes:
        return {}
    normalized_refs = {
        str(ref_symbol).upper(): ref_close
        for ref_symbol, ref_close in reference_closes.items()
        if ref_close is not None
    }
    ratios: Dict[str, float] = {}
    for ref_symbol, prefix in _resolve_multi_dominance_pairs(symbol):
        ref_close = _lookup_reference_close(normalized_refs, ref_symbol)
        if ref_close is not None and ref_close > 0:
            ratios[prefix] = close / ref_close
    return ratios


def _lookup_reference_close(reference_closes: Dict[str, float], ref_symbol: str) -> Optional[float]:
    """Look up a reference close price with alias resolution."""
    for alias in _symbol_aliases(ref_symbol):
        ref_close = reference_closes.get(alias)
        if ref_close is not None:
            return ref_close
    return None


def _window_zscore(values: Deque[float], window: int) -> Optional[float]:
    """Compute z-score of the last value in a deque over the given window."""
    if len(values) < window:
        return None
    window_vals = list(values)[-window:]
    mean = sum(window_vals) / len(window_vals)
    var = sum((value - mean) ** 2 for value in window_vals) / len(window_vals)
    std = sqrt(var) if var > 0 else 0.0
    return (window_vals[-1] - mean) / std if std > 1e-8 else 0.0
