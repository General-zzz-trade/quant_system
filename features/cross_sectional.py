"""Cross-sectional features — momentum ranking, relative strength, beta.

These features compare assets against each other (cross-section)
rather than looking at single-asset time series.
"""
from __future__ import annotations

import math
from typing import Dict, List, Mapping, Optional, Sequence

from features.types import FeatureSeries

from _quant_hotpath import (
    cpp_momentum_rank as _cpp_momentum_rank,
    cpp_rolling_beta as _cpp_rolling_beta,
    cpp_relative_strength as _cpp_relative_strength,
)

_NAN = float("nan")


def _to_nan(seq: Sequence[Optional[float]]) -> List[float]:
    return [_NAN if v is None else float(v) for v in seq]


def _from_nan(seq: List[float]) -> FeatureSeries:
    return [None if math.isnan(v) else v for v in seq]


def momentum_rank(
    returns: Mapping[str, Sequence[Optional[float]]],
    lookback: int = 20,
) -> Dict[str, FeatureSeries]:
    """Cross-sectional momentum rank (0=worst, 1=best)."""
    symbols = list(returns.keys())
    n_periods = min(len(v) for v in returns.values()) if returns else 0

    if n_periods > 0:
        matrix = [_to_nan(returns[s]) for s in symbols]
        rank_matrix = _cpp_momentum_rank(matrix, lookback)
        return {s: _from_nan(rank_matrix[i]) for i, s in enumerate(symbols)}

    return {s: [] for s in symbols}


def relative_strength(
    target_returns: Sequence[Optional[float]],
    benchmark_returns: Sequence[Optional[float]],
    window: int = 20,
) -> FeatureSeries:
    """Relative strength vs benchmark over a rolling window."""
    return _from_nan(_cpp_relative_strength(_to_nan(target_returns), _to_nan(benchmark_returns), window))


def rolling_beta(
    asset_returns: Sequence[Optional[float]],
    market_returns: Sequence[Optional[float]],
    window: int = 60,
) -> FeatureSeries:
    """Rolling beta relative to market. beta = cov(asset, market) / var(market)."""
    return _from_nan(_cpp_rolling_beta(_to_nan(asset_returns), _to_nan(market_returns), window))
