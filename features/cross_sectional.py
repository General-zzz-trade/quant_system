"""Cross-sectional features — momentum ranking, relative strength, beta.

These features compare assets against each other (cross-section)
rather than looking at single-asset time series.
"""
from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence

from features.types import FeatureSeries


def momentum_rank(
    returns: Mapping[str, Sequence[Optional[float]]],
    lookback: int = 20,
) -> Dict[str, FeatureSeries]:
    """Cross-sectional momentum rank (0=worst, 1=best).

    For each time step, ranks all symbols by their lookback-period return.
    """
    symbols = list(returns.keys())
    n_periods = min(len(v) for v in returns.values()) if returns else 0
    result: Dict[str, FeatureSeries] = {s: [] for s in symbols}

    for i in range(n_periods):
        if i < lookback:
            for s in symbols:
                result[s].append(None)
            continue

        # Compute cumulative return over lookback for each symbol
        cum_returns: Dict[str, float] = {}
        for s in symbols:
            rets = returns[s][i - lookback + 1: i + 1]
            valid = [r for r in rets if r is not None]
            if len(valid) >= lookback // 2:
                cum_ret = 1.0
                for r in valid:
                    cum_ret *= (1 + r)
                cum_returns[s] = cum_ret - 1.0

        if len(cum_returns) < 2:
            for s in symbols:
                result[s].append(None)
            continue

        # Rank: 0 = worst, 1 = best
        sorted_syms = sorted(cum_returns.keys(), key=lambda s: cum_returns[s])
        n = len(sorted_syms)
        for rank_idx, s in enumerate(sorted_syms):
            result[s].append(rank_idx / max(n - 1, 1))

        for s in symbols:
            if s not in cum_returns:
                result[s].append(None)

    return result


def relative_strength(
    target_returns: Sequence[Optional[float]],
    benchmark_returns: Sequence[Optional[float]],
    window: int = 20,
) -> FeatureSeries:
    """Relative strength vs benchmark over a rolling window.

    RS = cumulative_return(target) / cumulative_return(benchmark)
    Values > 1 = outperforming, < 1 = underperforming.
    """
    n = min(len(target_returns), len(benchmark_returns))
    result: FeatureSeries = []

    for i in range(n):
        if i < window - 1:
            result.append(None)
            continue

        t_cum = 1.0
        b_cum = 1.0
        valid = True
        for j in range(i - window + 1, i + 1):
            tr = target_returns[j]
            br = benchmark_returns[j]
            if tr is None or br is None:
                valid = False
                break
            t_cum *= (1 + tr)
            b_cum *= (1 + br)

        if valid and b_cum != 0:
            result.append(t_cum / b_cum)
        else:
            result.append(None)

    return result


def rolling_beta(
    asset_returns: Sequence[Optional[float]],
    market_returns: Sequence[Optional[float]],
    window: int = 60,
) -> FeatureSeries:
    """Rolling beta relative to market.

    beta = cov(asset, market) / var(market)
    """
    n = min(len(asset_returns), len(market_returns))
    result: FeatureSeries = []

    for i in range(n):
        if i < window - 1:
            result.append(None)
            continue

        a_vals: List[float] = []
        m_vals: List[float] = []
        for j in range(i - window + 1, i + 1):
            a = asset_returns[j]
            m = market_returns[j]
            if a is not None and m is not None:
                a_vals.append(a)
                m_vals.append(m)

        if len(a_vals) < window // 2:
            result.append(None)
            continue

        a_mean = sum(a_vals) / len(a_vals)
        m_mean = sum(m_vals) / len(m_vals)

        cov = sum((a - a_mean) * (m - m_mean) for a, m in zip(a_vals, m_vals)) / len(a_vals)
        var_m = sum((m - m_mean) ** 2 for m in m_vals) / len(m_vals)

        if var_m > 0:
            result.append(cov / var_m)
        else:
            result.append(None)

    return result
