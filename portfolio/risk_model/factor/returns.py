# portfolio/risk_model/factor/returns.py
"""Factor return estimation (cross-sectional regression proxy)."""
from __future__ import annotations

from typing import Mapping, Sequence


def estimate_market_factor(
    symbols: Sequence[str],
    returns: Mapping[str, Sequence[float]],
) -> list[float]:
    """市场因子 = 等权平均收益率。"""
    n_obs = min(len(returns.get(s, ())) for s in symbols) if symbols else 0
    result = []
    for t in range(n_obs):
        avg = sum(returns[s][t] for s in symbols) / len(symbols)
        result.append(avg)
    return result


def estimate_momentum_factor(
    symbols: Sequence[str],
    returns: Mapping[str, Sequence[float]],
    lookback: int = 20,
) -> list[float]:
    """动量因子 = 过去N期累计收益的截面多空组合。"""
    n_obs = min(len(returns.get(s, ())) for s in symbols) if symbols else 0
    result = []
    for t in range(n_obs):
        if t < lookback:
            result.append(0.0)
            continue
        cum = {}
        for s in symbols:
            cum[s] = sum(returns[s][t - lookback:t])
        sorted_syms = sorted(cum, key=lambda s: cum[s])
        n = len(sorted_syms)
        top = sorted_syms[n // 2:]
        bottom = sorted_syms[:n // 2]
        long_ret = sum(returns[s][t] for s in top) / max(len(top), 1)
        short_ret = sum(returns[s][t] for s in bottom) / max(len(bottom), 1)
        result.append(long_ret - short_ret)
    return result
