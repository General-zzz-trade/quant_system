# portfolio/risk_model/factor/specific_risk.py
"""Specific (idiosyncratic) risk estimation."""
from __future__ import annotations

from typing import Mapping, Sequence

from portfolio.risk_model.factor.exposure import compute_beta


def estimate_specific_risk(
    symbols: Sequence[str],
    returns: Mapping[str, Sequence[float]],
    factor_returns: Mapping[str, Sequence[float]],
    exposures: Mapping[str, Mapping[str, float]],
) -> dict[str, float]:
    """估计每个资产的特异性风险（残差方差）。"""
    factors = list(factor_returns.keys())
    n_obs = min(len(returns.get(s, ())) for s in symbols) if symbols else 0
    if n_obs < 2:
        return {s: 0.0 for s in symbols}

    f_n_obs = min(len(factor_returns[f]) for f in factors) if factors else 0
    obs = min(n_obs, f_n_obs)

    result: dict[str, float] = {}
    for s in symbols:
        betas = exposures.get(s, {})
        residuals = []
        for t in range(obs):
            predicted = sum(
                betas.get(f, 0.0) * factor_returns[f][t] for f in factors
            )
            residuals.append(returns[s][t] - predicted)
        if len(residuals) < 2:
            result[s] = 0.0
        else:
            mean_r = sum(residuals) / len(residuals)
            result[s] = sum((r - mean_r) ** 2 for r in residuals) / (len(residuals) - 1)
    return result
