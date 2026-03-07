# portfolio/risk_model/factor/specific_risk.py
"""Specific (idiosyncratic) risk estimation."""
from __future__ import annotations

from typing import Mapping, Sequence

from portfolio.risk_model.factor.exposure import compute_beta

try:
    from _quant_hotpath import cpp_estimate_specific_risk as _cpp_estimate_specific_risk
    _USING_CPP = True
except ImportError:
    _USING_CPP = False


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

    if _USING_CPP and obs >= 2 and factors:
        a_mat = [list(returns[s][:obs]) for s in symbols]
        f_mat = [list(factor_returns[f][:obs]) for f in factors]
        exp_mat = [[exposures.get(s, {}).get(f, 0.0) for f in factors] for s in symbols]
        sr_vec = _cpp_estimate_specific_risk(a_mat, f_mat, exp_mat)
        return {s: sr_vec[i] for i, s in enumerate(symbols)}

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
