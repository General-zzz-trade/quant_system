# portfolio/risk_model/factor/covariance.py
"""Factor covariance estimation."""
from __future__ import annotations

from typing import Mapping, Sequence


def estimate_factor_covariance(
    factor_returns: Mapping[str, Sequence[float]],
) -> dict[str, dict[str, float]]:
    """估计因子收益率的协方差矩阵。"""
    factors = list(factor_returns.keys())
    n_obs = min(len(factor_returns[f]) for f in factors) if factors else 0

    if n_obs < 2:
        return {f1: {f2: 0.0 for f2 in factors} for f1 in factors}

    means = {f: sum(factor_returns[f][:n_obs]) / n_obs for f in factors}
    result: dict[str, dict[str, float]] = {}
    for f1 in factors:
        row: dict[str, float] = {}
        for f2 in factors:
            cov = sum(
                (factor_returns[f1][t] - means[f1]) * (factor_returns[f2][t] - means[f2])
                for t in range(n_obs)
            ) / (n_obs - 1)
            row[f2] = cov
        result[f1] = row
    return result


def factor_model_covariance(
    symbols: Sequence[str],
    exposures: Mapping[str, Mapping[str, float]],
    factor_cov: Mapping[str, Mapping[str, float]],
    specific_risk: Mapping[str, float],
) -> dict[str, dict[str, float]]:
    """从因子模型构造资产协方差矩阵: Σ = B·F·B' + D。"""
    factors = list(factor_cov.keys())
    result: dict[str, dict[str, float]] = {}
    for s1 in symbols:
        row: dict[str, float] = {}
        b1 = exposures.get(s1, {})
        for s2 in symbols:
            b2 = exposures.get(s2, {})
            cov = 0.0
            for f1 in factors:
                for f2 in factors:
                    cov += b1.get(f1, 0.0) * factor_cov[f1].get(f2, 0.0) * b2.get(f2, 0.0)
            if s1 == s2:
                cov += specific_risk.get(s1, 0.0)
            row[s2] = cov
        result[s1] = row
    return result
