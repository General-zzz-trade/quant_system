# portfolio/risk_model/factor/covariance.py
"""Factor covariance estimation."""
from __future__ import annotations

from typing import Mapping, Sequence

from _quant_hotpath import (
    cpp_sample_covariance as _cpp_sample_covariance,
    cpp_factor_model_covariance as _cpp_factor_model_covariance,
)


def estimate_factor_covariance(
    factor_returns: Mapping[str, Sequence[float]],
) -> dict[str, dict[str, float]]:
    """估计因子收益率的协方差矩阵。"""
    factors = list(factor_returns.keys())
    n_obs = min(len(factor_returns[f]) for f in factors) if factors else 0

    if n_obs < 2:
        return {f1: {f2: 0.0 for f2 in factors} for f1 in factors}

    matrix = [list(factor_returns[f][:n_obs]) for f in factors]
    cov_mat = _cpp_sample_covariance(matrix)
    return {
        f1: {f2: cov_mat[i][j] for j, f2 in enumerate(factors)}
        for i, f1 in enumerate(factors)
    }


def factor_model_covariance(
    symbols: Sequence[str],
    exposures: Mapping[str, Mapping[str, float]],
    factor_cov: Mapping[str, Mapping[str, float]],
    specific_risk: Mapping[str, float],
) -> dict[str, dict[str, float]]:
    """从因子模型构造资产协方差矩阵: Σ = B·F·B' + D。"""
    factors = list(factor_cov.keys())

    if not factors:
        return {s1: {s2: 0.0 for s2 in symbols} for s1 in symbols}

    exp_mat = [[exposures.get(s, {}).get(f, 0.0) for f in factors] for s in symbols]
    fcov_mat = [[factor_cov[f1].get(f2, 0.0) for f2 in factors] for f1 in factors]
    sr_vec = [specific_risk.get(s, 0.0) for s in symbols]
    cov_mat = _cpp_factor_model_covariance(exp_mat, fcov_mat, sr_vec)
    return {
        s1: {s2: cov_mat[i][j] for j, s2 in enumerate(symbols)}
        for i, s1 in enumerate(symbols)
    }
