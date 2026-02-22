# portfolio/risk_model/aggregation/marginal.py
"""Marginal risk contribution."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True, slots=True)
class MarginalContribution:
    """资产边际风险贡献。"""
    symbol: str
    marginal_variance: float
    risk_contribution: float
    pct_contribution: float


def compute_marginal_risk(
    weights: Mapping[str, float],
    covariance: Mapping[str, Mapping[str, float]],
) -> list[MarginalContribution]:
    """计算每个资产对组合风险的边际贡献。"""
    symbols = list(weights.keys())
    # σ²_p = w' Σ w
    port_var = 0.0
    for s1 in symbols:
        row = covariance.get(s1, {})
        for s2 in symbols:
            port_var += weights[s1] * weights[s2] * row.get(s2, 0.0)

    port_vol = math.sqrt(max(port_var, 1e-12))
    results = []
    for s in symbols:
        # 边际方差 = ∂σ²/∂w_i = 2·Σ_j cov(i,j)·w_j
        row = covariance.get(s, {})
        mvar = sum(row.get(s2, 0.0) * weights[s2] for s2 in symbols)
        # 风险贡献 = w_i · mvar / σ_p
        rc = weights[s] * mvar / port_vol if port_vol > 0 else 0.0
        pct = rc / port_vol if port_vol > 0 else 0.0
        results.append(MarginalContribution(
            symbol=s,
            marginal_variance=mvar,
            risk_contribution=rc,
            pct_contribution=pct,
        ))
    return results
