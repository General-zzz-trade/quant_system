# portfolio/risk_model/aggregation/decomposition.py
"""Risk decomposition — by asset and by factor."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from portfolio.risk_model.aggregation.marginal import (
    MarginalContribution,
    compute_marginal_risk,
)


@dataclass(frozen=True, slots=True)
class RiskDecomposition:
    """风险分解结果。"""
    total_variance: float
    by_asset: tuple[MarginalContribution, ...]
    systematic_pct: float
    idiosyncratic_pct: float


def decompose_risk(
    weights: Mapping[str, float],
    covariance: Mapping[str, Mapping[str, float]],
    specific_risk: Mapping[str, float] | None = None,
) -> RiskDecomposition:
    """分解组合风险为系统性/特异性。"""
    symbols = list(weights.keys())
    total_var = 0.0
    for s1 in symbols:
        row = covariance.get(s1, {})
        for s2 in symbols:
            total_var += weights[s1] * weights[s2] * row.get(s2, 0.0)

    idio_var = 0.0
    if specific_risk:
        for s in symbols:
            idio_var += weights[s] ** 2 * specific_risk.get(s, 0.0)

    if total_var > 0:
        idio_pct = idio_var / total_var
        sys_pct = 1.0 - idio_pct
    else:
        idio_pct = 0.0
        sys_pct = 0.0

    by_asset = compute_marginal_risk(weights, covariance)
    return RiskDecomposition(
        total_variance=total_var,
        by_asset=tuple(by_asset),
        systematic_pct=sys_pct,
        idiosyncratic_pct=idio_pct,
    )
