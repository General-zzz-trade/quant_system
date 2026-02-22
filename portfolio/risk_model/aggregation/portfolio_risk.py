# portfolio/risk_model/aggregation/portfolio_risk.py
"""Portfolio-level risk aggregation."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True, slots=True)
class PortfolioRisk:
    """组合风险指标。"""
    variance: float
    volatility: float
    var_95: float           # 95% 参数 VaR
    var_99: float           # 99% 参数 VaR
    expected_shortfall_95: float


def compute_portfolio_risk(
    weights: Mapping[str, float],
    covariance: Mapping[str, Mapping[str, float]],
    annualization: float = 365.0,
) -> PortfolioRisk:
    """计算组合整体风险指标。"""
    symbols = list(weights.keys())
    variance = 0.0
    for s1 in symbols:
        w1 = weights[s1]
        row = covariance.get(s1, {})
        for s2 in symbols:
            variance += w1 * weights[s2] * row.get(s2, 0.0)

    ann_var = variance * annualization
    vol = math.sqrt(max(ann_var, 0.0))
    # 参数法 VaR (正态假设)
    var_95 = vol * 1.645
    var_99 = vol * 2.326
    es_95 = vol * 2.063  # E[X | X > VaR_95] for normal
    return PortfolioRisk(
        variance=ann_var,
        volatility=vol,
        var_95=var_95,
        var_99=var_99,
        expected_shortfall_95=es_95,
    )
