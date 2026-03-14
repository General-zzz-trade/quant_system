# portfolio/risk_model/diagnostics/explain.py
"""Human-readable risk model explanations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from portfolio.risk_model.aggregation.marginal import (
    compute_marginal_risk,
)


@dataclass(frozen=True, slots=True)
class RiskExplanation:
    """可读的风险解释。"""
    summary: str
    top_contributors: tuple[str, ...]
    risk_level: str  # LOW / MEDIUM / HIGH / EXTREME
    details: dict[str, str]


def explain_portfolio_risk(
    weights: Mapping[str, float],
    covariance: Mapping[str, Mapping[str, float]],
    volatilities: Mapping[str, float],
) -> RiskExplanation:
    """生成组合风险的可读解释。"""
    import math

    # 组合波动率
    port_var = 0.0
    symbols = list(weights.keys())
    for s1 in symbols:
        row = covariance.get(s1, {})
        for s2 in symbols:
            port_var += weights[s1] * weights[s2] * row.get(s2, 0.0)
    port_vol = math.sqrt(max(port_var * 365, 0.0))

    # 风险等级
    if port_vol < 0.15:
        level = "LOW"
    elif port_vol < 0.30:
        level = "MEDIUM"
    elif port_vol < 0.60:
        level = "HIGH"
    else:
        level = "EXTREME"

    # 主要贡献者
    contribs = compute_marginal_risk(weights, covariance)
    sorted_c = sorted(contribs, key=lambda c: abs(c.risk_contribution), reverse=True)
    top = tuple(c.symbol for c in sorted_c[:3])

    details = {
        "portfolio_vol": f"{port_vol:.2%}",
        "n_assets": str(len(symbols)),
        "max_weight": f"{max(abs(v) for v in weights.values()):.2%}" if weights else "0",
    }

    return RiskExplanation(
        summary=f"Portfolio annualized vol: {port_vol:.2%}, risk level: {level}",
        top_contributors=top,
        risk_level=level,
        details=details,
    )
