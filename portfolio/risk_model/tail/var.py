# portfolio/risk_model/tail/var.py
"""Value at Risk (VaR) estimation."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True, slots=True)
class VaREstimate:
    """VaR 估计结果。"""
    confidence: float
    var_parametric: float
    var_historical: float
    method: str


def parametric_var(
    returns: Sequence[float],
    confidence: float = 0.95,
    annualization: float = 1.0,
) -> float:
    """参数法 VaR（正态假设）。"""
    n = len(returns)
    if n < 2:
        return 0.0
    mean = sum(returns) / n
    std = math.sqrt(sum((r - mean) ** 2 for r in returns) / (n - 1))
    # 正态分位数近似
    z_map = {0.90: 1.282, 0.95: 1.645, 0.99: 2.326}
    z = z_map.get(confidence, 1.645)
    return (mean - z * std) * math.sqrt(annualization)


def historical_var(
    returns: Sequence[float],
    confidence: float = 0.95,
) -> float:
    """历史模拟法 VaR。"""
    if not returns:
        return 0.0
    sorted_r = sorted(returns)
    idx = int((1 - confidence) * len(sorted_r))
    idx = max(0, min(idx, len(sorted_r) - 1))
    return sorted_r[idx]


def compute_var(
    returns: Sequence[float],
    confidence: float = 0.95,
) -> VaREstimate:
    """计算 VaR（参数法 + 历史法）。"""
    return VaREstimate(
        confidence=confidence,
        var_parametric=parametric_var(returns, confidence),
        var_historical=historical_var(returns, confidence),
        method="parametric+historical",
    )
