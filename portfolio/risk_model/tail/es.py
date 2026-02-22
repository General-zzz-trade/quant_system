# portfolio/risk_model/tail/es.py
"""Expected Shortfall (CVaR) estimation."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True, slots=True)
class ESEstimate:
    """Expected Shortfall 估计。"""
    confidence: float
    es_historical: float
    es_parametric: float


def historical_es(
    returns: Sequence[float],
    confidence: float = 0.95,
) -> float:
    """历史法 Expected Shortfall: 超过 VaR 的平均损失。"""
    if not returns:
        return 0.0
    sorted_r = sorted(returns)
    cutoff = int((1 - confidence) * len(sorted_r))
    cutoff = max(1, cutoff)
    tail = sorted_r[:cutoff]
    return sum(tail) / len(tail)


def parametric_es(
    returns: Sequence[float],
    confidence: float = 0.95,
) -> float:
    """参数法 ES（正态假设）。"""
    n = len(returns)
    if n < 2:
        return 0.0
    mean = sum(returns) / n
    std = math.sqrt(sum((r - mean) ** 2 for r in returns) / (n - 1))
    # ES = μ - σ · φ(z) / (1-α)  where φ is standard normal pdf
    z_map = {0.90: 1.282, 0.95: 1.645, 0.99: 2.326}
    z = z_map.get(confidence, 1.645)
    phi_z = math.exp(-0.5 * z ** 2) / math.sqrt(2 * math.pi)
    alpha = 1 - confidence
    return mean - std * phi_z / alpha


def compute_es(
    returns: Sequence[float],
    confidence: float = 0.95,
) -> ESEstimate:
    """计算 Expected Shortfall。"""
    return ESEstimate(
        confidence=confidence,
        es_historical=historical_es(returns, confidence),
        es_parametric=parametric_es(returns, confidence),
    )
