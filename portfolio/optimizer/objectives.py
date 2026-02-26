# portfolio/optimizer/objectives.py
"""Optimization objectives — what to optimize for."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence


class Objective(Protocol):
    """优化目标协议。"""
    name: str

    def evaluate(self, weights: Mapping[str, float], input_data: object) -> float:
        """计算目标函数值。"""
        ...


def _portfolio_variance(weights: Mapping[str, float], cov: Mapping[str, Mapping[str, float]]) -> float:
    """w'Cov*w"""
    variance = 0.0
    for s1, w1 in weights.items():
        row = cov.get(s1, {})
        for s2, w2 in weights.items():
            variance += w1 * w2 * row.get(s2, 0.0)
    return variance


@dataclass(frozen=True, slots=True)
class MaxSharpe:
    """最大化夏普比率: -(w'mu - rf) / sqrt(w'Cov*w)"""
    name: str = "max_sharpe"
    risk_free_rate: float = 0.0

    def evaluate(self, weights: Mapping[str, float], input_data: object) -> float:
        inp = input_data
        er = getattr(inp, "expected_returns", {})
        cov = getattr(inp, "covariance", {})

        port_return = sum(weights.get(s, 0) * er.get(s, 0) for s in weights)
        port_vol = math.sqrt(max(_portfolio_variance(weights, cov), 1e-20))

        # 最小化负夏普 = 最大化夏普
        return -(port_return - self.risk_free_rate) / port_vol


@dataclass(frozen=True, slots=True)
class MinVariance:
    """最小化组合方差。"""
    name: str = "min_variance"

    def evaluate(self, weights: Mapping[str, float], input_data: object) -> float:
        inp = input_data
        cov = getattr(inp, "covariance", {})
        return _portfolio_variance(weights, cov)


@dataclass(frozen=True, slots=True)
class RiskParity:
    """风险平价 — 等风险贡献: min sum((RC_i/TotalRC - 1/N)^2)"""
    name: str = "risk_parity"

    def evaluate(self, weights: Mapping[str, float], input_data: object) -> float:
        n = len(weights)
        if n == 0:
            return 0.0

        inp = input_data
        cov = getattr(inp, "covariance", {})

        # 边际风险贡献: marginal_i = (Cov @ w)_i
        symbols = list(weights.keys())
        marginal = []
        for s1 in symbols:
            row = cov.get(s1, {})
            mc = sum(weights.get(s2, 0) * row.get(s2, 0) for s2 in symbols)
            marginal.append(mc)

        # 风险贡献: RC_i = w_i * marginal_i
        risk_contrib = [weights[s] * m for s, m in zip(symbols, marginal)]
        total_risk = sum(risk_contrib)

        if abs(total_risk) < 1e-20:
            return 0.0

        # 目标: 每个品种贡献 1/N 的总风险
        target = 1.0 / n
        return sum((rc / total_risk - target) ** 2 for rc in risk_contrib)
