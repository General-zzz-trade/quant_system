# portfolio/optimizer/objectives.py
"""Optimization objectives — what to optimize for."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence


class Objective(Protocol):
    """优化目标协议。"""
    name: str

    def evaluate(self, weights: Mapping[str, float], input_data: object) -> float:
        """计算目标函数值。"""
        ...


@dataclass(frozen=True, slots=True)
class MaxSharpe:
    """最大化夏普比率。"""
    name: str = "max_sharpe"

    def evaluate(self, weights: Mapping[str, float], input_data: object) -> float:
        inp = input_data
        er = getattr(inp, "expected_returns", {})
        port_return = sum(weights.get(s, 0) * er.get(s, 0) for s in weights)
        return -port_return  # 最小化负收益 = 最大化收益


@dataclass(frozen=True, slots=True)
class MinVariance:
    """最小化组合方差。"""
    name: str = "min_variance"

    def evaluate(self, weights: Mapping[str, float], input_data: object) -> float:
        inp = input_data
        cov = getattr(inp, "covariance", {})
        variance = 0.0
        for s1, w1 in weights.items():
            row = cov.get(s1, {})
            for s2, w2 in weights.items():
                variance += w1 * w2 * row.get(s2, 0)
        return variance


@dataclass(frozen=True, slots=True)
class RiskParity:
    """风险平价 — 等风险贡献。"""
    name: str = "risk_parity"

    def evaluate(self, weights: Mapping[str, float], input_data: object) -> float:
        n = len(weights)
        if n == 0:
            return 0.0
        target = 1.0 / n
        deviation = sum((w - target) ** 2 for w in weights.values())
        return deviation
