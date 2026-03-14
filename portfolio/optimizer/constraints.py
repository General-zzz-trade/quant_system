# portfolio/optimizer/constraints.py
"""Optimization constraints."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol


class OptConstraint(Protocol):
    """优化约束协议。"""
    name: str

    def is_feasible(self, weights: Mapping[str, float]) -> bool:
        """检查权重是否满足约束。"""
        ...

    def project(self, weights: dict[str, float]) -> dict[str, float]:
        """将权重投影到可行域。"""
        ...


@dataclass(frozen=True, slots=True)
class LongOnlyConstraint:
    """多头约束 — 所有权重 >= 0。"""
    name: str = "long_only"

    def is_feasible(self, weights: Mapping[str, float]) -> bool:
        return all(w >= 0 for w in weights.values())

    def project(self, weights: dict[str, float]) -> dict[str, float]:
        return {k: max(0.0, v) for k, v in weights.items()}


@dataclass(frozen=True, slots=True)
class MaxWeightConstraint:
    """单品种最大权重约束。"""
    name: str = "max_weight"
    max_weight: float = 0.3

    def is_feasible(self, weights: Mapping[str, float]) -> bool:
        return all(abs(w) <= self.max_weight for w in weights.values())

    def project(self, weights: dict[str, float]) -> dict[str, float]:
        clamped = {}
        for k, v in weights.items():
            if v > self.max_weight:
                clamped[k] = self.max_weight
            elif v < -self.max_weight:
                clamped[k] = -self.max_weight
            else:
                clamped[k] = v
        return clamped


@dataclass(frozen=True, slots=True)
class FullInvestmentConstraint:
    """满仓约束 — 权重之和 = 1。"""
    name: str = "full_investment"
    tolerance: float = 0.01

    def is_feasible(self, weights: Mapping[str, float]) -> bool:
        return abs(sum(weights.values()) - 1.0) <= self.tolerance

    def project(self, weights: dict[str, float]) -> dict[str, float]:
        total = sum(weights.values())
        if total == 0:
            return weights
        return {k: v / total for k, v in weights.items()}
