# portfolio/optimizer/base.py
"""Optimizer base protocol and result type."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Protocol, Sequence

from portfolio.optimizer.constraints import OptConstraint
from portfolio.optimizer.input import OptimizationInput
from portfolio.optimizer.objectives import Objective


@dataclass(frozen=True, slots=True)
class OptimizationResult:
    """优化结果。"""
    weights: dict[str, float]
    objective_value: float
    converged: bool
    iterations: int
    message: str = ""
    diagnostics: dict[str, object] = field(default_factory=dict)

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(self.weights.keys())

    def weight(self, symbol: str) -> float:
        return self.weights.get(symbol, 0.0)


class Optimizer(Protocol):
    """优化器协议。"""
    name: str

    def optimize(
        self,
        inp: OptimizationInput,
        objectives: Sequence[Objective],
        constraints: Sequence[OptConstraint],
    ) -> OptimizationResult:
        """执行优化, 返回最优权重。"""
        ...
