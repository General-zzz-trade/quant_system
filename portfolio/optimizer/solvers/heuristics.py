# portfolio/optimizer/solvers/heuristics.py
"""Heuristic (closed-form) portfolio solvers."""
from __future__ import annotations

from typing import Sequence
import math

from portfolio.optimizer.base import OptimizationResult
from portfolio.optimizer.constraints import OptConstraint
from portfolio.optimizer.input import OptimizationInput
from portfolio.optimizer.objectives import Objective


class EqualWeightSolver:
    """等权分配求解器。"""
    name: str = "equal_weight"

    def optimize(
        self,
        inp: OptimizationInput,
        objectives: Sequence[Objective],
        constraints: Sequence[OptConstraint],
    ) -> OptimizationResult:
        n = inp.n_assets
        if n == 0:
            return OptimizationResult(
                weights={}, objective_value=0.0, converged=True, iterations=0,
            )
        w = 1.0 / n
        weights = {s: w for s in inp.symbols}
        obj_val = objectives[0].evaluate(weights, inp) if objectives else 0.0
        return OptimizationResult(
            weights=weights,
            objective_value=obj_val,
            converged=True,
            iterations=1,
            message="equal weight",
        )


class InverseVolatilitySolver:
    """反波动率加权求解器。"""
    name: str = "inverse_volatility"

    def optimize(
        self,
        inp: OptimizationInput,
        objectives: Sequence[Objective],
        constraints: Sequence[OptConstraint],
    ) -> OptimizationResult:
        vols: dict[str, float] = {}
        for s in inp.symbols:
            var = inp.covariance.get(s, {}).get(s, 1.0)
            vols[s] = math.sqrt(max(var, 1e-12))

        inv_vols = {s: 1.0 / v for s, v in vols.items()}
        total = sum(inv_vols.values())
        if total == 0:
            weights = {s: 1.0 / len(inp.symbols) for s in inp.symbols}
        else:
            weights = {s: iv / total for s, iv in inv_vols.items()}

        for c in constraints:
            weights = c.project(weights)

        obj_val = objectives[0].evaluate(weights, inp) if objectives else 0.0
        return OptimizationResult(
            weights=weights,
            objective_value=obj_val,
            converged=True,
            iterations=1,
            message="inverse volatility",
        )


class MaxReturnSolver:
    """最大收益率（集中于最高期望收益品种）。"""
    name: str = "max_return"

    def optimize(
        self,
        inp: OptimizationInput,
        objectives: Sequence[Objective],
        constraints: Sequence[OptConstraint],
    ) -> OptimizationResult:
        er = inp.expected_returns
        if not er:
            weights = {s: 1.0 / inp.n_assets for s in inp.symbols}
        else:
            best = max(inp.symbols, key=lambda s: er.get(s, 0.0))
            weights = {s: (1.0 if s == best else 0.0) for s in inp.symbols}
        for c in constraints:
            weights = c.project(weights)
        obj_val = objectives[0].evaluate(weights, inp) if objectives else 0.0
        return OptimizationResult(
            weights=weights,
            objective_value=obj_val,
            converged=True,
            iterations=1,
            message="max return heuristic",
        )
