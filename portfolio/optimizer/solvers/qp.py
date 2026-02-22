# portfolio/optimizer/solvers/qp.py
"""Quadratic programming solver (projected gradient descent)."""
from __future__ import annotations

from typing import Sequence

from portfolio.optimizer.base import OptimizationResult
from portfolio.optimizer.constraints import OptConstraint
from portfolio.optimizer.input import OptimizationInput
from portfolio.optimizer.objectives import Objective


class QPSolver:
    """二次规划投影梯度下降求解器。"""
    name: str = "qp"

    def __init__(
        self,
        max_iter: int = 500,
        learning_rate: float = 0.005,
        tol: float = 1e-9,
    ) -> None:
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.tol = tol

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

        weights = {s: 1.0 / n for s in inp.symbols}
        obj_fn = objectives[0] if objectives else None
        prev_val = float("inf")

        for it in range(self.max_iter):
            if obj_fn is None:
                break
            val = obj_fn.evaluate(weights, inp)
            if abs(val - prev_val) < self.tol:
                return OptimizationResult(
                    weights=weights,
                    objective_value=val,
                    converged=True,
                    iterations=it + 1,
                    message="converged",
                )
            prev_val = val

            # 二次目标的解析梯度: ∂/∂w_i = 2 * Σ_j cov(i,j) * w_j
            grad = {}
            for s1 in inp.symbols:
                g = 0.0
                row = inp.covariance.get(s1, {})
                for s2 in inp.symbols:
                    g += 2.0 * row.get(s2, 0.0) * weights[s2]
                grad[s1] = g

            # 梯度下降
            for s in inp.symbols:
                weights[s] -= self.learning_rate * grad[s]

            # 投影
            for c in constraints:
                weights = c.project(weights)

        final_val = obj_fn.evaluate(weights, inp) if obj_fn else 0.0
        return OptimizationResult(
            weights=weights,
            objective_value=final_val,
            converged=False,
            iterations=self.max_iter,
            message="max iterations reached",
        )
