# portfolio/optimizer/solvers/linear.py
"""Linear programming solver (simplified projected gradient)."""
from __future__ import annotations

from typing import Sequence

from portfolio.optimizer.base import OptimizationResult
from portfolio.optimizer.constraints import OptConstraint
from portfolio.optimizer.input import OptimizationInput
from portfolio.optimizer.objectives import Objective


class LinearSolver:
    """线性目标的投影梯度求解器。"""
    name: str = "linear"

    def __init__(
        self,
        max_iter: int = 200,
        learning_rate: float = 0.01,
        tol: float = 1e-8,
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

        # 初始等权
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

            # 数值梯度
            grad = {}
            eps = 1e-7
            for s in inp.symbols:
                w_plus = dict(weights)
                w_plus[s] += eps
                grad[s] = (obj_fn.evaluate(w_plus, inp) - val) / eps

            # 梯度步
            for s in inp.symbols:
                weights[s] -= self.learning_rate * grad[s]

            # 投影到约束域
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
