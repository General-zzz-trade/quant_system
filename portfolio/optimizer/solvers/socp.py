# portfolio/optimizer/solvers/socp.py
"""Second-order cone programming solver (projected gradient with cone projection)."""
from __future__ import annotations

import math
from typing import Sequence

from portfolio.optimizer.base import OptimizationResult
from portfolio.optimizer.constraints import OptConstraint
from portfolio.optimizer.input import OptimizationInput
from portfolio.optimizer.objectives import Objective


class SOCPSolver:
    """SOCP 近似求解器（投影梯度 + 二阶锥约束投影）。"""
    name: str = "socp"

    def __init__(
        self,
        max_iter: int = 500,
        learning_rate: float = 0.005,
        tol: float = 1e-9,
        max_risk: float = 0.2,
    ) -> None:
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.tol = tol
        self.max_risk = max_risk

    def _portfolio_vol(
        self, weights: dict[str, float], inp: OptimizationInput
    ) -> float:
        """组合波动率 = sqrt(w' Σ w)。"""
        variance = 0.0
        for s1, w1 in weights.items():
            row = inp.covariance.get(s1, {})
            for s2, w2 in weights.items():
                variance += w1 * w2 * row.get(s2, 0.0)
        return math.sqrt(max(variance, 0.0))

    def _project_risk(
        self, weights: dict[str, float], inp: OptimizationInput
    ) -> dict[str, float]:
        """将权重投影到风险约束球内。"""
        vol = self._portfolio_vol(weights, inp)
        if vol <= self.max_risk or vol == 0:
            return weights
        scale = self.max_risk / vol
        return {k: v * scale for k, v in weights.items()}

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

            # 数值梯度
            eps = 1e-7
            grad = {}
            for s in inp.symbols:
                w_plus = dict(weights)
                w_plus[s] += eps
                grad[s] = (obj_fn.evaluate(w_plus, inp) - val) / eps

            for s in inp.symbols:
                weights[s] -= self.learning_rate * grad[s]

            # 先约束投影, 再风险投影
            for c in constraints:
                weights = c.project(weights)
            weights = self._project_risk(weights, inp)

        final_val = obj_fn.evaluate(weights, inp) if obj_fn else 0.0
        return OptimizationResult(
            weights=weights,
            objective_value=final_val,
            converged=False,
            iterations=self.max_iter,
            message="max iterations reached",
        )
