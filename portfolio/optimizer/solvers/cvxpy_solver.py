# portfolio/optimizer/solvers/cvxpy_solver.py
"""CVXPY-based portfolio optimizer. Optional dependency (cvxpy)."""
from __future__ import annotations

import logging
import math
from typing import Mapping, Optional, Sequence

from portfolio.optimizer.base import OptimizationResult
from portfolio.optimizer.constraints import OptConstraint
from portfolio.optimizer.input import OptimizationInput
from portfolio.optimizer.objectives import (
    MaxSharpe,
    MinVariance,
    Objective,
    RiskParity,
    _portfolio_variance,
)

logger = logging.getLogger(__name__)


class CvxpySolver:
    """CVXPY-based portfolio optimizer.

    Uses CVXPY for convex optimization with support for turnover and CVaR
    constraints. Falls back to QPSolver when cvxpy is not installed.
    """

    name: str = "cvxpy"

    def __init__(
        self,
        *,
        solver: str = "SCS",
        verbose: bool = False,
        max_iter: int = 5000,
    ) -> None:
        self._solver = solver
        self._verbose = verbose
        self._max_iter = max_iter

    def optimize(
        self,
        inp: OptimizationInput,
        objectives: Sequence[Objective],
        constraints: Sequence[OptConstraint],
        *,
        turnover_limit: Optional[float] = None,
        cvar_limit: Optional[float] = None,
    ) -> OptimizationResult:
        """Solve using CVXPY. Falls back to QPSolver if cvxpy not installed."""
        try:
            import cvxpy as cp
        except ImportError:
            logger.warning("cvxpy not installed, falling back to QPSolver")
            return self._fallback(inp, objectives, constraints)

        n = inp.n_assets
        if n == 0:
            return OptimizationResult(
                weights={}, objective_value=0.0, converged=True, iterations=0,
            )

        symbols = list(inp.symbols)

        # Build covariance matrix
        cov_matrix = []
        for si in symbols:
            row = []
            cov_row = inp.covariance.get(si, {})
            for sj in symbols:
                row.append(cov_row.get(sj, 0.0))
            cov_matrix.append(row)

        # Build expected returns vector
        mu = [inp.expected_returns.get(s, 0.0) for s in symbols]

        # Decision variable
        w = cp.Variable(n)

        # Determine objective from first objective type
        obj = objectives[0] if objectives else None
        cvxpy_obj = self._build_objective(cp, w, mu, cov_matrix, obj, inp)

        # Build constraints
        cvxpy_constraints = self._build_constraints(
            cp, w, n, symbols, constraints, inp,
            turnover_limit=turnover_limit,
        )

        # CVaR constraint (simplified: use variance as proxy when scenarios unavailable)
        if cvar_limit is not None:
            portfolio_var = cp.quad_form(w, cp.psd_wrap(cov_matrix))
            # Approximate: limit portfolio volatility as CVaR proxy
            cvxpy_constraints.append(portfolio_var <= cvar_limit ** 2)

        problem = cp.Problem(cvxpy_obj, cvxpy_constraints)

        try:
            problem.solve(
                solver=self._solver,
                verbose=self._verbose,
                max_iters=self._max_iter,
            )
        except cp.SolverError as exc:
            logger.warning("CVXPY solver failed: %s, falling back to QPSolver", exc)
            return self._fallback(inp, objectives, constraints)

        if problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            raw_weights = w.value
            weights = {symbols[i]: float(raw_weights[i]) for i in range(n)}
            obj_val = float(problem.value) if problem.value is not None else 0.0
            return OptimizationResult(
                weights=weights,
                objective_value=obj_val,
                converged=True,
                iterations=problem.solver_stats.num_iters if problem.solver_stats else 0,
                message=f"cvxpy:{problem.status}",
                diagnostics={"solver": self._solver, "status": problem.status},
            )

        logger.warning("CVXPY status=%s, falling back to QPSolver", problem.status)
        return self._fallback(inp, objectives, constraints)

    def _build_objective(self, cp, w, mu, cov_matrix, obj, inp):
        """Build CVXPY objective expression from Objective protocol."""
        import cvxpy as cp_mod

        if isinstance(obj, MinVariance):
            return cp_mod.Minimize(cp_mod.quad_form(w, cp_mod.psd_wrap(cov_matrix)))

        if isinstance(obj, MaxSharpe):
            # Maximize return / vol approximation: minimize variance - lambda*return
            ret = mu @ w
            var = cp_mod.quad_form(w, cp_mod.psd_wrap(cov_matrix))
            return cp_mod.Minimize(var - ret)

        if isinstance(obj, RiskParity):
            # Risk parity is non-convex; approximate with min variance
            logger.info("RiskParity approximated as MinVariance in CVXPY solver")
            return cp_mod.Minimize(cp_mod.quad_form(w, cp_mod.psd_wrap(cov_matrix)))

        # Default: minimize variance
        return cp_mod.Minimize(cp_mod.quad_form(w, cp_mod.psd_wrap(cov_matrix)))

    def _build_constraints(
        self, cp, w, n, symbols, constraints, inp,
        *, turnover_limit=None,
    ):
        """Translate OptConstraint list to CVXPY constraints."""
        import cvxpy as cp_mod

        cvxpy_constraints = []

        for c in constraints:
            cname = getattr(c, "name", "")

            if cname == "long_only":
                cvxpy_constraints.append(w >= 0)

            elif cname == "max_weight":
                max_w = getattr(c, "max_weight", 0.3)
                cvxpy_constraints.append(w <= max_w)
                cvxpy_constraints.append(w >= -max_w)

            elif cname == "full_investment":
                tol = getattr(c, "tolerance", 0.01)
                cvxpy_constraints.append(cp_mod.sum(w) >= 1.0 - tol)
                cvxpy_constraints.append(cp_mod.sum(w) <= 1.0 + tol)

        # Turnover constraint
        if turnover_limit is not None:
            current = [inp.current_weights.get(s, 0.0) for s in symbols]
            cvxpy_constraints.append(
                cp_mod.norm(w - current, 1) <= turnover_limit
            )

        return cvxpy_constraints

    def _fallback(
        self,
        inp: OptimizationInput,
        objectives: Sequence[Objective],
        constraints: Sequence[OptConstraint],
    ) -> OptimizationResult:
        """Fall back to QPSolver."""
        from portfolio.optimizer.solvers.qp import QPSolver

        return QPSolver().optimize(inp, objectives, constraints)
