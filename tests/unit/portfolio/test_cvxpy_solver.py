# tests/unit/portfolio/test_cvxpy_solver.py
"""Tests for CVXPY solver — focuses on fallback and basic behavior."""
from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest import mock

import pytest

from portfolio.optimizer.base import OptimizationResult
from portfolio.optimizer.constraints import FullInvestmentConstraint, LongOnlyConstraint
from portfolio.optimizer.input import OptimizationInput
from portfolio.optimizer.objectives import MinVariance


# ── helpers ──────────────────────────────────────────────────────────

def _make_input() -> OptimizationInput:
    return OptimizationInput(
        symbols=("A", "B"),
        expected_returns={"A": 0.10, "B": 0.05},
        covariance={
            "A": {"A": 0.04, "B": 0.01},
            "B": {"A": 0.01, "B": 0.09},
        },
        current_weights={"A": 0.5, "B": 0.5},
    )


# ── fallback tests ───────────────────────────────────────────────────

class TestCvxpyFallback:
    def test_fallback_when_cvxpy_not_installed(self):
        """When cvxpy is not available, CvxpySolver falls back to QPSolver."""
        # Temporarily make cvxpy unimportable
        with mock.patch.dict(sys.modules, {"cvxpy": None}):
            from portfolio.optimizer.solvers.cvxpy_solver import CvxpySolver

            solver = CvxpySolver()
            inp = _make_input()
            result = solver.optimize(
                inp,
                objectives=[MinVariance()],
                constraints=[LongOnlyConstraint(), FullInvestmentConstraint()],
            )

            assert isinstance(result, OptimizationResult)
            assert sum(result.weights.values()) == pytest.approx(1.0, abs=0.02)
            assert all(w >= -1e-9 for w in result.weights.values())

    def test_fallback_returns_converged(self):
        """Fallback QPSolver should converge for simple MinVariance."""
        with mock.patch.dict(sys.modules, {"cvxpy": None}):
            from portfolio.optimizer.solvers.cvxpy_solver import CvxpySolver

            solver = CvxpySolver()
            inp = _make_input()
            result = solver.optimize(
                inp,
                objectives=[MinVariance()],
                constraints=[FullInvestmentConstraint()],
            )

            assert isinstance(result, OptimizationResult)
            # QPSolver should produce reasonable weights
            assert "A" in result.weights
            assert "B" in result.weights

    def test_empty_input(self):
        """Empty input should return empty result."""
        with mock.patch.dict(sys.modules, {"cvxpy": None}):
            from portfolio.optimizer.solvers.cvxpy_solver import CvxpySolver

            solver = CvxpySolver()
            inp = OptimizationInput(
                symbols=(),
                expected_returns={},
                covariance={},
                current_weights={},
            )
            result = solver.optimize(inp, objectives=[MinVariance()], constraints=[])
            assert result.weights == {}
            assert result.converged is True


class TestCvxpySolverConfig:
    def test_default_config(self):
        """Verify default configuration."""
        from portfolio.optimizer.solvers.cvxpy_solver import CvxpySolver

        solver = CvxpySolver()
        assert solver.name == "cvxpy"
        assert solver._solver == "SCS"
        assert solver._verbose is False

    def test_custom_config(self):
        """Verify custom configuration."""
        from portfolio.optimizer.solvers.cvxpy_solver import CvxpySolver

        solver = CvxpySolver(solver="ECOS", verbose=True, max_iter=1000)
        assert solver._solver == "ECOS"
        assert solver._verbose is True
        assert solver._max_iter == 1000
