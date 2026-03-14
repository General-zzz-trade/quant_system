"""Tests for portfolio optimizer."""
from __future__ import annotations

from portfolio.optimizer.input import OptimizationInput
from portfolio.optimizer.objectives import MaxSharpe
from portfolio.optimizer.constraints import LongOnlyConstraint, FullInvestmentConstraint
from portfolio.optimizer.solvers.heuristics import EqualWeightSolver


def test_equal_weight_solver():
    inp = OptimizationInput(
        symbols=("BTC", "ETH", "SOL"),
        expected_returns={"BTC": 0.1, "ETH": 0.15, "SOL": 0.2},
        covariance={
            "BTC": {"BTC": 0.04, "ETH": 0.01, "SOL": 0.005},
            "ETH": {"BTC": 0.01, "ETH": 0.06, "SOL": 0.008},
            "SOL": {"BTC": 0.005, "ETH": 0.008, "SOL": 0.09},
        },
        current_weights={"BTC": 0.0, "ETH": 0.0, "SOL": 0.0},
    )
    solver = EqualWeightSolver()
    result = solver.optimize(inp, [MaxSharpe()], [LongOnlyConstraint()])
    assert len(result.weights) == 3
    assert result.converged
    for w in result.weights.values():
        assert abs(w - 1/3) < 1e-9


def test_long_only_constraint():
    c = LongOnlyConstraint()
    weights = {"A": 0.5, "B": -0.3, "C": 0.8}
    assert not c.is_feasible(weights)
    projected = c.project(weights)
    assert all(v >= 0 for v in projected.values())


def test_full_investment_constraint():
    c = FullInvestmentConstraint()
    weights = {"A": 0.3, "B": 0.3, "C": 0.4}
    assert c.is_feasible(weights)
