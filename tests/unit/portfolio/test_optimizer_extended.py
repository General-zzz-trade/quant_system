"""Extended tests for portfolio optimizer: constraints, solvers, edge cases."""
from __future__ import annotations

import pytest

from portfolio.optimizer.objectives import MaxSharpe, MinVariance, RiskParity
from portfolio.optimizer.constraints import (
    LongOnlyConstraint,
    MaxWeightConstraint,
    FullInvestmentConstraint,
)
from portfolio.optimizer.input import OptimizationInput
from portfolio.optimizer.base import OptimizationResult
from portfolio.optimizer.solvers.linear import LinearSolver
from portfolio.optimizer.solvers.heuristics import (
    EqualWeightSolver,
    InverseVolatilitySolver,
)


def _make_input(symbols=("BTC", "ETH", "SOL")):
    returns = {"BTC": 0.05, "ETH": 0.08, "SOL": 0.12}
    cov = {
        "BTC": {"BTC": 0.04, "ETH": 0.02, "SOL": 0.01},
        "ETH": {"BTC": 0.02, "ETH": 0.09, "SOL": 0.03},
        "SOL": {"BTC": 0.01, "ETH": 0.03, "SOL": 0.16},
    }
    ret = {s: returns.get(s, 0.05) for s in symbols}
    c = {s: {t: cov.get(s, {}).get(t, 0.0) for t in symbols} for s in symbols}
    return OptimizationInput(
        symbols=tuple(symbols),
        expected_returns=ret,
        covariance=c,
        current_weights={s: 1.0 / len(symbols) for s in symbols},
    )


# ── MaxWeightConstraint ──────────────────────────────────────

class TestMaxWeightConstraint:
    def test_feasible_within_limit(self):
        c = MaxWeightConstraint(max_weight=0.5)
        assert c.is_feasible({"BTC": 0.3, "ETH": 0.4}) is True

    def test_infeasible_over_limit(self):
        c = MaxWeightConstraint(max_weight=0.5)
        assert c.is_feasible({"BTC": 0.6, "ETH": 0.4}) is False

    def test_project_clips(self):
        c = MaxWeightConstraint(max_weight=0.4)
        result = c.project({"BTC": 0.7, "ETH": 0.3})
        assert result["BTC"] <= 0.4 + 1e-9


class TestFullInvestmentConstraint:
    def test_feasible_sums_to_one(self):
        c = FullInvestmentConstraint()
        assert c.is_feasible({"BTC": 0.5, "ETH": 0.5}) is True

    def test_infeasible_not_one(self):
        c = FullInvestmentConstraint(tolerance=0.01)
        assert c.is_feasible({"BTC": 0.3, "ETH": 0.3}) is False

    def test_project_normalizes(self):
        c = FullInvestmentConstraint()
        result = c.project({"BTC": 0.4, "ETH": 0.2, "SOL": 0.1})
        total = sum(result.values())
        assert abs(total - 1.0) < 0.02


# ── MinVariance objective ────────────────────────────────────

class TestMinVariance:
    def test_variance_non_negative(self):
        obj = MinVariance()
        inp = _make_input()
        val = obj.evaluate({"BTC": 0.5, "ETH": 0.3, "SOL": 0.2}, inp)
        assert val >= 0

    def test_concentrated_portfolio_higher_variance(self):
        obj = MinVariance()
        inp = _make_input()
        diversified = {"BTC": 0.34, "ETH": 0.33, "SOL": 0.33}
        concentrated = {"BTC": 0.0, "ETH": 0.0, "SOL": 1.0}  # highest vol asset
        v_div = obj.evaluate(diversified, inp)
        v_con = obj.evaluate(concentrated, inp)
        assert v_con > v_div


# ── RiskParity objective ─────────────────────────────────────

class TestRiskParity:
    def test_returns_float(self):
        obj = RiskParity()
        inp = _make_input()
        val = obj.evaluate({"BTC": 0.33, "ETH": 0.33, "SOL": 0.34}, inp)
        assert isinstance(val, float)


# ── InverseVolatilitySolver ──────────────────────────────────

class TestInverseVolatilitySolver:
    def test_lower_vol_higher_weight(self):
        solver = InverseVolatilitySolver()
        inp = _make_input()
        result = solver.optimize(inp, [], [])
        # BTC: vol=0.04 (lowest), SOL: vol=0.16 (highest)
        assert result.weights["BTC"] > result.weights["SOL"]

    def test_weights_sum_to_one(self):
        solver = InverseVolatilitySolver()
        inp = _make_input()
        result = solver.optimize(inp, [], [])
        assert abs(sum(result.weights.values()) - 1.0) < 0.01


# ── LinearSolver ─────────────────────────────────────────────

class TestLinearSolver:
    def test_converges_min_variance(self):
        solver = LinearSolver(max_iter=200, learning_rate=0.01)
        inp = _make_input()
        result = solver.optimize(
            inp, [MinVariance()],
            [LongOnlyConstraint(), FullInvestmentConstraint()],
        )
        assert result.iterations > 0
        for w in result.weights.values():
            assert w >= -0.01

    def test_empty_input(self):
        solver = LinearSolver()
        inp = OptimizationInput(
            symbols=(), expected_returns={}, covariance={}, current_weights={},
        )
        result = solver.optimize(inp, [], [])
        assert result.weights == {}
        assert result.converged is True

    def test_two_assets(self):
        solver = LinearSolver(max_iter=100)
        inp = _make_input(symbols=("BTC", "ETH"))
        result = solver.optimize(
            inp, [MinVariance()], [LongOnlyConstraint()],
        )
        assert len(result.weights) == 2
