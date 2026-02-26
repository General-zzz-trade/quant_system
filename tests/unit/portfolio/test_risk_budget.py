# tests/unit/portfolio/test_risk_budget.py
"""Tests for risk budget allocation and objective."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from portfolio.optimizer.objectives import RiskParity
from portfolio.risk_budget import RiskBudget, RiskBudgetObjective


# ── helpers ──────────────────────────────────────────────────────────

_COV_EQUAL = {
    "A": {"A": 0.04, "B": 0.0},
    "B": {"A": 0.0, "B": 0.04},
}

_COV_DIFF = {
    "A": {"A": 0.04, "B": 0.0},
    "B": {"A": 0.0, "B": 0.09},
}


def _inp(cov: dict) -> SimpleNamespace:
    return SimpleNamespace(covariance=cov)


# ── RiskBudget dataclass ─────────────────────────────────────────────

class TestRiskBudget:
    def test_target_risk_contribution(self):
        rb = RiskBudget(budgets={"A": 0.6, "B": 0.4})
        assert rb.target_risk_contribution("A") == 0.6
        assert rb.target_risk_contribution("B") == 0.4
        assert rb.target_risk_contribution("C") == 0.0

    def test_total(self):
        rb = RiskBudget(budgets={"A": 0.6, "B": 0.4})
        assert rb.total == pytest.approx(1.0)

    def test_is_valid(self):
        assert RiskBudget(budgets={"A": 0.6, "B": 0.4}).is_valid()
        assert not RiskBudget(budgets={"A": 0.6, "B": 0.5}).is_valid()
        assert not RiskBudget(budgets={"A": -0.1, "B": 1.1}).is_valid()

    def test_equal_factory(self):
        rb = RiskBudget.equal(("A", "B", "C"))
        assert rb.target_risk_contribution("A") == pytest.approx(1 / 3)
        assert rb.total == pytest.approx(1.0)

    def test_equal_empty(self):
        rb = RiskBudget.equal(())
        assert rb.budgets == {}


# ── RiskBudgetObjective ──────────────────────────────────────────────

class TestRiskBudgetObjective:
    def test_equal_budget_matches_risk_parity(self):
        """Equal risk budget should behave like RiskParity for same weights."""
        budget = RiskBudget.equal(("A", "B"))
        rb_obj = RiskBudgetObjective(budget)
        rp_obj = RiskParity()

        # At equal weights with equal vol => both should be ~0
        w = {"A": 0.5, "B": 0.5}
        inp = _inp(_COV_EQUAL)

        rb_val = rb_obj.evaluate(w, inp)
        rp_val = rp_obj.evaluate(w, inp)

        assert rb_val == pytest.approx(rp_val, abs=1e-12)

    def test_equal_budget_skewed_weights(self):
        """Skewed weights should have > 0 objective for equal budget."""
        budget = RiskBudget.equal(("A", "B"))
        obj = RiskBudgetObjective(budget)

        w = {"A": 0.8, "B": 0.2}
        inp = _inp(_COV_EQUAL)

        assert obj.evaluate(w, inp) > 0.0

    def test_non_equal_budget_shifts_optimum(self):
        """Non-equal budget should favor the asset with higher budget.

        For uncorrelated equal-vol assets with budget A=0.7, B=0.3:
        The optimal weights should give A more risk contribution.
        At weights that match the budget, objective should be low.
        """
        budget = RiskBudget(budgets={"A": 0.7, "B": 0.3})
        obj = RiskBudgetObjective(budget)
        inp = _inp(_COV_EQUAL)

        # Equal weights: risk contributions are equal (50/50) vs target (70/30)
        val_equal = obj.evaluate({"A": 0.5, "B": 0.5}, inp)

        # Skewed toward A: should be closer to target
        # For uncorr equal-vol: RC_i ~ w_i^2 * sigma^2
        # Target: w_A^2 / (w_A^2 + w_B^2) = 0.7 => w_A/w_B = sqrt(7/3) ~ 1.528
        # w_A ~ 0.604, w_B ~ 0.396
        import math
        ratio = math.sqrt(0.7 / 0.3)
        w_a = ratio / (1 + ratio)
        w_b = 1 - w_a
        val_skewed = obj.evaluate({"A": w_a, "B": w_b}, inp)

        assert val_skewed < val_equal

    def test_empty_weights(self):
        budget = RiskBudget.equal(("A", "B"))
        obj = RiskBudgetObjective(budget)
        assert obj.evaluate({}, _inp(_COV_EQUAL)) == 0.0

    def test_zero_covariance(self):
        """Zero covariance -> total risk ~ 0 -> returns 0."""
        cov_zero = {"A": {"A": 0.0, "B": 0.0}, "B": {"A": 0.0, "B": 0.0}}
        budget = RiskBudget.equal(("A", "B"))
        obj = RiskBudgetObjective(budget)
        assert obj.evaluate({"A": 0.5, "B": 0.5}, _inp(cov_zero)) == 0.0

    def test_three_asset_custom_budget(self):
        """3 assets with custom budget. Verify objective decreases near target."""
        cov3 = {
            "A": {"A": 0.04, "B": 0.0, "C": 0.0},
            "B": {"A": 0.0, "B": 0.04, "C": 0.0},
            "C": {"A": 0.0, "B": 0.0, "C": 0.04},
        }
        budget = RiskBudget(budgets={"A": 0.5, "B": 0.3, "C": 0.2})
        obj = RiskBudgetObjective(budget)
        inp = _inp(cov3)

        # Equal weights: RC each = 1/3, far from target
        val_equal = obj.evaluate({"A": 1 / 3, "B": 1 / 3, "C": 1 / 3}, inp)
        assert val_equal > 0.0

        # Weights proportional to sqrt(budget) for equal vol uncorrelated
        import math
        raw = {s: math.sqrt(budget.target_risk_contribution(s)) for s in ("A", "B", "C")}
        total = sum(raw.values())
        w_target = {s: v / total for s, v in raw.items()}
        val_target = obj.evaluate(w_target, inp)

        assert val_target < val_equal
