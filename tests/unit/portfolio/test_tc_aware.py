# tests/unit/portfolio/test_tc_aware.py
"""Tests for transaction-cost-aware objective."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from portfolio.optimizer.objectives import MinVariance
from portfolio.optimizer.tc_aware import TransactionCostAwareObjective, TransactionCostConfig


# ── helpers ──────────────────────────────────────────────────────────

_COV = {
    "A": {"A": 0.04, "B": 0.0},
    "B": {"A": 0.0, "B": 0.04},
}


def _inp(current_weights: dict, total_equity: float = 100_000.0) -> SimpleNamespace:
    return SimpleNamespace(
        expected_returns={"A": 0.1, "B": 0.05},
        covariance=_COV,
        current_weights=current_weights,
        total_equity=total_equity,
    )


# ── tests ────────────────────────────────────────────────────────────

class TestTransactionCostPenalty:
    def test_zero_turnover_no_penalty(self):
        """When new weights == current weights, TC penalty should be zero."""
        inner = MinVariance()
        tc_obj = TransactionCostAwareObjective(inner)

        weights = {"A": 0.5, "B": 0.5}
        inp = _inp(current_weights={"A": 0.5, "B": 0.5})

        base_val = inner.evaluate(weights, inp)
        tc_val = tc_obj.evaluate(weights, inp)

        assert tc_val == pytest.approx(base_val, abs=1e-12)

    def test_high_turnover_increases_objective(self):
        """Large weight changes should increase the objective value."""
        inner = MinVariance()
        tc_obj = TransactionCostAwareObjective(inner)

        new_weights = {"A": 0.9, "B": 0.1}
        inp = _inp(current_weights={"A": 0.1, "B": 0.9})

        base_val = inner.evaluate(new_weights, inp)
        tc_val = tc_obj.evaluate(new_weights, inp)

        assert tc_val > base_val

    def test_higher_fee_higher_penalty(self):
        """Higher fee_bps should produce a larger penalty."""
        inner = MinVariance()
        low_fee = TransactionCostAwareObjective(inner, TransactionCostConfig(fee_bps=1.0))
        high_fee = TransactionCostAwareObjective(inner, TransactionCostConfig(fee_bps=20.0))

        new_weights = {"A": 0.8, "B": 0.2}
        inp = _inp(current_weights={"A": 0.2, "B": 0.8})

        assert high_fee.evaluate(new_weights, inp) > low_fee.evaluate(new_weights, inp)

    def test_no_current_weights_no_penalty(self):
        """When input_data has no current_weights, return base value."""
        inner = MinVariance()
        tc_obj = TransactionCostAwareObjective(inner)

        weights = {"A": 0.5, "B": 0.5}
        inp = SimpleNamespace(expected_returns={}, covariance=_COV)

        base_val = inner.evaluate(weights, inp)
        tc_val = tc_obj.evaluate(weights, inp)

        assert tc_val == pytest.approx(base_val, abs=1e-12)


class TestTurnoverEstimation:
    def test_zero_turnover(self):
        inner = MinVariance()
        tc_obj = TransactionCostAwareObjective(inner)

        turnover = tc_obj.estimate_turnover(
            {"A": 0.5, "B": 0.5},
            {"A": 0.5, "B": 0.5},
        )
        assert turnover == pytest.approx(0.0)

    def test_full_turnover(self):
        """Swapping entirely: |0.0-1.0| + |1.0-0.0| = 2.0."""
        inner = MinVariance()
        tc_obj = TransactionCostAwareObjective(inner)

        turnover = tc_obj.estimate_turnover(
            {"A": 0.0, "B": 1.0},
            {"A": 1.0, "B": 0.0},
        )
        assert turnover == pytest.approx(2.0)

    def test_partial_turnover(self):
        inner = MinVariance()
        tc_obj = TransactionCostAwareObjective(inner)

        turnover = tc_obj.estimate_turnover(
            {"A": 0.6, "B": 0.4},
            {"A": 0.5, "B": 0.5},
        )
        assert turnover == pytest.approx(0.2)


class TestCostBreakdown:
    def test_breakdown_components(self):
        inner = MinVariance()
        tc_obj = TransactionCostAwareObjective(inner, TransactionCostConfig(fee_bps=10.0))

        breakdown = tc_obj.estimate_cost_breakdown(
            {"A": 0.8, "B": 0.2},
            {"A": 0.3, "B": 0.7},
            _inp({"A": 0.3, "B": 0.7}),
        )

        assert "turnover" in breakdown
        assert "fees" in breakdown
        assert "impact" in breakdown
        assert "total" in breakdown
        assert breakdown["turnover"] == pytest.approx(1.0)
        assert breakdown["fees"] > 0
        assert breakdown["total"] > 0
