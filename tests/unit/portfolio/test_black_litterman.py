# tests/unit/portfolio/test_black_litterman.py
"""Tests for Black-Litterman model."""
from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from portfolio.optimizer.black_litterman import (
    BlackLittermanConfig,
    BlackLittermanModel,
    BlackLittermanResult,
    ViewSpec,
)


# ── fixtures ─────────────────────────────────────────────────────────

_COV_2 = {
    "A": {"A": 0.04, "B": 0.006},
    "B": {"A": 0.006, "B": 0.09},
}

_COV_UNCORR = {
    "A": {"A": 0.04, "B": 0.0},
    "B": {"A": 0.0, "B": 0.09},
}

_MARKET_WEIGHTS = {"A": 0.6, "B": 0.4}


def _model(tau: float = 0.05, delta: float = 2.5) -> BlackLittermanModel:
    return BlackLittermanModel(BlackLittermanConfig(tau=tau, risk_aversion=delta))


# ── equilibrium returns ──────────────────────────────────────────────

class TestEquilibriumReturns:
    def test_basic_computation(self):
        """pi = delta * Sigma * w_mkt. Verify values manually."""
        model = _model(tau=0.05, delta=2.5)
        pi = model.equilibrium_returns(_MARKET_WEIGHTS, _COV_2)

        # pi_A = 2.5 * (0.04*0.6 + 0.006*0.4) = 2.5 * (0.024 + 0.0024) = 0.066
        # pi_B = 2.5 * (0.006*0.6 + 0.09*0.4)  = 2.5 * (0.0036 + 0.036) = 0.099
        assert pi["A"] == pytest.approx(0.066, abs=1e-10)
        assert pi["B"] == pytest.approx(0.099, abs=1e-10)

    def test_higher_risk_aversion_higher_returns(self):
        """Higher delta -> higher implied equilibrium returns."""
        low = _model(delta=1.0).equilibrium_returns(_MARKET_WEIGHTS, _COV_2)
        high = _model(delta=5.0).equilibrium_returns(_MARKET_WEIGHTS, _COV_2)
        assert high["A"] > low["A"]
        assert high["B"] > low["B"]

    def test_zero_weight_zero_return(self):
        """Zero weight for an asset => its return is still from cov with others."""
        model = _model()
        pi = model.equilibrium_returns({"A": 1.0, "B": 0.0}, _COV_2)
        # pi_A = 2.5 * (0.04*1 + 0.006*0) = 0.1
        assert pi["A"] == pytest.approx(0.1, abs=1e-10)
        # pi_B = 2.5 * (0.006*1 + 0.09*0) = 0.015
        assert pi["B"] == pytest.approx(0.015, abs=1e-10)

    def test_uncorrelated(self):
        """Uncorrelated assets: pi_i = delta * sigma_i^2 * w_i."""
        model = _model(delta=2.0)
        w = {"A": 0.5, "B": 0.5}
        pi = model.equilibrium_returns(w, _COV_UNCORR)
        assert pi["A"] == pytest.approx(2.0 * 0.04 * 0.5, abs=1e-12)
        assert pi["B"] == pytest.approx(2.0 * 0.09 * 0.5, abs=1e-12)


# ── posterior with views ─────────────────────────────────────────────

class TestPosterior:
    def test_no_views_equals_equilibrium(self):
        """With no views, posterior = equilibrium."""
        model = _model()
        result = model.posterior(_MARKET_WEIGHTS, _COV_2, views=[])
        eq = model.equilibrium_returns(_MARKET_WEIGHTS, _COV_2)

        for s in ("A", "B"):
            assert result.posterior_returns[s] == pytest.approx(eq[s], abs=1e-12)
            assert result.equilibrium_returns[s] == pytest.approx(eq[s], abs=1e-12)

    def test_single_absolute_view(self):
        """An absolute bullish view on A should increase A's posterior return."""
        model = _model()
        eq = model.equilibrium_returns(_MARKET_WEIGHTS, _COV_2)

        # Bullish view: A will return 15% (higher than equilibrium ~6.6%)
        view = ViewSpec(
            assets=("A",),
            weights=(1.0,),
            expected_return=0.15,
            confidence=2.0,
        )
        result = model.posterior(_MARKET_WEIGHTS, _COV_2, views=[view])

        # Posterior return for A should be pulled toward the view
        assert result.posterior_returns["A"] > eq["A"]

    def test_relative_view(self):
        """Relative view: A outperforms B. A's posterior should increase relative to B."""
        model = _model()
        eq = model.equilibrium_returns(_MARKET_WEIGHTS, _COV_2)

        # View: A outperforms B by 5%
        view = ViewSpec(
            assets=("A", "B"),
            weights=(1.0, -1.0),
            expected_return=0.05,
            confidence=1.0,
        )
        result = model.posterior(_MARKET_WEIGHTS, _COV_2, views=[view])

        # The spread should widen in favor of A
        eq_spread = eq["A"] - eq["B"]
        post_spread = result.posterior_returns["A"] - result.posterior_returns["B"]
        assert post_spread > eq_spread

    def test_posterior_differs_from_equilibrium(self):
        """Any non-trivial view should shift the posterior away from equilibrium."""
        model = _model()
        view = ViewSpec(
            assets=("B",),
            weights=(1.0,),
            expected_return=0.20,
            confidence=1.0,
        )
        result = model.posterior(_MARKET_WEIGHTS, _COV_2, views=[view])

        # At least one posterior return should differ from equilibrium
        diffs = [
            abs(result.posterior_returns[s] - result.equilibrium_returns[s])
            for s in ("A", "B")
        ]
        assert max(diffs) > 1e-6

    def test_high_confidence_pulls_more(self):
        """Higher confidence should pull posterior closer to the view."""
        model = _model()

        view_low = ViewSpec(assets=("A",), weights=(1.0,), expected_return=0.20, confidence=0.5)
        view_high = ViewSpec(assets=("A",), weights=(1.0,), expected_return=0.20, confidence=5.0)

        result_low = model.posterior(_MARKET_WEIGHTS, _COV_2, views=[view_low])
        result_high = model.posterior(_MARKET_WEIGHTS, _COV_2, views=[view_high])

        # High confidence should pull A's return closer to 0.20
        dist_low = abs(result_low.posterior_returns["A"] - 0.20)
        dist_high = abs(result_high.posterior_returns["A"] - 0.20)
        assert dist_high < dist_low

    def test_result_has_posterior_covariance(self):
        """Posterior result should contain a valid covariance matrix."""
        model = _model()
        view = ViewSpec(assets=("A",), weights=(1.0,), expected_return=0.10, confidence=1.0)
        result = model.posterior(_MARKET_WEIGHTS, _COV_2, views=[view])

        # Covariance should be symmetric and have positive diagonal
        for s1 in ("A", "B"):
            assert result.posterior_covariance[s1][s1] > 0
            for s2 in ("A", "B"):
                assert result.posterior_covariance[s1][s2] == pytest.approx(
                    result.posterior_covariance[s2][s1], abs=1e-12,
                )
