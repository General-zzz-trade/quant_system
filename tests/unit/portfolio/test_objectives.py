# tests/unit/portfolio/test_objectives.py
"""Tests for portfolio optimization objectives — mathematical correctness."""
from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from portfolio.optimizer.objectives import (
    MaxSharpe,
    MinVariance,
    RiskParity,
    _portfolio_variance,
)


# ── helpers ──────────────────────────────────────────────────────

def _inp(er: dict, cov: dict) -> SimpleNamespace:
    return SimpleNamespace(expected_returns=er, covariance=cov)


# 2-asset universe: uncorrelated, equal vol (sigma=0.2)
_COV_EQUAL = {
    "A": {"A": 0.04, "B": 0.0},
    "B": {"A": 0.0, "B": 0.04},
}

# 2-asset: different vol, some correlation
_COV_ASYM = {
    "A": {"A": 0.04, "B": 0.01},
    "B": {"A": 0.01, "B": 0.09},
}


# ── _portfolio_variance ─────────────────────────────────────────

class TestPortfolioVariance:
    def test_single_asset(self):
        cov = {"X": {"X": 0.16}}
        assert _portfolio_variance({"X": 1.0}, cov) == pytest.approx(0.16)

    def test_equal_weight_uncorrelated(self):
        # var = 0.5^2*0.04 + 0.5^2*0.04 = 0.02
        v = _portfolio_variance({"A": 0.5, "B": 0.5}, _COV_EQUAL)
        assert v == pytest.approx(0.02)

    def test_equal_weight_correlated(self):
        # var = 0.5^2*0.04 + 2*0.5*0.5*0.01 + 0.5^2*0.09 = 0.01+0.005+0.0225 = 0.0375
        v = _portfolio_variance({"A": 0.5, "B": 0.5}, _COV_ASYM)
        assert v == pytest.approx(0.0375)

    def test_zero_weights(self):
        assert _portfolio_variance({"A": 0.0, "B": 0.0}, _COV_EQUAL) == 0.0


# ── MaxSharpe ────────────────────────────────────────────────────

class TestMaxSharpe:
    def test_analytical_two_asset(self):
        """2-asset analytical: Sharpe = (w'mu) / sqrt(w'Cov*w), evaluate returns negative."""
        obj = MaxSharpe()
        er = {"A": 0.10, "B": 0.20}
        w = {"A": 0.5, "B": 0.5}
        inp = _inp(er, _COV_EQUAL)

        port_ret = 0.5 * 0.10 + 0.5 * 0.20  # 0.15
        port_vol = math.sqrt(0.02)  # ~0.1414
        expected_neg_sharpe = -port_ret / port_vol

        assert obj.evaluate(w, inp) == pytest.approx(expected_neg_sharpe)

    def test_higher_return_better_sharpe(self):
        """Higher return portfolio should have lower (more negative) objective."""
        obj = MaxSharpe()
        inp = _inp({"A": 0.05, "B": 0.20}, _COV_EQUAL)

        val_a = obj.evaluate({"A": 1.0, "B": 0.0}, inp)  # 100% low-return
        val_b = obj.evaluate({"A": 0.0, "B": 1.0}, inp)  # 100% high-return

        # Same vol (single asset, same cov), but B has higher return → more negative
        assert val_b < val_a

    def test_risk_free_rate(self):
        """Risk-free rate shifts the numerator."""
        er = {"A": 0.10, "B": 0.10}
        w = {"A": 0.5, "B": 0.5}
        inp = _inp(er, _COV_EQUAL)

        obj_0 = MaxSharpe(risk_free_rate=0.0)
        obj_5 = MaxSharpe(risk_free_rate=0.05)

        # With rf=0.05, numerator is 0.10 - 0.05 = 0.05 vs 0.10
        assert obj_5.evaluate(w, inp) > obj_0.evaluate(w, inp)  # less negative

    def test_zero_return_zero_sharpe(self):
        """Zero expected return → Sharpe = 0 → evaluate = 0."""
        obj = MaxSharpe()
        w = {"A": 0.5, "B": 0.5}
        inp = _inp({"A": 0.0, "B": 0.0}, _COV_EQUAL)
        assert obj.evaluate(w, inp) == pytest.approx(0.0)


# ── MinVariance ──────────────────────────────────────────────────

class TestMinVariance:
    def test_matches_portfolio_variance(self):
        obj = MinVariance()
        w = {"A": 0.6, "B": 0.4}
        inp = _inp({}, _COV_ASYM)
        assert obj.evaluate(w, inp) == pytest.approx(
            _portfolio_variance(w, _COV_ASYM)
        )

    def test_minimum_variance_portfolio_analytical(self):
        """For 2 uncorrelated equal-vol assets, MVP is equal weight."""
        obj = MinVariance()
        inp = _inp({}, _COV_EQUAL)

        val_eq = obj.evaluate({"A": 0.5, "B": 0.5}, inp)
        val_skew = obj.evaluate({"A": 0.8, "B": 0.2}, inp)

        assert val_eq < val_skew


# ── RiskParity ───────────────────────────────────────────────────

class TestRiskParity:
    def test_equal_weight_equal_vol_is_zero(self):
        """Equal weight + equal uncorrelated vol → perfect risk parity → objective = 0."""
        obj = RiskParity()
        w = {"A": 0.5, "B": 0.5}
        inp = _inp({}, _COV_EQUAL)
        assert obj.evaluate(w, inp) == pytest.approx(0.0, abs=1e-15)

    def test_skewed_weight_nonzero(self):
        """Unequal weights with equal vol → not risk parity → objective > 0."""
        obj = RiskParity()
        w = {"A": 0.8, "B": 0.2}
        inp = _inp({}, _COV_EQUAL)
        assert obj.evaluate(w, inp) > 0.0

    def test_risk_parity_with_different_vols(self):
        """For uncorrelated assets with sigma_A=0.2, sigma_B=0.3:
        RP weights: w_A = (1/sigma_A) / sum(1/sigma_i), w_B = (1/sigma_B) / sum(1/sigma_i)
        → w_A = 5/(5+10/3) = 0.6, w_B = 0.4 ...
        Actually for uncorrelated: RC_i = w_i^2 * sigma_i^2
        Equal RC: w_A^2 * 0.04 = w_B^2 * 0.09 → w_A/w_B = 3/2 = 1.5
        With w_A + w_B = 1: w_A = 0.6, w_B = 0.4
        """
        cov_diff = {
            "A": {"A": 0.04, "B": 0.0},
            "B": {"A": 0.0, "B": 0.09},
        }
        obj = RiskParity()
        inp = _inp({}, cov_diff)

        # Risk parity weights
        val_rp = obj.evaluate({"A": 0.6, "B": 0.4}, inp)
        # Equal weights (not RP for different vols)
        val_eq = obj.evaluate({"A": 0.5, "B": 0.5}, inp)

        assert val_rp < val_eq  # RP weights are closer to objective
        assert val_rp == pytest.approx(0.0, abs=1e-10)

    def test_three_asset_equal_vol(self):
        """3 uncorrelated equal-vol assets: RP = equal weight = 1/3 each."""
        cov3 = {
            "A": {"A": 0.04, "B": 0.0, "C": 0.0},
            "B": {"A": 0.0, "B": 0.04, "C": 0.0},
            "C": {"A": 0.0, "B": 0.0, "C": 0.04},
        }
        obj = RiskParity()
        inp = _inp({}, cov3)
        w = {"A": 1/3, "B": 1/3, "C": 1/3}
        assert obj.evaluate(w, inp) == pytest.approx(0.0, abs=1e-15)

    def test_empty_weights(self):
        obj = RiskParity()
        assert obj.evaluate({}, _inp({}, _COV_EQUAL)) == 0.0

    def test_zero_risk_returns_zero(self):
        """All-zero covariance → total_risk ≈ 0 → returns 0."""
        cov_zero = {"A": {"A": 0.0, "B": 0.0}, "B": {"A": 0.0, "B": 0.0}}
        obj = RiskParity()
        assert obj.evaluate({"A": 0.5, "B": 0.5}, _inp({}, cov_zero)) == 0.0
