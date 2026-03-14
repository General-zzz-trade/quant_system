"""Tests for portfolio/risk_model aggregation and tail-risk modules."""
from __future__ import annotations

import math

from portfolio.risk_model.aggregation.portfolio_risk import (
    compute_portfolio_risk,
)
from portfolio.risk_model.aggregation.marginal import (
    compute_marginal_risk,
)
from portfolio.risk_model.aggregation.decomposition import (
    decompose_risk,
)
from portfolio.risk_model.tail.var import (
    compute_var,
    historical_var,
    parametric_var,
)
from portfolio.risk_model.tail.es import (
    compute_es,
    historical_es,
    parametric_es,
)
from portfolio.risk_model.tail.drawdown import (
    analyze_drawdowns,
    compute_drawdowns,
)


# ── PortfolioRisk ──────────────────────────────────────────────────


class TestPortfolioRisk:
    def test_variance_computation(self):
        """Port variance = w' Sigma w, annualized."""
        weights = {"A": 0.6, "B": 0.4}
        cov = {
            "A": {"A": 0.04, "B": 0.01},
            "B": {"A": 0.01, "B": 0.09},
        }
        result = compute_portfolio_risk(weights, cov, annualization=1.0)
        # 0.6^2*0.04 + 2*0.6*0.4*0.01 + 0.4^2*0.09
        expected_var = 0.36 * 0.04 + 2 * 0.24 * 0.01 + 0.16 * 0.09
        assert abs(result.variance - expected_var) < 1e-12
        assert abs(result.volatility - math.sqrt(expected_var)) < 1e-12

    def test_var_and_es(self):
        weights = {"A": 0.5, "B": 0.5}
        cov = {
            "A": {"A": 0.04, "B": 0.01},
            "B": {"A": 0.01, "B": 0.04},
        }
        result = compute_portfolio_risk(weights, cov, annualization=1.0)
        assert abs(result.var_95 - result.volatility * 1.645) < 1e-12
        assert abs(result.var_99 - result.volatility * 2.326) < 1e-12
        assert abs(result.expected_shortfall_95 - result.volatility * 2.063) < 1e-12

    def test_annualization(self):
        weights = {"A": 1.0}
        cov = {"A": {"A": 0.01}}
        r1 = compute_portfolio_risk(weights, cov, annualization=1.0)
        r365 = compute_portfolio_risk(weights, cov, annualization=365.0)
        assert abs(r365.variance - r1.variance * 365.0) < 1e-12

    def test_zero_weight(self):
        weights = {"A": 0.0, "B": 0.0}
        cov = {"A": {"A": 0.04, "B": 0.01}, "B": {"A": 0.01, "B": 0.09}}
        result = compute_portfolio_risk(weights, cov, annualization=1.0)
        assert result.variance == 0.0
        assert result.volatility == 0.0


# ── MarginalContribution ──────────────────────────────────────────


class TestMarginalContribution:
    def test_risk_contribution(self):
        weights = {"A": 0.6, "B": 0.4}
        cov = {
            "A": {"A": 0.04, "B": 0.01},
            "B": {"A": 0.01, "B": 0.09},
        }
        results = compute_marginal_risk(weights, cov)
        assert len(results) == 2
        # Marginal variance for A = sum_j cov(A,j)*w_j
        mvar_a = 0.04 * 0.6 + 0.01 * 0.4
        assert abs(results[0].marginal_variance - mvar_a) < 1e-12

    def test_percentage_breakdown(self):
        weights = {"A": 0.5, "B": 0.5}
        cov = {
            "A": {"A": 0.04, "B": 0.0},
            "B": {"A": 0.0, "B": 0.04},
        }
        results = compute_marginal_risk(weights, cov)
        # Equal weights, equal variance, zero correlation => equal contribution
        pcts = [r.pct_contribution for r in results]
        assert abs(pcts[0] - pcts[1]) < 1e-12

    def test_contributions_sum(self):
        """Sum of risk contributions = portfolio volatility."""
        weights = {"A": 0.6, "B": 0.4}
        cov = {
            "A": {"A": 0.04, "B": 0.01},
            "B": {"A": 0.01, "B": 0.09},
        }
        results = compute_marginal_risk(weights, cov)
        total_rc = sum(r.risk_contribution for r in results)
        port_var = sum(
            weights[s1] * weights[s2] * cov[s1][s2]
            for s1 in weights for s2 in weights
        )
        port_vol = math.sqrt(port_var)
        assert abs(total_rc - port_vol) < 1e-10


# ── RiskDecomposition ─────────────────────────────────────────────


class TestRiskDecomposition:
    def test_systematic_idiosyncratic_split(self):
        weights = {"A": 0.6, "B": 0.4}
        cov = {
            "A": {"A": 0.04, "B": 0.01},
            "B": {"A": 0.01, "B": 0.09},
        }
        specific = {"A": 0.005, "B": 0.01}
        result = decompose_risk(weights, cov, specific_risk=specific)

        total_var = sum(
            weights[s1] * weights[s2] * cov[s1][s2]
            for s1 in weights for s2 in weights
        )
        idio = 0.6 ** 2 * 0.005 + 0.4 ** 2 * 0.01
        assert abs(result.total_variance - total_var) < 1e-12
        assert abs(result.idiosyncratic_pct - idio / total_var) < 1e-12
        assert abs(result.systematic_pct + result.idiosyncratic_pct - 1.0) < 1e-12

    def test_no_specific_risk(self):
        weights = {"A": 0.5, "B": 0.5}
        cov = {"A": {"A": 0.04, "B": 0.01}, "B": {"A": 0.01, "B": 0.04}}
        result = decompose_risk(weights, cov)
        assert result.idiosyncratic_pct == 0.0
        assert result.systematic_pct == 1.0

    def test_by_asset(self):
        weights = {"A": 0.5, "B": 0.5}
        cov = {"A": {"A": 0.04, "B": 0.01}, "B": {"A": 0.01, "B": 0.04}}
        result = decompose_risk(weights, cov)
        assert len(result.by_asset) == 2


# ── VaR ────────────────────────────────────────────────────────────


class TestVaR:
    def test_parametric_var(self):
        returns = [0.01, -0.02, 0.03, -0.01, 0.005, -0.015, 0.02, -0.025]
        mean = sum(returns) / len(returns)
        std = math.sqrt(sum((r - mean) ** 2 for r in returns) / (len(returns) - 1))
        result = parametric_var(returns, confidence=0.95)
        expected = mean - 1.645 * std
        assert abs(result - expected) < 1e-12

    def test_historical_var(self):
        returns = [-0.05, -0.03, -0.01, 0.01, 0.02, 0.03, 0.04, 0.05,
                   0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13,
                   0.14, 0.15, 0.16, 0.17]
        result = historical_var(returns, confidence=0.95)
        # 5% of 20 = 1, idx=1 => sorted_r[1] = -0.03
        assert abs(result - (-0.03)) < 1e-12

    def test_var_empty(self):
        assert parametric_var([], confidence=0.95) == 0.0
        assert historical_var([], confidence=0.95) == 0.0

    def test_compute_var_combined(self):
        returns = [0.01, -0.02, 0.03, -0.01, 0.005]
        result = compute_var(returns, confidence=0.95)
        assert result.confidence == 0.95
        assert result.method == "parametric+historical"


# ── Expected Shortfall ─────────────────────────────────────────────


class TestES:
    def test_historical_es_tail_average(self):
        returns = [-0.10, -0.05, -0.02, 0.01, 0.02, 0.03, 0.04, 0.05,
                   0.06, 0.07]
        result = historical_es(returns, confidence=0.90)
        # 10% of 10 = 1 element in tail
        assert abs(result - (-0.10)) < 1e-12

    def test_parametric_es(self):
        returns = [0.01, -0.02, 0.03, -0.01, 0.005, -0.015, 0.02, -0.025]
        result = parametric_es(returns, confidence=0.95)
        mean = sum(returns) / len(returns)
        std = math.sqrt(sum((r - mean) ** 2 for r in returns) / (len(returns) - 1))
        z = 1.645
        phi_z = math.exp(-0.5 * z ** 2) / math.sqrt(2 * math.pi)
        expected = mean - std * phi_z / 0.05
        assert abs(result - expected) < 1e-12

    def test_es_empty(self):
        assert historical_es([], confidence=0.95) == 0.0
        assert parametric_es([], confidence=0.95) == 0.0

    def test_compute_es_combined(self):
        returns = [0.01, -0.02, 0.03, -0.01, 0.005]
        result = compute_es(returns, confidence=0.95)
        assert result.confidence == 0.95


# ── DrawdownStats ──────────────────────────────────────────────────


class TestDrawdownStats:
    def test_max_drawdown(self):
        returns = [0.10, -0.05, -0.08, 0.02, 0.15]
        stats = analyze_drawdowns(returns, annualization=1.0)
        dds = compute_drawdowns(returns)
        assert abs(stats.max_drawdown - min(dds)) < 1e-12
        assert stats.max_drawdown < 0

    def test_duration(self):
        # Peak at cumsum=0.10, then 2 drops, then recovery
        returns = [0.10, -0.05, -0.03, 0.10, -0.01]
        stats = analyze_drawdowns(returns)
        # After return 2 and 3 we're in drawdown, return 4 recovers past peak
        # Then return 5 causes new drawdown of 1 period
        assert stats.max_drawdown_duration >= 2

    def test_calmar_ratio(self):
        returns = [0.01] * 365  # 1% daily for a year
        stats = analyze_drawdowns(returns, annualization=365.0)
        # No drawdown => calmar = 0 (max_dd = 0)
        assert stats.calmar_ratio == 0.0

    def test_calmar_with_drawdown(self):
        returns = [0.10, -0.20, 0.05, 0.10, -0.05]
        stats = analyze_drawdowns(returns, annualization=1.0)
        total_ret = sum(returns)
        ann_ret = total_ret * 1.0 / len(returns)
        assert abs(stats.calmar_ratio - ann_ret / abs(stats.max_drawdown)) < 1e-12

    def test_current_drawdown(self):
        returns = [0.10, -0.05, -0.03]
        stats = analyze_drawdowns(returns)
        dds = compute_drawdowns(returns)
        assert abs(stats.current_drawdown - dds[-1]) < 1e-12

    def test_empty_returns(self):
        stats = analyze_drawdowns([])
        assert stats.max_drawdown == 0.0
        assert stats.max_drawdown_duration == 0
        assert stats.calmar_ratio == 0.0
