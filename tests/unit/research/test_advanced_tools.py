# tests/unit/research/test_advanced_tools.py
"""Tests for advanced research tools: Monte Carlo, Sensitivity, Significance."""
from __future__ import annotations


import pytest

from research.monte_carlo import MonteCarloSimulator
from research.sensitivity import SensitivityAnalyzer
from research.significance import (
    ic_ttest,
    minimum_oos_period,
    multiple_testing_correction,
)


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------

class TestMonteCarloSimulator:
    def test_bootstrap_correct_number_of_paths(self):
        sim = MonteCarloSimulator(seed=1)
        returns = [0.001] * 100
        result = sim.simulate_paths(returns, n_paths=500, horizon=60, method="bootstrap")
        assert result.paths == 500

    def test_parametric_known_params(self):
        sim = MonteCarloSimulator(seed=42)
        # Strong positive drift — most paths should end above 1.0
        returns = [0.01] * 200
        result = sim.simulate_paths(returns, n_paths=1000, horizon=100, method="parametric")
        assert result.mean_final > 1.0
        assert result.prob_loss < 0.5

    def test_prob_loss_for_negative_returns(self):
        sim = MonteCarloSimulator(seed=7)
        returns = [-0.005] * 200
        result = sim.simulate_paths(returns, n_paths=500, horizon=100, method="bootstrap")
        assert result.prob_loss > 0.5

    def test_block_bootstrap_preserves_path_length(self):
        sim = MonteCarloSimulator(seed=3)
        returns = [0.001 * (i % 10 - 5) for i in range(100)]
        for horizon in [10, 50, 100]:
            path = sim.simulate_single_path(returns, horizon, method="bootstrap")
            assert len(path) == horizon + 1
            assert path[0] == 1.0

    def test_parametric_path_length(self):
        sim = MonteCarloSimulator(seed=4)
        returns = [0.001] * 50
        path = sim.simulate_single_path(returns, 30, method="parametric")
        assert len(path) == 31
        assert path[0] == 1.0

    def test_percentiles_ordered(self):
        sim = MonteCarloSimulator(seed=10)
        returns = [0.002 * ((-1) ** i) for i in range(200)]
        result = sim.simulate_paths(returns, n_paths=1000, horizon=100)
        assert result.percentile_5 <= result.median_final <= result.percentile_95

    def test_max_drawdown_positive(self):
        sim = MonteCarloSimulator(seed=11)
        returns = [0.01, -0.02, 0.01, -0.03, 0.02] * 40
        result = sim.simulate_paths(returns, n_paths=200, horizon=50)
        assert result.max_drawdown_mean > 0
        assert result.max_drawdown_95 >= result.max_drawdown_mean

    def test_prob_target(self):
        sim = MonteCarloSimulator(seed=5)
        returns = [0.01] * 200
        result = sim.simulate_paths(
            returns, n_paths=500, horizon=100,
            method="parametric", target_return=0.5,
        )
        assert 0.0 <= result.prob_target <= 1.0

    def test_empty_returns(self):
        sim = MonteCarloSimulator(seed=0)
        result = sim.simulate_paths([], n_paths=100, horizon=50)
        assert result.paths == 0
        assert result.mean_final == 1.0

    def test_single_return(self):
        sim = MonteCarloSimulator(seed=6)
        result = sim.simulate_paths([0.01], n_paths=100, horizon=20, method="bootstrap")
        assert result.paths == 100
        # All returns are 0.01, so final should be (1.01)^20
        expected = 1.01 ** 20
        assert result.mean_final == pytest.approx(expected, rel=0.01)


# ---------------------------------------------------------------------------
# Sensitivity
# ---------------------------------------------------------------------------

class TestSensitivityAnalyzer:
    def test_single_param_sweep(self):
        def evaluate(params):
            return -((params["x"] - 5) ** 2) + 10  # peak at x=5

        analyzer = SensitivityAnalyzer()
        results = analyzer.analyze(
            base_params={"x": 0},
            param_ranges={"x": [1, 3, 5, 7, 9]},
            evaluate_fn=evaluate,
        )
        assert len(results) == 1
        r = results[0]
        assert r.param_name == "x"
        assert r.best_value == 5
        assert r.best_metric == 10.0
        assert len(r.param_values) == 5

    def test_multi_param_ranking(self):
        def evaluate(params):
            # y has much higher sensitivity than x
            return params["x"] * 0.1 + params["y"] * 10.0

        analyzer = SensitivityAnalyzer()
        results = analyzer.analyze(
            base_params={"x": 1, "y": 1},
            param_ranges={
                "x": [1, 2, 3, 4, 5],
                "y": [1, 2, 3, 4, 5],
            },
            evaluate_fn=evaluate,
        )
        ranking = analyzer.rank_parameters(results)
        assert len(ranking) == 2
        # y should be ranked more sensitive
        assert ranking[0][0] == "y"

    def test_constant_metric_zero_sensitivity(self):
        def evaluate(params):
            return 42.0

        analyzer = SensitivityAnalyzer()
        results = analyzer.analyze(
            base_params={"x": 0},
            param_ranges={"x": [1, 2, 3]},
            evaluate_fn=evaluate,
        )
        assert results[0].sensitivity_score == 0.0

    def test_empty_param_range(self):
        analyzer = SensitivityAnalyzer()
        results = analyzer.analyze(
            base_params={"x": 1},
            param_ranges={"x": []},
            evaluate_fn=lambda p: p["x"],
        )
        assert len(results) == 0

    def test_sensitivity_score_positive(self):
        def evaluate(params):
            return params["x"] ** 2

        analyzer = SensitivityAnalyzer()
        results = analyzer.analyze(
            base_params={"x": 0},
            param_ranges={"x": [1, 2, 3, 4, 5]},
            evaluate_fn=evaluate,
        )
        assert results[0].sensitivity_score > 0


# ---------------------------------------------------------------------------
# Significance — t-test
# ---------------------------------------------------------------------------

class TestICTTest:
    def test_significant_factor(self):
        # High mean, low std — clearly significant
        ic_series = [0.05 + 0.001 * i for i in range(100)]
        result = ic_ttest("strong_factor", ic_series)
        assert result.is_significant
        assert result.p_value < 0.01
        assert result.t_stat > 0
        assert result.n_observations == 100

    def test_insignificant_factor(self):
        # Mean near zero, high variance
        ic_series = [0.1 * ((-1) ** i) for i in range(100)]
        result = ic_ttest("noise_factor", ic_series)
        assert not result.is_significant
        assert result.p_value > 0.05

    def test_empty_series(self):
        result = ic_ttest("empty", [])
        assert result.p_value == 1.0
        assert not result.is_significant
        assert result.n_observations == 0

    def test_single_value(self):
        result = ic_ttest("single", [0.05])
        assert result.p_value == 1.0
        assert result.n_observations == 1

    def test_constant_nonzero_series(self):
        result = ic_ttest("const", [0.1] * 50)
        # All same value, mean != 0 → significant
        assert result.is_significant
        assert result.mean_ic == pytest.approx(0.1)

    def test_mean_ic_and_std(self):
        ic_series = [0.05, 0.06, 0.04, 0.05, 0.07]
        result = ic_ttest("test", ic_series)
        expected_mean = sum(ic_series) / len(ic_series)
        assert result.mean_ic == pytest.approx(expected_mean)
        assert result.std_ic > 0


# ---------------------------------------------------------------------------
# Significance — multiple testing
# ---------------------------------------------------------------------------

class TestMultipleTestingCorrection:
    def _make_results(self):
        """Create a mix of significant and non-significant results."""
        return [
            ic_ttest("sig1", [0.05 + 0.001 * i for i in range(100)]),
            ic_ttest("sig2", [0.04 + 0.001 * i for i in range(100)]),
            ic_ttest("noise1", [0.1 * ((-1) ** i) for i in range(100)]),
            ic_ttest("noise2", [0.05 * ((-1) ** i) for i in range(50)]),
        ]

    def test_bonferroni(self):
        results = self._make_results()
        corrected = multiple_testing_correction(results, method="bonferroni")
        assert corrected.method == "bonferroni"
        assert corrected.n_significant <= sum(1 for r in results if r.is_significant)

    def test_holm(self):
        results = self._make_results()
        corrected = multiple_testing_correction(results, method="holm")
        assert corrected.method == "holm"

    def test_fdr_bh(self):
        results = self._make_results()
        corrected = multiple_testing_correction(results, method="fdr_bh")
        assert corrected.method == "fdr_bh"

    def test_bonferroni_more_conservative_than_holm(self):
        results = self._make_results()
        bonf = multiple_testing_correction(results, method="bonferroni")
        holm = multiple_testing_correction(results, method="holm")
        # Bonferroni <= Holm in number of rejections
        assert bonf.n_significant <= holm.n_significant

    def test_bonferroni_more_conservative_than_fdr(self):
        results = self._make_results()
        bonf = multiple_testing_correction(results, method="bonferroni")
        fdr = multiple_testing_correction(results, method="fdr_bh")
        assert bonf.n_significant <= fdr.n_significant

    def test_empty_results(self):
        corrected = multiple_testing_correction([], method="holm")
        assert corrected.n_significant == 0
        assert corrected.significant_factors == ()

    def test_unknown_method_raises(self):
        results = self._make_results()
        with pytest.raises(ValueError, match="Unknown method"):
            multiple_testing_correction(results, method="invalid")

    def test_significant_factors_listed(self):
        results = self._make_results()
        corrected = multiple_testing_correction(results, method="fdr_bh")
        for name in corrected.significant_factors:
            assert any(r.factor_name == name for r in corrected.results if r.is_significant)


# ---------------------------------------------------------------------------
# Significance — minimum OOS period
# ---------------------------------------------------------------------------

class TestMinimumOOSPeriod:
    def test_increases_with_n_trials(self):
        days_10 = minimum_oos_period(n_trials=10)
        days_100 = minimum_oos_period(n_trials=100)
        days_1000 = minimum_oos_period(n_trials=1000)
        assert days_10 < days_100 < days_1000

    def test_decreases_with_higher_sharpe(self):
        days_low = minimum_oos_period(n_trials=50, target_sharpe=0.5)
        days_high = minimum_oos_period(n_trials=50, target_sharpe=2.0)
        assert days_high < days_low

    def test_single_trial(self):
        days = minimum_oos_period(n_trials=1, target_sharpe=1.0)
        assert days > 0

    def test_zero_sharpe_returns_zero(self):
        days = minimum_oos_period(n_trials=10, target_sharpe=0.0)
        assert days == 0

    def test_returns_positive_int(self):
        days = minimum_oos_period(n_trials=20, target_sharpe=1.5)
        assert isinstance(days, int)
        assert days >= 1
