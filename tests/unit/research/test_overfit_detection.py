# tests/unit/research/test_overfit_detection.py
"""Tests for overfit detection and combinatorial CV."""
from __future__ import annotations

import pytest

from research.overfit_detection import (
    deflated_sharpe_ratio,
    parameter_stability,
    probability_of_backtest_overfitting,
)
from research.combinatorial_cv import (
    combinatorial_purged_cv,
    _generate_purged_splits,
)


class TestDeflatedSharpeRatio:
    def test_high_sharpe_few_trials_significant(self):
        result = deflated_sharpe_ratio(
            observed_sharpe=2.5,
            n_trials=5,
            n_observations=500,
        )
        assert result.observed_sharpe == 2.5
        assert result.n_trials == 5
        # High Sharpe with few trials should be significant
        assert result.p_value < 0.5

    def test_low_sharpe_many_trials_not_significant(self):
        result = deflated_sharpe_ratio(
            observed_sharpe=0.5,
            n_trials=1000,
            n_observations=252,
        )
        # Low Sharpe with many trials likely not significant
        assert result.p_value > 0.05

    def test_single_trial_no_deflation(self):
        result = deflated_sharpe_ratio(
            observed_sharpe=1.0,
            n_trials=1,
            n_observations=252,
        )
        assert result.expected_max_sharpe == pytest.approx(0.0)

    def test_edge_case_zero_observations(self):
        result = deflated_sharpe_ratio(
            observed_sharpe=1.0,
            n_trials=10,
            n_observations=1,
        )
        assert result.p_value == 1.0
        assert not result.is_significant

    def test_significance_flag(self):
        result = deflated_sharpe_ratio(
            observed_sharpe=3.0,
            n_trials=3,
            n_observations=1000,
            significance=0.05,
        )
        # With very high Sharpe, should likely be significant
        if result.p_value < 0.05:
            assert result.is_significant


class TestParameterStability:
    def test_stable_parameter(self):
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        metrics = [0.9, 0.95, 1.0, 0.95, 0.9]  # smooth peak
        result = parameter_stability(values, metrics)
        assert result.stability_score > 0.5
        assert result.is_stable

    def test_unstable_parameter(self):
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        metrics = [0.1, 0.1, 1.0, 0.1, 0.1]  # sharp cliff
        result = parameter_stability(values, metrics)
        assert result.stability_score < 0.5
        assert not result.is_stable

    def test_too_few_values(self):
        result = parameter_stability([10.0, 20.0], [0.5, 1.0])
        assert result.stability_score == 0.0
        assert not result.is_stable

    def test_best_value_identified(self):
        values = [5.0, 10.0, 15.0, 20.0]
        metrics = [0.5, 0.8, 1.2, 0.6]
        result = parameter_stability(values, metrics)
        assert result.best_value == 15.0


class TestProbabilityOfBacktestOverfitting:
    def test_no_overfitting(self):
        # IS-best also OOS-best → low PBO
        is_sharpes = [0.5, 1.0, 1.5, 2.0]
        oos_sharpes = [0.4, 0.9, 1.4, 1.8]
        pbo = probability_of_backtest_overfitting(is_sharpes, oos_sharpes)
        assert pbo <= 0.5

    def test_overfitting(self):
        # IS-best is OOS-worst → high PBO
        is_sharpes = [0.5, 1.0, 1.5, 2.0]
        oos_sharpes = [1.8, 1.4, 0.9, 0.1]
        pbo = probability_of_backtest_overfitting(is_sharpes, oos_sharpes)
        assert pbo > 0.5

    def test_mismatched_lengths(self):
        with pytest.raises(ValueError):
            probability_of_backtest_overfitting([1.0, 2.0], [1.0])

    def test_single_strategy(self):
        pbo = probability_of_backtest_overfitting([1.0], [0.5])
        assert pbo == 0.5  # uninformative


class TestCombinatorialPurgedCV:
    def test_basic_cv(self):
        # Simple evaluator: returns average of data range
        def _eval(start, end):
            return (start + end) / 2000.0

        result = combinatorial_purged_cv(
            data_length=1200,
            evaluate_fn=_eval,
            n_groups=6,
            n_test_groups=2,
            purge_groups=1,
        )
        assert len(result.folds) > 0
        assert result.n_combinations > 0
        assert result.avg_metric > 0

    def test_purging_reduces_folds(self):
        # More purging → fewer valid folds
        splits_no_purge = _generate_purged_splits(6, 2, 0)
        splits_purge = _generate_purged_splits(6, 2, 1)
        # With purging, some splits have empty train sets
        assert len(splits_purge) <= len(splits_no_purge)

    def test_all_folds_have_train_data(self):
        splits = _generate_purged_splits(8, 2, 1)
        for train, test in splits:
            assert len(train) > 0
            assert len(test) == 2

    def test_std_metric_computed(self):
        result = combinatorial_purged_cv(
            data_length=1000,
            evaluate_fn=lambda s, e: float(e - s),
            n_groups=6,
            n_test_groups=2,
        )
        assert result.std_metric >= 0.0

    def test_eval_failure_handled(self):
        call_count = 0

        def _sometimes_fail(start, end):
            nonlocal call_count
            call_count += 1
            if call_count % 3 == 0:
                raise ValueError("oops")
            return 1.0

        result = combinatorial_purged_cv(
            data_length=600,
            evaluate_fn=_sometimes_fail,
            n_groups=6,
            n_test_groups=2,
        )
        # Some folds should still succeed
        assert len(result.folds) > 0
