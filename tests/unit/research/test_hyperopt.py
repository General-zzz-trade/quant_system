"""Tests for hyperparameter optimization: search space, pruner, optimizer."""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from research.hyperopt.optimizer import HyperOptConfig, HyperOptimizer, HyperOptResult
from research.hyperopt.pruner import TimeSeriesPruner
from research.hyperopt.search_space import CategoricalParam, ParamRange, SearchSpace


class TestSearchSpace:
    def test_suggest_float_params(self):
        space = SearchSpace(
            name="test",
            float_params=(
                ParamRange(name="lr", low=0.001, high=0.1),
                ParamRange(name="momentum", low=0.5, high=0.99),
            ),
        )
        trial = MagicMock()
        trial.suggest_float.side_effect = [0.01, 0.9]

        params = space.suggest(trial)

        assert params == {"lr": 0.01, "momentum": 0.9}
        assert trial.suggest_float.call_count == 2

    def test_suggest_int_params(self):
        space = SearchSpace(
            name="test",
            int_params=(ParamRange(name="window", low=5, high=50),),
        )
        trial = MagicMock()
        trial.suggest_int.return_value = 20

        params = space.suggest(trial)

        assert params == {"window": 20}
        trial.suggest_int.assert_called_once_with("window", 5, 50)

    def test_suggest_categorical_params(self):
        space = SearchSpace(
            name="test",
            categorical_params=(
                CategoricalParam(name="strategy", choices=("ma_cross", "rsi")),
            ),
        )
        trial = MagicMock()
        trial.suggest_categorical.return_value = "rsi"

        params = space.suggest(trial)

        assert params == {"strategy": "rsi"}

    def test_suggest_combined_params(self):
        space = SearchSpace(
            name="combined",
            float_params=(ParamRange(name="threshold", low=0.0, high=1.0),),
            int_params=(ParamRange(name="period", low=5, high=30),),
            categorical_params=(CategoricalParam(name="mode", choices=("fast", "slow")),),
        )
        trial = MagicMock()
        trial.suggest_float.return_value = 0.5
        trial.suggest_int.return_value = 14
        trial.suggest_categorical.return_value = "fast"

        params = space.suggest(trial)

        assert params == {"threshold": 0.5, "period": 14, "mode": "fast"}

    def test_sample_random(self):
        import random
        space = SearchSpace(
            name="test",
            float_params=(ParamRange(name="lr", low=0.01, high=0.1),),
            int_params=(ParamRange(name="n", low=1, high=10),),
            categorical_params=(CategoricalParam(name="x", choices=("a", "b")),),
        )
        rng = random.Random(42)
        params = space.sample_random(rng)

        assert 0.01 <= params["lr"] <= 0.1
        assert 1 <= params["n"] <= 10
        assert params["x"] in ("a", "b")

    def test_dimension_count(self):
        space = SearchSpace(
            name="test",
            float_params=(ParamRange(name="a", low=0, high=1),),
            int_params=(ParamRange(name="b", low=0, high=10),),
            categorical_params=(CategoricalParam(name="c", choices=("x",)),),
        )
        assert space.dimension_count == 3

    def test_empty_space(self):
        space = SearchSpace(name="empty")
        trial = MagicMock()
        params = space.suggest(trial)
        assert params == {}
        assert space.dimension_count == 0


class TestTimeSeriesPruner:
    def test_no_prune_below_min_trials(self):
        pruner = TimeSeriesPruner(min_trials=5, patience=3)
        for i in range(5):
            assert not pruner.should_prune(0.5, i)

    def test_prune_after_patience_exhausted(self):
        pruner = TimeSeriesPruner(min_trials=2, patience=3, min_improvement=0.01)
        pruner.should_prune(1.0, 0)
        pruner.should_prune(1.0, 1)
        pruner.should_prune(1.05, 2)
        assert not pruner.should_prune(1.04, 3)
        assert not pruner.should_prune(1.04, 4)
        assert pruner.should_prune(1.04, 5)

    def test_improvement_resets_counter(self):
        pruner = TimeSeriesPruner(min_trials=2, patience=2, min_improvement=0.01)
        pruner.should_prune(1.0, 0)
        pruner.should_prune(1.0, 1)
        pruner.should_prune(0.9, 2)
        assert not pruner.should_prune(1.5, 3)
        assert pruner.trials_without_improvement == 0

    def test_reset(self):
        pruner = TimeSeriesPruner(min_trials=1, patience=2)
        pruner.should_prune(1.0, 0)
        pruner.should_prune(1.0, 1)
        pruner.reset()
        assert pruner.best_value is None
        assert pruner.trials_without_improvement == 0

    def test_best_value_tracked(self):
        pruner = TimeSeriesPruner(min_trials=0, patience=100, min_improvement=0.0)
        pruner.should_prune(1.0, 0)
        pruner.should_prune(2.0, 1)
        pruner.should_prune(1.5, 2)
        assert pruner.best_value == 2.0


class TestHyperOptimizer:
    def test_random_search_fallback(self):
        space = SearchSpace(
            name="test",
            float_params=(ParamRange(name="x", low=-5.0, high=5.0),),
        )

        def objective(params: dict[str, Any]) -> float:
            x = params["x"]
            return -(x - 1.0) ** 2

        config = HyperOptConfig(n_trials=50, seed=42, direction="maximize")
        optimizer = HyperOptimizer(
            search_space=space,
            objective_fn=objective,
            config=config,
        )
        result = optimizer._random_search()

        assert result.n_trials > 0
        assert result.best_value > -25.0
        assert "x" in result.best_params
        assert len(result.all_trials) == result.n_trials

    def test_random_search_minimize(self):
        space = SearchSpace(
            name="test",
            float_params=(ParamRange(name="x", low=-5.0, high=5.0),),
        )

        config = HyperOptConfig(n_trials=30, seed=123, direction="minimize")
        optimizer = HyperOptimizer(
            search_space=space,
            objective_fn=lambda p: p["x"] ** 2,
            config=config,
        )
        result = optimizer._random_search()

        assert result.n_trials > 0
        assert result.best_value < 25.0

    def test_random_search_handles_failing_trials(self):
        space = SearchSpace(
            name="test",
            float_params=(ParamRange(name="x", low=0.0, high=1.0),),
        )
        call_count = 0

        def flaky_objective(params):
            nonlocal call_count
            call_count += 1
            if call_count % 2 == 0:
                raise ValueError("boom")
            return params["x"]

        config = HyperOptConfig(n_trials=10, seed=42, direction="maximize")
        optimizer = HyperOptimizer(
            search_space=space,
            objective_fn=flaky_objective,
            config=config,
        )
        result = optimizer._random_search()

        completed = [t for t in result.all_trials if t["state"] == "COMPLETE"]
        failed = [t for t in result.all_trials if t["state"] == "FAIL"]
        assert len(completed) > 0
        assert len(failed) > 0

    def test_result_dataclass(self):
        result = HyperOptResult(
            best_params={"x": 1.0},
            best_value=0.5,
            n_trials=10,
            all_trials=(),
        )
        assert result.best_params == {"x": 1.0}
        assert result.best_value == 0.5
