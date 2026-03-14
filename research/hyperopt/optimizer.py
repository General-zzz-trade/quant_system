"""Optuna-based hyperparameter optimizer with random-search fallback."""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Any, Callable, Optional

from research.hyperopt.pruner import TimeSeriesPruner
from research.hyperopt.search_space import SearchSpace

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class HyperOptConfig:
    """Configuration for hyperparameter optimization."""
    n_trials: int = 100
    timeout_sec: Optional[float] = None
    direction: str = "maximize"
    sampler: str = "tpe"
    seed: Optional[int] = None
    pruner_patience: int = 20
    pruner_min_trials: int = 10


@dataclass(frozen=True, slots=True)
class HyperOptResult:
    """Outcome of a hyperparameter optimization run."""
    best_params: dict[str, Any]
    best_value: float
    n_trials: int
    all_trials: tuple[dict[str, Any], ...]


class HyperOptimizer:
    """Optuna-based hyperparameter optimizer for trading strategies.

    Falls back to random search when Optuna is not installed.

    Usage:
        space = SearchSpace(name="ma_cross", float_params=(...,))
        opt = HyperOptimizer(
            search_space=space,
            objective_fn=lambda params: run_backtest(params).sharpe,
        )
        result = opt.optimize()
        print(result.best_params, result.best_value)
    """

    def __init__(
        self,
        *,
        search_space: SearchSpace,
        objective_fn: Callable[[dict[str, Any]], float],
        config: HyperOptConfig = HyperOptConfig(),
    ) -> None:
        self._space = search_space
        self._objective_fn = objective_fn
        self._config = config

    def optimize(self) -> HyperOptResult:
        """Run optimization. Uses Optuna if available, falls back to random search."""
        try:
            import optuna  # noqa: F401
            return self._optuna_optimize()
        except ImportError:
            logger.warning("optuna not installed, falling back to random search")
            return self._random_search()

    def _optuna_optimize(self) -> HyperOptResult:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        sampler = self._create_sampler()
        study = optuna.create_study(
            direction=self._config.direction,
            sampler=sampler,
        )

        pruner = TimeSeriesPruner(
            min_trials=self._config.pruner_min_trials,
            patience=self._config.pruner_patience,
        )

        def objective(trial: optuna.Trial) -> float:
            params = self._space.suggest(trial)
            value = self._objective_fn(params)
            if pruner.should_prune(value, trial.number):
                study.stop()
            return value

        study.optimize(
            objective,
            n_trials=self._config.n_trials,
            timeout=self._config.timeout_sec,
        )

        all_trials = tuple(
            {
                "params": t.params,
                "value": t.value if t.value is not None else float("nan"),
                "state": str(t.state),
            }
            for t in study.trials
        )

        best = study.best_trial
        return HyperOptResult(
            best_params=best.params,
            best_value=best.value,
            n_trials=len(study.trials),
            all_trials=all_trials,
        )

    def _create_sampler(self) -> Any:
        import optuna

        seed = self._config.seed
        if self._config.sampler == "tpe":
            return optuna.samplers.TPESampler(seed=seed)
        elif self._config.sampler == "random":
            return optuna.samplers.RandomSampler(seed=seed)
        elif self._config.sampler == "cmaes":
            return optuna.samplers.CmaEsSampler(seed=seed)
        else:
            logger.warning("Unknown sampler %r, defaulting to TPE", self._config.sampler)
            return optuna.samplers.TPESampler(seed=seed)

    def _random_search(self) -> HyperOptResult:
        rng = random.Random(self._config.seed)
        maximize = self._config.direction == "maximize"

        best_params: dict[str, Any] = {}
        best_value = float("-inf") if maximize else float("inf")
        all_trials: list[dict[str, Any]] = []

        pruner = TimeSeriesPruner(
            min_trials=self._config.pruner_min_trials,
            patience=self._config.pruner_patience,
        )

        for i in range(self._config.n_trials):
            params = self._space.sample_random(rng)

            try:
                value = self._objective_fn(params)
            except Exception:
                logger.warning("Trial %d failed", i, exc_info=True)
                all_trials.append({"params": params, "value": float("nan"), "state": "FAIL"})
                continue

            all_trials.append({"params": params, "value": value, "state": "COMPLETE"})

            is_better = (value > best_value) if maximize else (value < best_value)
            if is_better:
                best_value = value
                best_params = params

            effective_value = value if maximize else -value
            if pruner.should_prune(effective_value, i):
                logger.info("Early stopping at trial %d", i)
                break

        if not best_params and all_trials:
            best_params = all_trials[0]["params"]
            best_value = all_trials[0]["value"]

        return HyperOptResult(
            best_params=best_params,
            best_value=best_value,
            n_trials=len(all_trials),
            all_trials=tuple(all_trials),
        )
