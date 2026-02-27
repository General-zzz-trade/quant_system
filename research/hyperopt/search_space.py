"""Hyperparameter search space definitions for strategy optimization."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ParamRange:
    """Continuous or discrete numeric parameter range.

    Args:
        name: Parameter identifier.
        low: Lower bound (inclusive).
        high: Upper bound (inclusive).
        step: Discretization step. None means continuous.
        log_scale: Sample in log-space (useful for learning rates).
    """
    name: str
    low: float
    high: float
    step: Optional[float] = None
    log_scale: bool = False


@dataclass(frozen=True, slots=True)
class CategoricalParam:
    """Categorical (enum-like) parameter."""
    name: str
    choices: tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class SearchSpace:
    """Define hyperparameter search space for a strategy.

    Supports float, int, and categorical parameters. Compatible with Optuna
    trial objects for suggest_* calls, and with pure-Python random sampling
    as a fallback.
    """
    name: str
    float_params: tuple[ParamRange, ...] = ()
    int_params: tuple[ParamRange, ...] = ()
    categorical_params: tuple[CategoricalParam, ...] = ()

    def suggest(self, trial: Any) -> dict[str, Any]:
        """Suggest parameters from an Optuna trial object."""
        params: dict[str, Any] = {}
        for p in self.float_params:
            kwargs: dict[str, Any] = {"log": p.log_scale}
            if p.step is not None:
                kwargs["step"] = p.step
            params[p.name] = trial.suggest_float(p.name, p.low, p.high, **kwargs)
        for p in self.int_params:
            kwargs_int: dict[str, Any] = {}
            if p.step is not None:
                kwargs_int["step"] = int(p.step)
            params[p.name] = trial.suggest_int(
                p.name, int(p.low), int(p.high), **kwargs_int,
            )
        for p in self.categorical_params:
            params[p.name] = trial.suggest_categorical(p.name, list(p.choices))
        return params

    def sample_random(self, rng: Any) -> dict[str, Any]:
        """Sample random parameters using a ``random.Random`` instance."""
        params: dict[str, Any] = {}
        for p in self.float_params:
            if p.log_scale and p.low > 0:
                import math
                log_low = math.log(p.low)
                log_high = math.log(p.high)
                val = math.exp(rng.uniform(log_low, log_high))
            else:
                val = rng.uniform(p.low, p.high)
            if p.step is not None:
                val = round(val / p.step) * p.step
            params[p.name] = val
        for p in self.int_params:
            step = int(p.step) if p.step is not None else 1
            val = rng.randrange(int(p.low), int(p.high) + 1, step)
            params[p.name] = val
        for p in self.categorical_params:
            params[p.name] = rng.choice(list(p.choices))
        return params

    @property
    def dimension_count(self) -> int:
        return len(self.float_params) + len(self.int_params) + len(self.categorical_params)
