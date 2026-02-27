"""Time-series aware pruning for hyperparameter optimization."""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class TimeSeriesPruner:
    """Prune Optuna trials using time-series aware criteria.

    Prevents wasted compute by stopping optimization early when no improvement
    is observed for ``patience`` consecutive trials. Requires at least
    ``min_trials`` before pruning can activate.
    """

    def __init__(
        self,
        *,
        min_trials: int = 5,
        patience: int = 10,
        min_improvement: float = 0.01,
    ) -> None:
        self._min_trials = min_trials
        self._patience = patience
        self._min_improvement = min_improvement
        self._best_value: Optional[float] = None
        self._no_improve_count: int = 0

    def should_prune(self, trial_value: float, trial_number: int) -> bool:
        """Check if optimization should stop early.

        Args:
            trial_value: Metric value of the latest trial (higher is better).
            trial_number: Zero-based trial index.

        Returns:
            True if optimization should be stopped.
        """
        if trial_number < self._min_trials:
            return False

        if self._best_value is None or trial_value > self._best_value + self._min_improvement:
            self._best_value = trial_value
            self._no_improve_count = 0
        else:
            self._no_improve_count += 1

        if self._no_improve_count >= self._patience:
            logger.info(
                "Pruning after %d trials without improvement (best=%.4f)",
                self._no_improve_count, self._best_value,
            )
            return True
        return False

    def reset(self) -> None:
        """Reset pruner state for a fresh optimization run."""
        self._best_value = None
        self._no_improve_count = 0

    @property
    def best_value(self) -> Optional[float]:
        return self._best_value

    @property
    def trials_without_improvement(self) -> int:
        return self._no_improve_count
