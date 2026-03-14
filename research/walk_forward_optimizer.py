"""Walk-forward optimization framework.

Systematically tests strategy parameters across expanding time windows
to identify robust parameter sets.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Sequence, Tuple

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class WalkForwardFold:
    """Result of a single walk-forward fold."""
    fold_idx: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    best_params: Dict[str, Any]
    train_metric: float
    test_metric: float


@dataclass(frozen=True, slots=True)
class WalkForwardResult:
    """Aggregate walk-forward optimization result."""
    folds: Tuple[WalkForwardFold, ...]
    avg_train_metric: float
    avg_test_metric: float
    best_params_frequency: Dict[str, int]

    @property
    def is_overfit(self) -> bool:
        """Simple overfit check: test significantly worse than train."""
        if self.avg_train_metric == 0:
            return False
        ratio = self.avg_test_metric / self.avg_train_metric
        return ratio < 0.5


def walk_forward_optimize(
    data_length: int,
    param_grid: Sequence[Dict[str, Any]],
    evaluate_fn: Callable[[Dict[str, Any], int, int], float],
    *,
    n_folds: int = 5,
    train_ratio: float = 0.6,
    expanding: bool = True,
    metric_higher_is_better: bool = True,
) -> WalkForwardResult:
    """Run walk-forward optimization.

    Args:
        data_length: Total number of data points.
        param_grid: List of parameter combinations to test.
        evaluate_fn: Callable(params, start_idx, end_idx) -> metric value.
        n_folds: Number of train/test folds.
        train_ratio: Fraction of each fold used for training.
        expanding: If True, training window expands from start.
        metric_higher_is_better: If True, maximize metric.

    Returns:
        WalkForwardResult with per-fold and aggregate results.
    """
    fold_size = data_length // n_folds
    folds: List[WalkForwardFold] = []
    param_wins: Dict[str, int] = {}

    for fold_idx in range(n_folds):
        if expanding:
            train_start = 0
        else:
            train_start = fold_idx * fold_size

        fold_end = (fold_idx + 1) * fold_size
        if fold_idx == n_folds - 1:
            fold_end = data_length

        split = train_start + int((fold_end - train_start) * train_ratio)
        train_end = split
        test_start = split
        test_end = fold_end

        if test_end <= test_start:
            continue

        # Find best params on training set
        best_metric = float("-inf") if metric_higher_is_better else float("inf")
        best_params: Dict[str, Any] = param_grid[0] if param_grid else {}

        for params in param_grid:
            try:
                metric = evaluate_fn(params, train_start, train_end)
            except Exception as e:
                logger.warning("Fold %d param eval failed: %s", fold_idx, e)
                continue

            is_better = (metric > best_metric) if metric_higher_is_better else (metric < best_metric)
            if is_better:
                best_metric = metric
                best_params = params

        # Evaluate best params on test set
        try:
            test_metric = evaluate_fn(best_params, test_start, test_end)
        except Exception:
            test_metric = 0.0

        folds.append(WalkForwardFold(
            fold_idx=fold_idx,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            best_params=best_params,
            train_metric=best_metric,
            test_metric=test_metric,
        ))

        # Track param frequency
        param_key = str(sorted(best_params.items()))
        param_wins[param_key] = param_wins.get(param_key, 0) + 1

        logger.info(
            "Fold %d/%d: train=%.4f test=%.4f params=%s",
            fold_idx + 1, n_folds, best_metric, test_metric, best_params,
        )

    avg_train = sum(f.train_metric for f in folds) / max(len(folds), 1)
    avg_test = sum(f.test_metric for f in folds) / max(len(folds), 1)

    return WalkForwardResult(
        folds=tuple(folds),
        avg_train_metric=avg_train,
        avg_test_metric=avg_test,
        best_params_frequency=param_wins,
    )
