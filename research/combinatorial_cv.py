# research/combinatorial_cv.py
"""Combinatorial Purged Cross-Validation — prevents look-ahead bias in time series."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CVFold:
    """Single CV fold result."""
    fold_idx: int
    train_indices: Tuple[int, ...]
    test_indices: Tuple[int, ...]
    metric: float


@dataclass(frozen=True, slots=True)
class CombinatorialCVResult:
    """Result of Combinatorial Purged CV."""
    folds: Tuple[CVFold, ...]
    avg_metric: float
    std_metric: float
    n_combinations: int

    @property
    def metrics(self) -> List[float]:
        return [f.metric for f in self.folds]


def _generate_purged_splits(
    n_groups: int,
    n_test_groups: int,
    purge_groups: int = 1,
) -> List[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
    """Generate train/test splits with purging between adjacent groups.

    Each combination of n_test_groups groups forms a test set.
    Groups adjacent to test groups are purged from training.
    """
    all_groups = list(range(n_groups))
    splits: List[Tuple[Tuple[int, ...], Tuple[int, ...]]] = []

    for test_combo in combinations(all_groups, n_test_groups):
        test_set = set(test_combo)
        purge_set: set[int] = set()
        for tg in test_set:
            for offset in range(1, purge_groups + 1):
                purge_set.add(tg - offset)
                purge_set.add(tg + offset)

        train_set = tuple(
            g for g in all_groups
            if g not in test_set and g not in purge_set
        )
        if not train_set:
            continue

        splits.append((train_set, tuple(sorted(test_set))))

    return splits


def combinatorial_purged_cv(
    data_length: int,
    evaluate_fn: Callable[[int, int], float],
    *,
    n_groups: int = 6,
    n_test_groups: int = 2,
    purge_groups: int = 1,
) -> CombinatorialCVResult:
    """Run Combinatorial Purged Cross-Validation.

    Args:
        data_length: Total number of data points.
        evaluate_fn: Callable(start_idx, end_idx) -> metric.
        n_groups: Number of time groups to split data into.
        n_test_groups: Number of groups per test set.
        purge_groups: Number of groups to purge around test set.

    Returns:
        CombinatorialCVResult with all fold metrics.
    """
    group_size = data_length // n_groups
    splits = _generate_purged_splits(n_groups, n_test_groups, purge_groups)

    folds: List[CVFold] = []
    for idx, (train_groups, test_groups) in enumerate(splits):
        # Convert group indices to data indices
        test_start = min(test_groups) * group_size
        test_end = (max(test_groups) + 1) * group_size
        if max(test_groups) == n_groups - 1:
            test_end = data_length

        try:
            metric = evaluate_fn(test_start, test_end)
        except Exception as e:
            logger.warning("CPCV fold %d failed: %s", idx, e)
            continue

        folds.append(CVFold(
            fold_idx=idx,
            train_indices=train_groups,
            test_indices=test_groups,
            metric=metric,
        ))

    metrics = [f.metric for f in folds]
    avg = sum(metrics) / max(len(metrics), 1)
    if len(metrics) > 1:
        var = sum((m - avg) ** 2 for m in metrics) / (len(metrics) - 1)
        std = var ** 0.5
    else:
        std = 0.0

    return CombinatorialCVResult(
        folds=tuple(folds),
        avg_metric=avg,
        std_metric=std,
        n_combinations=len(splits),
    )
