"""Purged walk-forward splits for overlapping labels.

Sample ``i`` is decided at ``decision_times[i]`` and its label covers
``(decision_times[i], label_ends[i]]``. For a test block ``[a, b)``:

- test  = samples decided in ``[a, b)`` whose label ends by ``limit``;
- train = samples whose label ended before ``a - embargo`` (purged: no
  training label reaches into the test block or beyond).

Blocks start at ``first_test`` and step by ``block`` up to ``limit``
(expanding training window). A label never ends after ``limit``, so data
from ``limit`` on is never used, not even as a test label.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Fold:
    test_start: int
    test_end: int
    train: np.ndarray      # sample indices
    test: np.ndarray


def walk_forward(decision_times, label_ends, *, first_test: int, block: int, limit: int, embargo: int = 0,
                 min_train: int = 1) -> list[Fold]:
    t = np.asarray(decision_times, dtype=np.int64)
    end = np.asarray(label_ends, dtype=np.int64)
    if t.shape != end.shape or (len(t) > 1 and np.any(np.diff(t) <= 0)):
        raise ValueError("decision times must be strictly increasing and match label ends")
    if np.any(end <= t):
        raise ValueError("every label must end after its decision time")
    if block <= 0 or embargo < 0 or not first_test < limit:
        raise ValueError("invalid block, embargo or window")
    folds, a = [], first_test
    while a < limit:
        b = min(a + block, limit)
        test = np.flatnonzero((t >= a) & (t < b) & (end <= limit))
        train = np.flatnonzero(end < a - embargo)
        if len(test) and len(train) >= min_train:
            folds.append(Fold(a, b, train, test))
        a = b
    return folds


__all__ = ["Fold", "walk_forward"]
