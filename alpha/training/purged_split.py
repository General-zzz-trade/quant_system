"""Purged train/test split for time-series data.

Enforces an embargo gap between train and test sets to prevent
information leakage from overlapping target windows.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def purged_train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    *,
    test_size: float = 0.2,
    embargo_bars: int = 5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split time-series data with embargo gap between train and test.

    Args:
        X: Feature matrix (n_samples, n_features).
        y: Target array (n_samples,).
        test_size: Fraction of data for test set.
        embargo_bars: Number of bars to drop between train and test
                      to prevent information leakage from target lookahead.

    Returns:
        (X_train, X_test, y_train, y_test)
    """
    n = len(X)
    split = int(n * (1.0 - test_size))
    train_end = max(split - embargo_bars, 1)
    test_start = split

    return X[:train_end], X[test_start:], y[:train_end], y[test_start:]
