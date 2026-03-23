"""Canonical utility functions for alpha research, training, and backtesting.

De-duplicated from ~38 copies across training/research/monitoring files.
"""

import numpy as np
from scipy.stats import spearmanr


def fast_ic(x: np.ndarray, y: np.ndarray) -> float:
    """Fast Spearman IC (Information Coefficient) with NaN handling.

    Parameters
    ----------
    x, y : array-like
        Prediction and target arrays. NaN values are masked out.

    Returns
    -------
    float
        Spearman rank correlation, or 0.0 if fewer than 50 valid pairs
        or the result is NaN.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = ~(np.isnan(x) | np.isnan(y))
    if m.sum() < 50:
        return 0.0
    r, _ = spearmanr(x[m], y[m])
    return float(r) if not np.isnan(r) else 0.0


def compute_target(closes: np.ndarray, horizon: int) -> np.ndarray:
    """Forward return target with 1st/99th percentile clipping.

    Parameters
    ----------
    closes : np.ndarray
        Close price series.
    horizon : int
        Number of bars to look ahead for return calculation.

    Returns
    -------
    np.ndarray
        Forward returns clipped at [p1, p99]. Trailing ``horizon`` bars
        are NaN (no future data available).
    """
    closes = np.asarray(closes, dtype=float)
    n = len(closes)
    y = np.full(n, np.nan)
    y[:n - horizon] = closes[horizon:] / closes[:n - horizon] - 1
    v = y[~np.isnan(y)]
    if len(v) > 10:
        p1, p99 = np.percentile(v, [1, 99])
        y = np.where(np.isnan(y), np.nan, np.clip(y, p1, p99))
    return y
