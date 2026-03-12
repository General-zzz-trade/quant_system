"""IC Monitor — tracks rolling information coefficient for a single horizon.

Uses EMA-based Spearman IC approximation for online computation.
"""
from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np


class ICMonitor:
    """Rolling IC tracker using rank correlation.

    Parameters
    ----------
    window : int
        Number of (pred, actual) pairs to keep for rolling IC.
    decay_threshold : float
        IC below this for `decay_lookback` consecutive checks → decaying.
    decay_lookback : int
        Number of IC evaluations to check for decay trend.
    """

    def __init__(
        self,
        window: int = 720,
        decay_threshold: float = 0.0,
        decay_lookback: int = 5,
    ):
        self._window = window
        self._decay_threshold = decay_threshold
        self._decay_lookback = decay_lookback
        self._preds: deque = deque(maxlen=window)
        self._actuals: deque = deque(maxlen=window)
        self._ic_history: deque = deque(maxlen=decay_lookback)

    def update(self, pred: float, actual: float) -> None:
        """Add a (prediction, realized_return) pair."""
        self._preds.append(pred)
        self._actuals.append(actual)

    @property
    def rolling_ic(self) -> float:
        """Compute rolling Spearman IC over the window."""
        if len(self._preds) < 50:
            return 0.0
        from scipy.stats import spearmanr
        r, _ = spearmanr(list(self._preds), list(self._actuals))
        ic = float(r) if not np.isnan(r) else 0.0
        self._ic_history.append(ic)
        return ic

    @property
    def n_samples(self) -> int:
        return len(self._preds)

    @property
    def decaying(self) -> bool:
        """True if IC has been below threshold for recent evaluations."""
        if len(self._ic_history) < self._decay_lookback:
            return False
        return all(ic < self._decay_threshold for ic in self._ic_history)
