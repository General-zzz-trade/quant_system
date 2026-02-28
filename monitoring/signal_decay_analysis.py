"""Signal decay analysis via Information Coefficient (IC) and half-life estimation.

Complements AlphaDecayMonitor (Sharpe-based) with rank correlation analysis
to detect signal efficacy decay across lag horizons.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from math import exp, log
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class SignalDecayAnalyzer:
    """Tracks signal scores vs forward returns at multiple lags to compute IC decay."""

    max_lags: int = 20

    _data: Dict[int, List[Tuple[float, float]]] = field(default_factory=lambda: defaultdict(list), repr=False)

    def record(self, signal_score: float, forward_return: float, lag: int = 0) -> None:
        """Record a signal score and its corresponding forward return at a given lag."""
        if lag < 0 or lag > self.max_lags:
            return
        self._data[lag].append((signal_score, forward_return))

    def compute_ic_series(self) -> Dict[int, float]:
        """Compute Information Coefficient (Spearman rank correlation) at each lag."""
        result: Dict[int, float] = {}
        for lag in range(self.max_lags + 1):
            pairs = self._data.get(lag, [])
            if len(pairs) < 3:
                continue
            result[lag] = _spearman_rank_corr(pairs)
        return result

    def half_life(self) -> Optional[float]:
        """Estimate half-life of IC decay via exponential fit.

        Fits IC(lag) = IC(0) * exp(-lambda * lag) and returns ln(2)/lambda.
        Returns None if insufficient data or poor fit.
        """
        ic_series = self.compute_ic_series()
        if 0 not in ic_series or ic_series[0] <= 0:
            return None

        ic0 = ic_series[0]
        # Collect (lag, log(IC/IC0)) points for linear regression
        points: List[Tuple[float, float]] = []
        for lag in sorted(ic_series.keys()):
            if lag == 0:
                continue
            ic = ic_series[lag]
            if ic <= 0:
                continue
            ratio = ic / ic0
            if ratio <= 0:
                continue
            points.append((float(lag), log(ratio)))

        if len(points) < 2:
            return None

        # Simple linear regression: log(IC/IC0) = -lambda * lag
        n = len(points)
        sum_x = sum(p[0] for p in points)
        sum_y = sum(p[1] for p in points)
        sum_xy = sum(p[0] * p[1] for p in points)
        sum_xx = sum(p[0] ** 2 for p in points)

        denom = n * sum_xx - sum_x ** 2
        if abs(denom) < 1e-12:
            return None

        slope = (n * sum_xy - sum_x * sum_y) / denom  # should be negative
        if slope >= 0:
            return None  # IC is not decaying

        lam = -slope
        return log(2) / lam

    def is_decayed(self, threshold_ic: float = 0.02) -> bool:
        """Check if latest IC has decayed below threshold."""
        ic_series = self.compute_ic_series()
        if not ic_series:
            return False
        latest_lag = max(ic_series.keys())
        return abs(ic_series[latest_lag]) < threshold_ic

    def summary(self) -> Dict[str, Any]:
        ic_series = self.compute_ic_series()
        hl = self.half_life()
        return {
            "ic_series": ic_series,
            "half_life": hl,
            "is_decayed": self.is_decayed(),
            "n_observations": {lag: len(pairs) for lag, pairs in self._data.items()},
        }


def _spearman_rank_corr(pairs: List[Tuple[float, float]]) -> float:
    """Compute Spearman rank correlation coefficient."""
    n = len(pairs)
    if n < 2:
        return 0.0

    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]

    rx = _rank(xs)
    ry = _rank(ys)

    mean_rx = sum(rx) / n
    mean_ry = sum(ry) / n

    num = sum((rx[i] - mean_rx) * (ry[i] - mean_ry) for i in range(n))
    den_x = sum((rx[i] - mean_rx) ** 2 for i in range(n))
    den_y = sum((ry[i] - mean_ry) ** 2 for i in range(n))

    denom = (den_x * den_y) ** 0.5
    if denom < 1e-12:
        return 0.0
    return num / denom


def _rank(values: List[float]) -> List[float]:
    """Assign average ranks to values (handles ties)."""
    n = len(values)
    indexed = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n - 1 and values[indexed[j + 1]] == values[indexed[j]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # 1-based
        for k in range(i, j + 1):
            ranks[indexed[k]] = avg_rank
        i = j + 1
    return ranks
