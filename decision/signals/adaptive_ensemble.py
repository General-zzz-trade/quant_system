"""Adaptive ensemble signal — auto-tunes weights based on recent performance.

Supports three methods:
- ic_weighted: weight by rolling IC (signal-return correlation)
- inverse_vol: weight by inverse signal variance (stability)
- ridge: refit stacking ensemble on recent data
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, List, Mapping, Optional, Sequence

from decision.types import SignalResult
from decision.signals.base import SignalModel


@dataclass(frozen=True, slots=True)
class AdaptiveEnsembleConfig:
    """Configuration for adaptive weight calibration."""

    lookback_bars: int = 200
    recalibrate_every: int = 50
    method: str = "ic_weighted"  # "ic_weighted" | "inverse_vol" | "ridge"
    shrinkage: float = 0.3  # toward equal weight


class AdaptiveEnsembleSignal:
    """Ensemble signal that periodically recalibrates weights.

    Satisfies SignalModel protocol.
    """

    def __init__(
        self,
        signals: Sequence[SignalModel],
        config: AdaptiveEnsembleConfig = AdaptiveEnsembleConfig(),
    ) -> None:
        self.name = "adaptive_ensemble"
        self._signals = list(signals)
        self._config = config

        # Weight state
        k = len(signals)
        equal_w = 1.0 / max(k, 1)
        self._weights: Dict[str, float] = {s.name: equal_w for s in signals}

        # History buffers for recalibration
        self._score_history: Dict[str, List[float]] = {s.name: [] for s in signals}
        self._return_history: List[float] = []
        self._bars_since_calibration: int = 0

    def compute(self, snapshot: Any, symbol: str) -> SignalResult:
        """Compute weighted ensemble signal."""
        scores: Dict[str, SignalResult] = {}
        weighted_score = Decimal("0")
        weighted_conf = Decimal("0")

        for sig in self._signals:
            result = sig.compute(snapshot, symbol)
            scores[sig.name] = result
            w = Decimal(str(self._weights.get(sig.name, 0.0)))
            weighted_score += result.score * w
            weighted_conf += result.confidence * abs(w)

        side = "flat"
        if weighted_score > 0:
            side = "buy"
        elif weighted_score < 0:
            side = "sell"

        meta = {
            "weights": dict(self._weights),
            "components": {
                name: {"side": r.side, "score": str(r.score)}
                for name, r in scores.items()
            },
        }

        return SignalResult(
            symbol=symbol, side=side,
            score=weighted_score, confidence=weighted_conf,
            meta=meta,
        )

    def record(self, signal_scores: Mapping[str, float], realized_return: float) -> None:
        """Record signal scores and realized return for calibration.

        Call after each bar with the raw scores from each sub-signal
        and the forward return that materialized.
        """
        for name, score in signal_scores.items():
            if name in self._score_history:
                buf = self._score_history[name]
                buf.append(score)
                if len(buf) > self._config.lookback_bars:
                    buf[:] = buf[-self._config.lookback_bars:]

        self._return_history.append(realized_return)
        if len(self._return_history) > self._config.lookback_bars:
            self._return_history[:] = self._return_history[-self._config.lookback_bars:]

        self._bars_since_calibration += 1
        if self._bars_since_calibration >= self._config.recalibrate_every:
            self.recalibrate()

    def recalibrate(self) -> None:
        """Recalibrate weights based on recent signal performance."""
        self._bars_since_calibration = 0

        if len(self._return_history) < 20:
            return

        method = self._config.method
        if method == "ic_weighted":
            raw_weights = self._ic_weights()
        elif method == "inverse_vol":
            raw_weights = self._inverse_vol_weights()
        elif method == "ridge":
            raw_weights = self._ridge_weights()
        else:
            return

        # Apply shrinkage toward equal weight
        k = len(self._signals)
        equal_w = 1.0 / max(k, 1)
        alpha = self._config.shrinkage

        for name in self._weights:
            raw = raw_weights.get(name, equal_w)
            self._weights[name] = alpha * equal_w + (1 - alpha) * raw

    def _ic_weights(self) -> Dict[str, float]:
        """Weight by rolling IC (Pearson correlation with returns)."""
        rets = self._return_history
        n = len(rets)
        ics: Dict[str, float] = {}

        for name, scores in self._score_history.items():
            m = min(len(scores), n)
            if m < 20:
                ics[name] = 0.0
                continue
            xs = scores[-m:]
            ys = rets[-m:]
            ics[name] = max(_pearson(xs, ys), 0.0)  # clip negative IC to 0

        total = sum(ics.values())
        if total < 1e-12:
            k = len(ics)
            return {name: 1.0 / max(k, 1) for name in ics}

        return {name: ic / total for name, ic in ics.items()}

    def _inverse_vol_weights(self) -> Dict[str, float]:
        """Weight by inverse signal variance (lower variance → higher weight)."""
        inv_vols: Dict[str, float] = {}

        for name, scores in self._score_history.items():
            if len(scores) < 20:
                inv_vols[name] = 1.0
                continue
            recent = scores[-self._config.lookback_bars:]
            mean = sum(recent) / len(recent)
            var = sum((s - mean) ** 2 for s in recent) / len(recent)
            inv_vols[name] = 1.0 / math.sqrt(var) if var > 1e-12 else 1.0

        total = sum(inv_vols.values())
        if total < 1e-12:
            k = len(inv_vols)
            return {name: 1.0 / max(k, 1) for name in inv_vols}

        return {name: iv / total for name, iv in inv_vols.items()}

    def _ridge_weights(self) -> Dict[str, float]:
        """Refit weights via ridge regression on recent data."""
        rets = self._return_history
        n = len(rets)
        names = [s.name for s in self._signals]
        k = len(names)

        signals_data: Dict[str, List[float]] = {}
        for name in names:
            scores = self._score_history.get(name, [])
            m = min(len(scores), n)
            signals_data[name] = scores[-m:] if m > 0 else []

        # Use minimum overlapping length
        min_len = min(len(signals_data[name]) for name in names) if names else 0
        min_len = min(min_len, n)

        if min_len < 20 or k == 0:
            return {name: 1.0 / max(k, 1) for name in names}

        target = rets[-min_len:]
        X = {name: signals_data[name][-min_len:] for name in names}

        # Build X'X + lambda*I and X'y
        reg = 0.01
        y_mean = sum(target) / min_len
        x_means = {name: sum(X[name]) / min_len for name in names}

        XtX = [[0.0] * k for _ in range(k)]
        Xty = [0.0] * k

        for i in range(min_len):
            for j in range(k):
                xj = X[names[j]][i] - x_means[names[j]]
                Xty[j] += xj * (target[i] - y_mean)
                for m in range(k):
                    XtX[j][m] += xj * (X[names[m]][i] - x_means[names[m]])

        for j in range(k):
            XtX[j][j] += reg * min_len

        beta = _solve_linear(XtX, Xty, k)

        # Normalize to sum to 1 (positive weights only)
        raw = {names[j]: max(beta[j], 0.0) for j in range(k)}
        total = sum(raw.values())
        if total < 1e-12:
            return {name: 1.0 / max(k, 1) for name in names}
        return {name: w / total for name, w in raw.items()}

    @property
    def weights(self) -> Dict[str, float]:
        return dict(self._weights)


def _pearson(x: List[float], y: List[float]) -> float:
    n = len(x)
    if n < 2 or len(y) != n:
        return 0.0
    mx = sum(x) / n
    my = sum(y) / n
    cov = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    vx = sum((xi - mx) ** 2 for xi in x)
    vy = sum((yi - my) ** 2 for yi in y)
    denom = math.sqrt(vx * vy)
    return cov / denom if denom > 1e-12 else 0.0


def _solve_linear(A: list[list[float]], b: list[float], k: int) -> list[float]:
    """Gaussian elimination with partial pivoting."""
    aug = [A[i][:] + [b[i]] for i in range(k)]

    for col in range(k):
        max_row = col
        for row in range(col + 1, k):
            if abs(aug[row][col]) > abs(aug[max_row][col]):
                max_row = row
        aug[col], aug[max_row] = aug[max_row], aug[col]

        pivot = aug[col][col]
        if abs(pivot) < 1e-12:
            continue

        for row in range(col + 1, k):
            factor = aug[row][col] / pivot
            for j in range(col, k + 1):
                aug[row][j] -= factor * aug[col][j]

    x = [0.0] * k
    for i in range(k - 1, -1, -1):
        if abs(aug[i][i]) < 1e-12:
            continue
        x[i] = aug[i][k]
        for j in range(i + 1, k):
            x[i] -= aug[i][j] * x[j]
        x[i] /= aug[i][i]

    return x
