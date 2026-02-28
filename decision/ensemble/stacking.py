"""Stacking ensemble: multiple base signals combined via a meta-learner."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Mapping, Sequence

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class StackingConfig:
    """Configuration for a stacking ensemble.

    Attributes:
        base_signal_names: Names of the base signals to combine.
        meta_method: Meta-learner type (``linear``, ``ridge``, ``mean``).
        lookback: Number of historical observations for fitting.
    """
    base_signal_names: tuple[str, ...]
    meta_method: str = "linear"
    lookback: int = 100


class StackingEnsemble:
    """Stacking ensemble: multiple base signals -> meta-learner -> combined signal.

    Fits a simple meta-learner (OLS or ridge) on historical signal values
    and target returns, then uses the learned weights for prediction.

    Usage:
        config = StackingConfig(base_signal_names=("momentum", "mean_rev"))
        ensemble = StackingEnsemble(config)
        ensemble.fit(signals={"momentum": [...], "mean_rev": [...]}, target=[...])
        combined = ensemble.predict({"momentum": 0.5, "mean_rev": -0.2})
    """

    def __init__(self, config: StackingConfig) -> None:
        self._config = config
        self._weights: dict[str, float] = {}
        self._bias: float = 0.0
        self._fitted: bool = False

    def fit(
        self,
        signals: Mapping[str, Sequence[float]],
        target: Sequence[float],
    ) -> None:
        """Fit meta-learner on historical signals and target returns."""
        if self._config.meta_method == "linear":
            self._fit_linear(signals, target, regularization=0.0)
        elif self._config.meta_method == "ridge":
            self._fit_linear(signals, target, regularization=0.01)
        elif self._config.meta_method == "mean":
            self._fit_equal_weight()
        else:
            raise ValueError(f"Unknown meta_method: {self._config.meta_method}")
        self._fitted = True

    def predict(self, signals: Mapping[str, float]) -> float:
        """Predict combined signal from base signals."""
        if not self._fitted:
            raise RuntimeError("Must call fit() before predict()")
        return self._bias + sum(
            self._weights.get(name, 0.0) * signals.get(name, 0.0)
            for name in self._config.base_signal_names
        )

    def _fit_linear(
        self,
        signals: Mapping[str, Sequence[float]],
        target: Sequence[float],
        regularization: float,
    ) -> None:
        """OLS or Ridge regression: w = (X'X + lambda*I)^-1 X'y."""
        names = list(self._config.base_signal_names)
        n = len(target)
        k = len(names)

        if n == 0 or k == 0:
            self._weights = {name: 1.0 / max(k, 1) for name in names}
            self._bias = 0.0
            return

        X = [[signals[name][i] for name in names] for i in range(n)]
        y = list(target)

        y_mean = sum(y) / n
        x_means = [sum(X[i][j] for i in range(n)) / n for j in range(k)]

        XtX = [[0.0] * k for _ in range(k)]
        Xty = [0.0] * k

        for i in range(n):
            for j in range(k):
                xj = X[i][j] - x_means[j]
                Xty[j] += xj * (y[i] - y_mean)
                for m in range(k):
                    XtX[j][m] += xj * (X[i][m] - x_means[m])

        for j in range(k):
            XtX[j][j] += regularization * n

        weights = self._solve_linear(XtX, Xty, k)

        self._bias = y_mean - sum(weights[j] * x_means[j] for j in range(k))
        self._weights = {names[j]: weights[j] for j in range(k)}

    def _fit_equal_weight(self) -> None:
        """Equal-weight combination (no fitting required)."""
        k = len(self._config.base_signal_names)
        w = 1.0 / max(k, 1)
        self._weights = {name: w for name in self._config.base_signal_names}
        self._bias = 0.0

    @staticmethod
    def _solve_linear(A: list[list[float]], b: list[float], k: int) -> list[float]:
        """Solve Ax = b via Gaussian elimination with partial pivoting."""
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

    def refit(
        self,
        signals: Mapping[str, Sequence[float]],
        target: Sequence[float],
    ) -> None:
        """Refit meta-learner on new data (convenience alias for fit)."""
        self.fit(signals, target)

    @property
    def weights(self) -> dict[str, float]:
        return dict(self._weights)

    @property
    def is_fitted(self) -> bool:
        return self._fitted
