"""Meta-learner implementations for signal combination."""
from __future__ import annotations

import logging
from typing import Protocol, Sequence

logger = logging.getLogger(__name__)


class MetaLearner(Protocol):
    """Protocol for meta-learners that combine base signals."""

    def fit(self, X: Sequence[Sequence[float]], y: Sequence[float]) -> None: ...
    def predict(self, X: Sequence[float]) -> float: ...


class LinearMetaLearner:
    """Simple linear combination meta-learner with optional L2 regularization.

    Implements ridge regression: w = (X'X + lambda*I)^{-1} X'y.
    Pure Python, no external dependencies.
    """

    def __init__(self, *, regularization: float = 0.01) -> None:
        self._reg = regularization
        self._weights: list[float] = []
        self._bias: float = 0.0
        self._fitted: bool = False

    def fit(self, X: Sequence[Sequence[float]], y: Sequence[float]) -> None:
        """Fit linear model on feature matrix X and target y."""
        n = len(y)
        if n == 0:
            return
        k = len(X[0]) if n > 0 else 0
        if k == 0:
            return

        y_mean = sum(y) / n
        x_means = [
            sum(X[i][j] for i in range(n)) / n
            for j in range(k)
        ]

        XtX = [[0.0] * k for _ in range(k)]
        Xty = [0.0] * k

        for i in range(n):
            for j in range(k):
                xj = X[i][j] - x_means[j]
                Xty[j] += xj * (y[i] - y_mean)
                for m in range(k):
                    XtX[j][m] += xj * (X[i][m] - x_means[m])

        for j in range(k):
            XtX[j][j] += self._reg * n

        self._weights = _solve(XtX, Xty, k)
        self._bias = y_mean - sum(
            self._weights[j] * x_means[j] for j in range(k)
        )
        self._fitted = True

    def predict(self, X: Sequence[float]) -> float:
        """Predict target from feature vector."""
        if not self._fitted:
            raise RuntimeError("Must call fit() before predict()")
        return self._bias + sum(w * x for w, x in zip(self._weights, X))

    @property
    def weights(self) -> list[float]:
        return list(self._weights)

    @property
    def bias(self) -> float:
        return self._bias


def _solve(A: list[list[float]], b: list[float], k: int) -> list[float]:
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
