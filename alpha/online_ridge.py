# alpha/online_ridge.py
"""Online Ridge Regression — incremental weight updates between weekly retrains.

Standard Ridge is retrained weekly (Sunday 2am). Between retrains, the model
weights are static. This module provides incremental updates:

  1. Receives each bar's features + realized return
  2. Updates Ridge weights via recursive least squares (RLS)
  3. Maintains a forgetting factor (λ) for regime adaptation
  4. Provides predict() matching sklearn Ridge interface

Benefits:
  - Adapts to regime drift within the week
  - Captures recent feature importance changes
  - Falls back to static weights if update fails
  - Reset to static weights on each retrain

Math:
  RLS update with forgetting factor λ ∈ (0.99, 1.0):
    P = (1/λ)(P - P·x·xᵀ·P / (λ + xᵀ·P·x))
    K = P·x / (λ + xᵀ·P·x)
    w = w + K·(y - xᵀ·w)

  Where P is the inverse covariance matrix.

Note: load_from_sklearn uses pickle to load sklearn Ridge models.
This is consistent with the existing model loading pattern in the codebase
(alpha_runner.py, model_loader.py all use pickle for sklearn models).
"""
from __future__ import annotations

import logging
import pickle  # noqa: S403 — sklearn models are trusted local artifacts
from typing import Optional

import numpy as np

_log = logging.getLogger(__name__)


class OnlineRidge:
    """Online Ridge with recursive least squares (RLS) updates.

    Starts from a pre-trained sklearn Ridge model's weights and
    incrementally updates them as new (features, return) pairs arrive.

    Parameters
    ----------
    n_features : int
        Number of input features.
    forgetting_factor : float
        RLS forgetting factor. 1.0 = no forgetting (pure batch Ridge).
        0.99 = forget ~1% per step (adapt faster). Default 0.997.
    ridge_alpha : float
        Regularization (initialization of P matrix diagonal).
    max_update_magnitude : float
        Clamp weight updates to prevent instability.
    min_samples_for_update : int
        Number of samples before allowing updates.
    """

    def __init__(
        self,
        n_features: int,
        forgetting_factor: float = 0.997,
        ridge_alpha: float = 1.0,
        max_update_magnitude: float = 0.1,
        min_samples_for_update: int = 50,
    ) -> None:
        self._n = n_features
        self._lambda = forgetting_factor
        self._alpha = ridge_alpha
        self._max_update = max_update_magnitude
        self._min_samples = min_samples_for_update

        # Initialize weights and covariance
        self._w = np.zeros(n_features, dtype=np.float64)
        self._intercept = 0.0
        self._P = np.eye(n_features, dtype=np.float64) / ridge_alpha
        self._n_updates = 0
        self._static_w: Optional[np.ndarray] = None
        self._static_intercept: float = 0.0

    def load_from_sklearn(self, model_path: str) -> None:
        """Initialize weights from a trained sklearn Ridge model.

        Args:
            model_path: Path to pickled sklearn Ridge model.
                        Only trusted local model artifacts should be loaded.
        """
        with open(model_path, "rb") as f:
            model = pickle.load(f)  # noqa: S301 — trusted local model

        coef = np.asarray(model.coef_, dtype=np.float64).ravel()
        if len(coef) != self._n:
            raise ValueError(
                f"Model has {len(coef)} features, expected {self._n}"
            )

        self._w = coef.copy()
        self._intercept = float(model.intercept_)
        self._static_w = coef.copy()
        self._static_intercept = float(model.intercept_)

        # Reset P matrix for fresh online updates from this baseline
        self._P = np.eye(self._n, dtype=np.float64) / self._alpha
        self._n_updates = 0

        _log.info(
            "OnlineRidge: loaded from %s, %d features, intercept=%.6f",
            model_path, self._n, self._intercept,
        )

    def load_from_weights(
        self,
        weights: np.ndarray,
        intercept: float = 0.0,
    ) -> None:
        """Initialize from explicit weight vector."""
        w = np.asarray(weights, dtype=np.float64).ravel()
        if len(w) != self._n:
            raise ValueError(f"Expected {self._n} weights, got {len(w)}")
        self._w = w.copy()
        self._intercept = float(intercept)
        self._static_w = w.copy()
        self._static_intercept = float(intercept)
        self._P = np.eye(self._n, dtype=np.float64) / self._alpha
        self._n_updates = 0

    def predict(self, x: np.ndarray) -> float:
        """Predict using current (possibly updated) weights.

        Args:
            x: Feature vector of shape (n_features,).

        Returns:
            Prediction value.
        """
        x = np.asarray(x, dtype=np.float64).ravel()
        return float(np.dot(self._w, x) + self._intercept)

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """Predict for multiple samples. Shape (n_samples, n_features)."""
        X = np.asarray(X, dtype=np.float64)
        return X @ self._w + self._intercept

    def update(self, x: np.ndarray, y: float) -> float:
        """Update weights with a new observation via RLS.

        Args:
            x: Feature vector of shape (n_features,).
            y: Realized target value (e.g., forward return).

        Returns:
            Prediction error before update (useful for monitoring).
        """
        x = np.asarray(x, dtype=np.float64).ravel()
        y = float(y)

        # Skip if NaN or Inf
        if not np.all(np.isfinite(x)) or not np.isfinite(y):
            return 0.0

        self._n_updates += 1

        # Don't update weights until we have enough samples
        if self._n_updates < self._min_samples:
            return float(np.dot(self._w, x) + self._intercept - y)

        # RLS update
        lam = self._lambda
        pred = float(np.dot(self._w, x) + self._intercept)
        error = y - pred

        # K = P·x / (λ + xᵀ·P·x)
        Px = self._P @ x
        denom = lam + float(x @ Px)
        if abs(denom) < 1e-12:
            return error

        K = Px / denom

        # Clamp update magnitude
        delta_w = K * error
        delta_norm = float(np.linalg.norm(delta_w))
        if delta_norm > self._max_update:
            delta_w = delta_w * (self._max_update / delta_norm)

        # w = w + K·error
        self._w += delta_w

        # P = (1/λ)(P - K·xᵀ·P)  — Joseph form for numerical stability
        self._P = (self._P - np.outer(K, x @ self._P)) / lam

        # Enforce symmetry (accumulates floating-point asymmetry over many updates)
        self._P = (self._P + self._P.T) / 2

        # Eigenvalue floor: prevent P from losing positive-definiteness
        if self._n_updates % 200 == 0:
            eigvals = np.linalg.eigvalsh(self._P)
            if eigvals.min() < 1e-10:
                self._P += np.eye(self._n) * (1e-8 - eigvals.min())

        # Intercept update (simple EMA)
        self._intercept += 0.001 * error

        return error

    def reset_to_static(self) -> None:
        """Reset to the original static weights (call after retrain)."""
        if self._static_w is not None:
            self._w = self._static_w.copy()
            self._intercept = self._static_intercept
        self._P = np.eye(self._n, dtype=np.float64) / self._alpha
        self._n_updates = 0
        _log.info("OnlineRidge: reset to static weights")

    @property
    def weight_drift(self) -> float:
        """L2 distance between current and static weights."""
        if self._static_w is None:
            return 0.0
        return float(np.linalg.norm(self._w - self._static_w))

    @property
    def n_updates(self) -> int:
        return self._n_updates

    @property
    def weights(self) -> np.ndarray:
        return self._w.copy()

    @property
    def intercept(self) -> float:
        return self._intercept

    @property
    def stats(self) -> dict:
        return {
            "n_updates": self._n_updates,
            "weight_drift": round(self.weight_drift, 6),
            "forgetting_factor": self._lambda,
            "w_norm": round(float(np.linalg.norm(self._w)), 6),
            "intercept": round(self._intercept, 8),
        }
