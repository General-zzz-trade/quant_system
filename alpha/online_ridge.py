# alpha/online_ridge.py
"""Online Ridge Regression — incremental weight updates between weekly retrains.

Standard Ridge is retrained weekly (Sunday 2am). Between retrains, the model
weights are static. This module provides incremental updates:

  1. Receives each bar's features + realized return
  2. Updates Ridge weights via recursive least squares (RLS)
  3. Maintains a forgetting factor (lam) for regime adaptation
  4. Provides predict() matching sklearn Ridge interface

Delegates core math to RustOnlineRidge (pure Rust RLS).

Note: load_from_sklearn uses pickle to load sklearn Ridge models.
This is consistent with the existing model loading pattern in the codebase
(alpha_runner.py, model_loader.py all use pickle for sklearn models).
"""
from __future__ import annotations

import logging
import pickle  # noqa: S403 — sklearn models are trusted local artifacts

import numpy as np

from _quant_hotpath import RustOnlineRidge  # type: ignore[import-untyped]

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
        self._rust = RustOnlineRidge(
            n_features,
            forgetting_factor,
            ridge_alpha,
            max_update_magnitude,
            min_samples_for_update,
        )

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

        self._rust.load_from_weights(coef.tolist(), float(model.intercept_))

        _log.info(
            "OnlineRidge: loaded from %s, %d features, intercept=%.6f",
            model_path, self._n, float(model.intercept_),
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
        self._rust.load_from_weights(w.tolist(), float(intercept))

    def predict(self, x: np.ndarray) -> float:
        """Predict using current (possibly updated) weights.

        Args:
            x: Feature vector of shape (n_features,).

        Returns:
            Prediction value.
        """
        x = np.asarray(x, dtype=np.float64).ravel()
        return self._rust.predict(x.tolist())

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """Predict for multiple samples. Shape (n_samples, n_features)."""
        X = np.asarray(X, dtype=np.float64)
        rows = [row.tolist() for row in X]
        return np.array(self._rust.predict_batch(rows), dtype=np.float64)

    def update(self, x: np.ndarray, y: float) -> float:
        """Update weights with a new observation via RLS.

        Args:
            x: Feature vector of shape (n_features,).
            y: Realized target value (e.g., forward return).

        Returns:
            Prediction error before update (useful for monitoring).
        """
        x = np.asarray(x, dtype=np.float64).ravel()
        return self._rust.update(x.tolist(), float(y))

    def reset_to_static(self) -> None:
        """Reset to the original static weights (call after retrain)."""
        self._rust.reset_to_static()
        _log.info("OnlineRidge: reset to static weights")

    @property
    def weight_drift(self) -> float:
        """L2 distance between current and static weights."""
        return self._rust.weight_drift

    @property
    def n_updates(self) -> int:
        return int(self._rust.n_updates)

    @property
    def weights(self) -> np.ndarray:
        return np.array(self._rust.weights_list, dtype=np.float64)

    @property
    def intercept(self) -> float:
        return self._rust.intercept

    @property
    def stats(self) -> dict:
        s = self._rust.stats()
        return {
            "n_updates": int(s["n_updates"]),
            "weight_drift": round(s["weight_drift"], 6),
            "forgetting_factor": s["forgetting_factor"],
            "w_norm": round(s["w_norm"], 6),
            "intercept": round(s["intercept"], 8),
        }
