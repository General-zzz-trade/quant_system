"""Regime-conditional training — separate models per volatility regime.

Splits data by vol_regime (0=low, 1=mid, 2=high) and trains independent
models. At inference, selects the model matching the current regime.
"""
from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from alpha.models.lgbm_alpha import LGBMAlphaModel

logger = logging.getLogger(__name__)

REGIMES = (0, 1, 2)


@dataclass
class RegimeModelBundle:
    """Bundle of per-regime models + a fallback all-data model."""

    models: Dict[int, LGBMAlphaModel] = field(default_factory=dict)
    fallback: Optional[LGBMAlphaModel] = None
    feature_names: Tuple[str, ...] = ()
    regime_col: str = "regime_vol"
    min_samples_per_regime: int = 500

    def predict_regime(
        self,
        regime: int,
        features: Dict[str, Any],
    ) -> Optional[float]:
        """Get raw prediction for the given regime."""
        model = self.models.get(regime, self.fallback)
        if model is None or model._model is None:
            return None
        x = [[features.get(f, float("nan")) for f in self.feature_names]]
        return float(model._model.predict(x)[0])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        bundle = {
            "feature_names": self.feature_names,
            "regime_col": self.regime_col,
            "models": {},
            "fallback": None,
        }
        for regime, model in self.models.items():
            bundle["models"][regime] = {
                "model": model._model,
                "features": model.feature_names,
                "is_classifier": model._is_classifier,
            }
        if self.fallback and self.fallback._model:
            bundle["fallback"] = {
                "model": self.fallback._model,
                "features": self.fallback.feature_names,
                "is_classifier": self.fallback._is_classifier,
            }
        with open(path, "wb") as f:
            pickle.dump(bundle, f)
        from infra.model_signing import sign_file
        sign_file(path)

    @classmethod
    def load(cls, path: Path) -> "RegimeModelBundle":
        from infra.model_signing import load_verified_pickle
        bundle = load_verified_pickle(path)
        result = cls(
            feature_names=tuple(bundle["feature_names"]),
            regime_col=bundle.get("regime_col", "regime_vol"),
        )
        for regime, data in bundle["models"].items():
            m = LGBMAlphaModel(
                name=f"regime_{regime}",
                feature_names=data["features"],
            )
            m._model = data["model"]
            m._is_classifier = data.get("is_classifier", False)
            result.models[int(regime)] = m
        if bundle.get("fallback"):
            fb = LGBMAlphaModel(
                name="fallback",
                feature_names=bundle["fallback"]["features"],
            )
            fb._model = bundle["fallback"]["model"]
            fb._is_classifier = bundle["fallback"].get("is_classifier", False)
            result.fallback = fb
        return result


def compute_vol_regime(vol_series: np.ndarray) -> np.ndarray:
    """Assign vol regime (0/1/2) via tercile binning."""
    valid = vol_series[~np.isnan(vol_series)]
    if len(valid) < 30:
        return np.ones(len(vol_series), dtype=int)
    p33, p67 = np.percentile(valid, [33, 67])
    regime = np.where(
        np.isnan(vol_series), 1,
        np.where(vol_series <= p33, 0,
                 np.where(vol_series <= p67, 1, 2)),
    )
    return regime.astype(int)


def train_regime_models(
    X: np.ndarray,
    y: np.ndarray,
    regimes: np.ndarray,
    feature_names: Sequence[str],
    *,
    params: Optional[Dict[str, Any]] = None,
    early_stopping_rounds: int = 50,
    embargo_bars: int = 8,
    sample_weight: Optional[np.ndarray] = None,
    min_samples: int = 500,
) -> RegimeModelBundle:
    """Train per-regime models + fallback on all data.

    Args:
        X: Feature matrix (n, p).
        y: Target array (n,).
        regimes: Regime labels (n,) with values in {0, 1, 2}.
        feature_names: Feature names.
        params: LightGBM parameters.
        early_stopping_rounds: Early stopping patience.
        embargo_bars: Embargo between train/val.
        sample_weight: Optional sample weights (n,).
        min_samples: Minimum samples to train a regime model.

    Returns:
        RegimeModelBundle with per-regime models and fallback.
    """
    bundle = RegimeModelBundle(
        feature_names=tuple(feature_names),
        min_samples_per_regime=min_samples,
    )

    # Fallback model on all data
    logger.info("Training fallback model on all %d samples", len(X))
    fallback = LGBMAlphaModel(name="fallback", feature_names=tuple(feature_names))
    fallback.fit(
        X, y,
        params=params,
        early_stopping_rounds=early_stopping_rounds,
        embargo_bars=embargo_bars,
        sample_weight=sample_weight,
    )
    bundle.fallback = fallback

    # Per-regime models
    for regime in REGIMES:
        mask = regimes == regime
        n_regime = int(mask.sum())
        if n_regime < min_samples:
            logger.info("Regime %d: %d samples < %d min, using fallback",
                        regime, n_regime, min_samples)
            continue

        X_r = X[mask]
        y_r = y[mask]
        w_r = sample_weight[mask] if sample_weight is not None else None

        logger.info("Training regime %d model on %d samples", regime, n_regime)
        model = LGBMAlphaModel(
            name=f"regime_{regime}",
            feature_names=tuple(feature_names),
        )
        model.fit(
            X_r, y_r,
            params=params,
            early_stopping_rounds=early_stopping_rounds,
            embargo_bars=embargo_bars,
            sample_weight=w_r,
        )
        bundle.models[regime] = model

    trained = list(bundle.models.keys())
    logger.info("Regime models trained: %s (fallback always available)", trained)
    return bundle


def apply_vol_regime(
    vol_series: np.ndarray,
    p33: float,
    p67: float,
) -> np.ndarray:
    """Assign vol regime using pre-computed IS percentile thresholds.

    Unlike compute_vol_regime() which computes thresholds from the data itself,
    this function uses externally-provided thresholds — critical for OOS evaluation
    to prevent lookahead bias.
    """
    regime = np.where(
        np.isnan(vol_series), 1,
        np.where(vol_series <= p33, 0,
                 np.where(vol_series <= p67, 1, 2)),
    )
    return regime.astype(int)


def detect_current_regime(
    vol_values: Sequence[float],
    p33: float,
    p67: float,
) -> int:
    """Detect current regime from recent vol values."""
    if not vol_values:
        return 1
    recent_vol = vol_values[-1] if vol_values else 0.0
    if np.isnan(recent_vol):
        return 1
    if recent_vol <= p33:
        return 0
    if recent_vol <= p67:
        return 1
    return 2
