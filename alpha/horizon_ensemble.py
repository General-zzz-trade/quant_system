"""Adaptive Horizon Ensemble — multi-horizon prediction.

Two modes:
- "mean_zscore" (v10 compatible): simple average of per-horizon z-scores
- "ic_weighted" (v11): EMA-IC weighted, auto-downweight decaying horizons
- "ridge_primary": AlphaRunner-style Ridge(primary)+LGBM ensemble, then static
  horizon weighting via config `ic`
"""
from __future__ import annotations

import logging
from collections import deque
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

from alpha.ic_monitor import ICMonitor  # noqa: E402


class AdaptiveHorizonEnsemble:
    """Multi-horizon ensemble with optional IC-based weighting.

    Parameters
    ----------
    config : V11Config
        Configuration with ensemble_method, horizons, ic params.
    horizon_models : list[dict]
        Per-horizon model dicts with 'horizon', 'lgbm', 'xgb', 'features',
        'zscore_buf' keys (same format as MLSignalDecisionModule).
    """

    def __init__(self, config: Any, horizon_models: List[Dict[str, Any]]) -> None:
        from alpha.v11_config import V11Config
        self._config: V11Config = config
        self._horizon_models = horizon_models
        self._method = config.ensemble_method
        self._lgbm_xgb_w = config.lgbm_xgb_weight
        self._ridge_w = getattr(config, "ridge_weight", 0.6)
        self._lgbm_w = getattr(config, "lgbm_weight", 0.4)

        # IC monitors per horizon (only used in ic_weighted mode)
        self._ic_monitors: Dict[int, ICMonitor] = {}
        if self._method == "ic_weighted":
            for hm in horizon_models:
                h = hm["horizon"]
                self._ic_monitors[h] = ICMonitor(
                    window=config.ic_ema_span,
                )

        # For delayed IC updates: store predictions keyed by (horizon, bar)
        self._pending_preds: Dict[int, deque[tuple[int, float]]] = {
            hm["horizon"]: deque(maxlen=200) for hm in horizon_models
        }

    def predict(self, features: Dict[str, float]) -> Optional[float]:
        """Run ensemble prediction across all horizons.

        Returns averaged z-score (mean_zscore) or IC-weighted z-score.
        Returns None if no horizon has warmed up.
        """
        z_values: Dict[int, float] = {}
        for hm in self._horizon_models:
            h = hm["horizon"]
            pred = self._predict_single(hm, features)
            z = hm["zscore_buf"].push(pred)
            z_values[h] = z

        # Check warmup
        if not any(hm["zscore_buf"].ready for hm in self._horizon_models):
            return 0.0

        if not z_values:
            return None

        if self._method == "mean_zscore":
            return float(np.mean(list(z_values.values())))
        if self._method == "ridge_primary":
            return self._ridge_primary_predict(z_values)
        else:
            return self._ic_weighted_predict(z_values)

    def _predict_single(self, hm: Dict[str, Any], features: Dict[str, float]) -> float:
        """Run lgbm+xgb ensemble for a single horizon."""
        x = np.zeros((1, len(hm["features"])))
        for j, fname in enumerate(hm["features"]):
            x[0, j] = features.get(fname, 0.0)

        lgbm_pred = float(hm["lgbm"].predict(x)[0])

        if self._method == "ridge_primary" and hm.get("ridge") is not None:
            ridge_features = hm.get("ridge_features") or hm["features"]
            rx = np.zeros((1, len(ridge_features)))
            for j, fname in enumerate(ridge_features):
                rx[0, j] = features.get(fname, 0.0)
            ridge_pred = float(hm["ridge"].predict(rx)[0])
            return float(self._ridge_w * ridge_pred + self._lgbm_w * lgbm_pred)

        if hm["xgb"] is not None:
            try:
                import xgboost as xgb
                xgb_pred = float(hm["xgb"].predict(xgb.DMatrix(x))[0])
                return float(self._lgbm_xgb_w * lgbm_pred + (1 - self._lgbm_xgb_w) * xgb_pred)
            except Exception as e:
                logger.warning("XGBoost prediction failed in horizon ensemble, using LGBM only: %s", e)

        return lgbm_pred

    def _ic_weighted_predict(self, z_values: Dict[int, float]) -> float:
        """IC-weighted ensemble: weight horizons by their rolling IC."""
        weights = self.get_weights()

        total_w = sum(weights.values())
        if total_w <= 0:
            # Fallback: if all weights are zero, equal-weight all horizons
            return float(np.mean(list(z_values.values())))

        result = 0.0
        for h, z in z_values.items():
            result += weights.get(h, 0.0) * z
        return result / total_w

    def _ridge_primary_predict(self, z_values: Dict[int, float]) -> float:
        """AlphaRunner-style static horizon weighting for ridge_primary configs."""
        weights = self.get_weights()
        total_w = sum(weights.values()) or 1.0
        result = 0.0
        for h, z in z_values.items():
            result += weights.get(h, 0.0) * z
        return result / total_w

    def update_ic(self, horizon: int, pred: float, actual_return: float) -> None:
        """Update rolling IC for one horizon after realized return is known."""
        monitor = self._ic_monitors.get(horizon)
        if monitor is not None:
            monitor.update(pred, actual_return)

    def get_weights(self) -> Dict[int, float]:
        """Current normalized weights per horizon.

        In mean_zscore mode: equal weights.
        In ic_weighted mode: max(0, rolling_ic) per horizon, with at least
        one horizon always active.
        """
        horizons = [hm["horizon"] for hm in self._horizon_models]

        if self._method == "mean_zscore":
            w = 1.0 / len(horizons)
            return {h: w for h in horizons}
        if self._method == "ridge_primary":
            raw_weights = {
                hm["horizon"]: max(float(hm.get("ic", 0.0)), 0.001)
                for hm in self._horizon_models
            }
            total_w = sum(raw_weights.values()) or 1.0
            return {h: w / total_w for h, w in raw_weights.items()}

        # IC-weighted
        raw_weights = {}
        for h in horizons:
            monitor = self._ic_monitors.get(h)
            if monitor is not None and monitor.n_samples >= 50:
                ic = monitor.rolling_ic
                raw_weights[h] = max(0.0, ic - self._config.ic_min_threshold)
            else:
                raw_weights[h] = 1.0  # default weight until enough data

        # Ensure at least one horizon is active
        if all(w <= 0 for w in raw_weights.values()):
            # Give equal weight to all — better than nothing
            return {h: 1.0 / len(horizons) for h in horizons}

        return raw_weights
