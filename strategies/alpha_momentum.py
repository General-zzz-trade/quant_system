"""Alpha Momentum strategy -- wraps Ridge+LightGBM ensemble in StrategyProtocol."""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np

from strategies.base import Signal

logger = logging.getLogger(__name__)


class AlphaMomentumStrategy:
    """StrategyProtocol wrapper around the production Ridge+LightGBM ensemble.

    The strategy takes enriched features, runs the ensemble prediction,
    applies z-score normalization and deadzone thresholding, and returns
    a discrete Signal.

    Parameters
    ----------
    model_info : dict
        Output of ``scripts.ops.model_loader.load_model`` -- must contain
        ``model``, ``features``, ``config``, ``deadzone``, ``min_hold``,
        ``zscore_window``, ``zscore_warmup``.
    symbol : str
        Trading symbol (e.g. "ETHUSDT").
    timeframe : str
        Bar timeframe, default "1h".
    ridge_weight : float
        Weight for Ridge in the Ridge+LightGBM blend (default 0.6).
    """

    name: str = "alpha_momentum"
    version: str = "1.0"
    venue: str = "binance/bybit"

    def __init__(
        self,
        model_info: Dict[str, Any],
        symbol: str = "ETHUSDT",
        timeframe: str = "1h",
        ridge_weight: float = 0.6,
    ) -> None:
        self.timeframe = timeframe
        self._symbol = symbol
        self._model = model_info["model"]
        self._features = model_info["features"]
        self._config = model_info["config"]
        self._deadzone = model_info["deadzone"]
        self._min_hold = model_info["min_hold"]
        self._zscore_window = model_info["zscore_window"]
        self._zscore_warmup = model_info["zscore_warmup"]
        self._ridge_weight = ridge_weight

        # Horizon models for ensemble (may include ridge, lgbm, xgb)
        self._horizon_models = model_info.get("horizon_models", [])

        # Rolling z-score state
        self._pred_buffer: list[float] = []
        self._current_signal: int = 0
        self._hold_counter: int = 0
        self._bars_processed: int = 0

    def generate_signal(self, features: Dict[str, Any]) -> Signal:
        """Run ensemble prediction and return a Signal."""
        self._bars_processed += 1

        raw_pred = self._predict(features)
        z = self._update_zscore(raw_pred)

        # Min-hold enforcement
        if self._hold_counter > 0:
            self._hold_counter -= 1
            return Signal(
                direction=self._current_signal,
                confidence=min(1.0, abs(z)) if z is not None else 0.5,
                meta={"raw_pred": raw_pred, "zscore": z, "held": True},
            )

        # Deadzone thresholding
        if z is None:
            direction = 0
        elif z > self._deadzone:
            direction = 1
        elif z < -self._deadzone:
            direction = -1
        else:
            direction = 0

        # Update hold counter on signal change
        if direction != 0 and direction != self._current_signal:
            self._hold_counter = self._min_hold
        self._current_signal = direction

        return Signal(
            direction=direction,
            confidence=min(1.0, abs(z)) if z is not None else 0.0,
            meta={"raw_pred": raw_pred, "zscore": z, "held": False},
        )

    def _predict(self, features: Dict[str, Any]) -> float:
        """Run Ridge+LightGBM blend across horizon models."""
        if not self._horizon_models:
            # Single-model fallback
            x = np.zeros((1, len(self._features)))
            for j, fname in enumerate(self._features):
                x[0, j] = _safe_val(features.get(fname))
            return float(self._model.predict(x)[0])

        preds: list[float] = []
        for hm in self._horizon_models:
            feat_names = hm["features"]
            x = np.zeros((1, len(feat_names)))
            for j, fname in enumerate(feat_names):
                x[0, j] = _safe_val(features.get(fname))

            lgbm_pred = float(hm["lgbm"].predict(x)[0])

            ridge = hm.get("ridge")
            if ridge is not None:
                ridge_feats = hm.get("ridge_features") or feat_names
                xr = np.zeros((1, len(ridge_feats)))
                for j, fname in enumerate(ridge_feats):
                    xr[0, j] = _safe_val(features.get(fname))
                ridge_pred = float(ridge.predict(xr)[0])
                blended = (self._ridge_weight * ridge_pred
                           + (1.0 - self._ridge_weight) * lgbm_pred)
                preds.append(blended)
            else:
                preds.append(lgbm_pred)

        return float(np.mean(preds))

    def _update_zscore(self, pred: float) -> Optional[float]:
        """Push prediction into rolling z-score buffer."""
        self._pred_buffer.append(pred)
        if len(self._pred_buffer) > self._zscore_window:
            self._pred_buffer = self._pred_buffer[-self._zscore_window:]
        if len(self._pred_buffer) < self._zscore_warmup:
            return None
        arr = np.array(self._pred_buffer)
        std = arr.std()
        if std < 1e-10:
            return 0.0
        return float((pred - arr.mean()) / std)

    def validate_config(self) -> bool:
        """Check that models and feature lists are present."""
        if self._model is None and not self._horizon_models:
            return False
        if not self._features:
            return False
        if self._deadzone <= 0:
            return False
        return True

    def describe(self) -> str:
        n_horizons = len(self._horizon_models) or 1
        return (
            f"Alpha Momentum ({self._symbol} {self.timeframe}): "
            f"Ridge({self._ridge_weight:.0%})+LightGBM({1 - self._ridge_weight:.0%}) "
            f"ensemble, {n_horizons} horizon(s), "
            f"deadzone={self._deadzone}, min_hold={self._min_hold}"
        )


def _safe_val(v: Any) -> float:
    """Convert value to float, treating None/NaN as 0.0."""
    if v is None:
        return 0.0
    try:
        f = float(v)
        if np.isnan(f):
            return 0.0
        return f
    except (TypeError, ValueError):
        return 0.0
