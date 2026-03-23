"""AlphaSignalModel — ensemble prediction + z-score discretization.

Replaces the prediction and signal logic from AlphaRunner with two
composable, testable components:

- EnsemblePredictor: Ridge(60%) + LGBM(40%) IC-weighted ensemble
- SignalDiscretizer: z-score normalize + deadzone + min-hold + z-clamp
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Neutral fallback values for NaN features — 0.0 would be a directional signal
# for ratio/RSI features where the neutral value is not zero.
_NEUTRAL_DEFAULTS: dict[str, float] = {
    "ls_ratio": 1.0,
    "top_trader_ls_ratio": 1.0,
    "taker_buy_ratio": 0.5,
    "vol_regime": 1.0,
    "bb_pctb_20": 0.5,
    "rsi_14": 50.0,
    "rsi_6": 50.0,
}


def _safe_val(v: Any, feat_name: str = "") -> float:
    """Convert None/NaN to neutral value for model input."""
    neutral = _NEUTRAL_DEFAULTS.get(feat_name, 0.0)
    if v is None:
        return neutral
    try:
        f = float(v)
        return neutral if np.isnan(f) else f
    except (TypeError, ValueError):
        return neutral


class EnsemblePredictor:
    """Ridge(60%) + LGBM(40%) IC-weighted ensemble predictor.

    For 4h models (config version contains "4h"), uses Ridge-only
    because LGBM overfits on low-frequency data.
    """

    def __init__(self, horizon_models: list[dict], config: dict) -> None:
        self._horizon_models = horizon_models
        self._config = config
        self._ridge_only_4h = "4h" in config.get("version", "")
        self._ridge_w = config.get("ridge_weight", 0.6)
        self._lgbm_w = config.get("lgbm_weight", 0.4)

    def predict(self, feat_dict: dict) -> float | None:
        """Produce ensemble prediction from feature dict.

        Returns weighted average across horizons (IC-weighted), or None
        if no valid predictions.
        """
        if not self._horizon_models:
            return None

        weighted_sum = 0.0
        weight_total = 0.0

        for hm in self._horizon_models:
            feats = hm["features"]
            x = [_safe_val(feat_dict.get(f), f) for f in feats]

            ic = max(hm["ic"], 0.001)

            ridge_model = hm.get("ridge")
            if ridge_model is not None:
                rf = hm.get("ridge_features") or feats
                rx = [_safe_val(feat_dict.get(f), f) for f in rf]
                ridge_pred = float(ridge_model.predict([rx])[0])

                if self._ridge_only_4h:
                    pred = ridge_pred
                else:
                    lgbm_pred = float(hm["lgbm"].predict([x])[0])
                    pred = ridge_pred * self._ridge_w + lgbm_pred * self._lgbm_w
            else:
                pred = float(hm["lgbm"].predict([x])[0])

            weighted_sum += pred * ic
            weight_total += ic

        if weight_total <= 0:
            return None
        return weighted_sum / weight_total


class SignalDiscretizer:
    """Z-score normalize + deadzone + min-hold signal discretizer.

    Uses RustInferenceBridge for z-score normalization and constraint
    application. Adds z-clamp guard for extreme z with no position.
    """

    def __init__(
        self,
        bridge: Any,
        symbol: str,
        deadzone: float,
        min_hold: int,
        max_hold: int,
        long_only: bool = False,
    ) -> None:
        self._bridge = bridge
        self._symbol = symbol
        self._deadzone = deadzone
        self._min_hold = min_hold
        self._max_hold = max_hold
        self._long_only = long_only

    @property
    def deadzone(self) -> float:
        return self._deadzone

    @deadzone.setter
    def deadzone(self, value: float) -> None:
        self._deadzone = value

    @property
    def min_hold(self) -> int:
        return self._min_hold

    @min_hold.setter
    def min_hold(self, value: int) -> None:
        self._min_hold = value

    @property
    def max_hold(self) -> int:
        return self._max_hold

    @max_hold.setter
    def max_hold(self, value: int) -> None:
        self._max_hold = value

    def discretize(
        self,
        pred: float,
        hour_key: int,
        regime_ok: bool,
        current_signal: int = 0,
    ) -> tuple[int, float]:
        """Discretize a raw prediction into a trading signal.

        Returns:
            (signal, z) where signal is +1/-1/0 and z is the
            (possibly clamped) z-score.
        """
        # Z-score via RustInferenceBridge (returns None during warmup)
        z_val = self._bridge.zscore_normalize(self._symbol, pred, hour_key)
        if z_val is None:
            return 0, 0.0

        # Clip extreme z-scores to [-5, 5]
        z = max(-5.0, min(5.0, z_val))

        # Z-clamp: |z| > 3.5 with no position -> cap +/-3.0
        if abs(z) > 3.5 and current_signal == 0:
            old_z = z
            z = 3.0 if z > 0 else -3.0
            logger.info(
                "%s Z_CLAMP: |z|=%.1f capped to %.1f "
                "(extreme z with no position -> likely unreliable)",
                self._symbol, abs(old_z), z,
            )

        # Regime filter: pass deadzone=999 to force flat
        effective_dz = 999.0 if not regime_ok else self.deadzone

        signal = int(self._bridge.apply_constraints(
            self._symbol, pred, hour_key,
            deadzone=effective_dz,
            min_hold=self._min_hold,
            max_hold=self._max_hold,
            long_only=self._long_only,
        ))

        return signal, z
