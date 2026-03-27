"""AlphaSignalModel — ensemble prediction + z-score discretization.

Replaces the prediction and signal logic from AlphaRunner with two
composable, testable components:

- EnsemblePredictor: Ridge(60%) + LGBM(40%) IC-weighted ensemble
  - Optional OnlineRidge: incremental RLS weight updates between retrains
- SignalDiscretizer: z-score normalize + deadzone + min-hold + z-clamp
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from _quant_hotpath import (  # type: ignore[import-untyped]
    RustMLDecision,
    rust_compute_feature_signal,
    rust_adaptive_ensemble_calibrate,
)

# Rust ML decision state machine — available for alternative inference path
# that bypasses Python ensemble when all features are Rust-computed.
MLDecisionType = RustMLDecision
_rust_feature_signal = rust_compute_feature_signal
_rust_ensemble_calibrate = rust_adaptive_ensemble_calibrate

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


def _rust_or_sklearn_ridge(hm: dict, feature_values: list[float]) -> float:
    """Predict using RustRidgePredictor if available, else sklearn Ridge."""
    rust_ridge = hm.get("rust_ridge")
    if rust_ridge is not None:
        try:
            return float(rust_ridge.predict(feature_values))
        except Exception:
            logger.debug("RustRidgePredictor.predict failed, falling back to sklearn", exc_info=True)
    return float(hm["ridge"].predict([feature_values])[0])


def _rust_or_sklearn_tree(hm: dict, feature_values: list[float]) -> float:
    """Predict using RustTreePredictor if available, else sklearn LGBM."""
    rust_tree = hm.get("rust_tree")
    if rust_tree is not None:
        try:
            return float(rust_tree.predict_array(feature_values))
        except Exception:
            logger.debug("RustTreePredictor.predict_array failed, falling back to sklearn", exc_info=True)
    return float(hm["lgbm"].predict([feature_values])[0])


class EnsemblePredictor:
    """Ridge(60%) + LGBM(40%) IC-weighted ensemble predictor.

    For 4h models (config version contains "4h"), uses Ridge-only
    because LGBM overfits on low-frequency data.

    When OnlineRidge is available, it replaces the static sklearn Ridge
    with incremental RLS updates between weekly retrains. The static
    Ridge remains as a fallback if OnlineRidge fails or is unavailable.
    """

    def __init__(self, horizon_models: list[dict], config: dict) -> None:
        self._horizon_models = horizon_models
        self._config = config
        self._ridge_only_4h = "4h" in config.get("version", "")
        self._ridge_w = config.get("ridge_weight", 0.6)
        self._lgbm_w = config.get("lgbm_weight", 0.4)

        # Online Ridge: incremental weight updates between retrains
        self._online_ridge: Any | None = None
        self._online_ridge_features: list[str] | None = None
        self._init_online_ridge(config)

    def _init_online_ridge(self, config: dict) -> None:
        """Try to create OnlineRidge from the first horizon model's Ridge weights.

        Best-effort: if anything fails, self._online_ridge stays None and
        the static Ridge path is used. Never raises.
        """
        if not self._horizon_models:
            return

        forgetting = config.get("online_ridge_forgetting", 0.99)

        try:
            from alpha.online_ridge import OnlineRidge
        except Exception:
            logger.debug("OnlineRidge import failed, using static Ridge", exc_info=True)
            return

        # Use the first horizon model that has a Ridge with extractable weights
        for hm in self._horizon_models:
            ridge_model = hm.get("ridge")
            if ridge_model is None:
                continue

            rf = hm.get("ridge_features") or hm.get("features", [])
            n_features = len(rf)
            if n_features == 0:
                continue

            try:
                # Extract weights from sklearn Ridge or compatible object
                coef = np.asarray(ridge_model.coef_, dtype=np.float64).ravel()
                intercept = float(ridge_model.intercept_)
                if len(coef) != n_features:
                    logger.warning(
                        "OnlineRidge: coef size %d != feature count %d, skipping",
                        len(coef), n_features,
                    )
                    continue

                online = OnlineRidge(
                    n_features=n_features,
                    forgetting_factor=forgetting,
                )
                online.load_from_weights(coef, intercept)
                self._online_ridge = online
                self._online_ridge_features = list(rf)
                logger.info(
                    "OnlineRidge initialized: %d features, forgetting=%.4f",
                    n_features, forgetting,
                )
                return
            except Exception:
                logger.debug("OnlineRidge init from Ridge weights failed", exc_info=True)
                continue

    def predict(self, feat_dict: dict) -> float | None:
        """Produce ensemble prediction from feature dict.

        Uses Rust-native predictors (RustRidgePredictor / RustTreePredictor)
        when available for faster inference. Falls back to sklearn if Rust
        predictors are absent or raise an exception.

        When OnlineRidge is available and initialized, it replaces the static
        Ridge prediction for the first horizon model. Falls back to static
        Ridge if OnlineRidge.predict raises.

        Returns weighted average across horizons (IC-weighted), or None
        if no valid predictions.
        """
        if not self._horizon_models:
            return None

        weighted_sum = 0.0
        weight_total = 0.0

        for hm_idx, hm in enumerate(self._horizon_models):
            feats = hm["features"]
            x = [_safe_val(feat_dict.get(f), f) for f in feats]

            ic = max(hm["ic"], 0.001)

            ridge_model = hm.get("ridge")
            if ridge_model is not None:
                rf = hm.get("ridge_features") or feats
                rx = [_safe_val(feat_dict.get(f), f) for f in rf]

                # Try OnlineRidge for the first horizon model
                ridge_pred = None
                if (
                    hm_idx == 0
                    and self._online_ridge is not None
                    and self._online_ridge_features is not None
                ):
                    try:
                        online_x = [
                            _safe_val(feat_dict.get(f), f)
                            for f in self._online_ridge_features
                        ]
                        ridge_pred = float(self._online_ridge.predict(
                            np.array(online_x, dtype=np.float64),
                        ))
                    except Exception:
                        logger.debug(
                            "OnlineRidge.predict failed, falling back to static Ridge",
                            exc_info=True,
                        )
                        ridge_pred = None

                # Fallback to static Ridge
                if ridge_pred is None:
                    ridge_pred = _rust_or_sklearn_ridge(hm, rx)

                if self._ridge_only_4h:
                    pred = ridge_pred
                else:
                    lgbm_pred = _rust_or_sklearn_tree(hm, x)
                    pred = ridge_pred * self._ridge_w + lgbm_pred * self._lgbm_w
            else:
                pred = _rust_or_sklearn_tree(hm, x)

            weighted_sum += pred * ic
            weight_total += ic

        if weight_total <= 0:
            return None

        # Cache last prediction features for online update
        self._last_feat_dict = feat_dict

        return weighted_sum / weight_total

    def update_online_ridge(self, realized_return: float) -> bool:
        """Update OnlineRidge weights with the realized return from the previous bar.

        Called after each bar when the actual return is known.

        Args:
            realized_return: The actual forward return (e.g., log return or pct return)
                that the model was trying to predict.

        Returns:
            True if the update succeeded, False otherwise.
        """
        if self._online_ridge is None or self._online_ridge_features is None:
            return False

        feat_dict = getattr(self, "_last_feat_dict", None)
        if feat_dict is None:
            return False

        try:
            x = np.array(
                [_safe_val(feat_dict.get(f), f) for f in self._online_ridge_features],
                dtype=np.float64,
            )
            pred_error = self._online_ridge.update(x, realized_return)

            # Log periodically (every 100 updates)
            n = self._online_ridge.n_updates
            if n > 0 and n % 100 == 0:
                logger.info(
                    "OnlineRidge update #%d: pred_error=%.6f, drift=%.6f",
                    n, pred_error, self._online_ridge.weight_drift,
                )
            return True
        except Exception:
            logger.debug("OnlineRidge.update failed", exc_info=True)
            return False

    def reset_online_ridge(self) -> None:
        """Reset OnlineRidge to static weights (call after weekly retrain)."""
        if self._online_ridge is not None:
            try:
                self._online_ridge.reset_to_static()
                logger.info("OnlineRidge reset to static weights")
            except Exception:
                logger.debug("OnlineRidge.reset_to_static failed", exc_info=True)

    @property
    def online_ridge_stats(self) -> dict | None:
        """Return OnlineRidge statistics, or None if unavailable."""
        if self._online_ridge is None:
            return None
        try:
            return self._online_ridge.stats
        except Exception:
            return None

    def calibrate_weights(
        self,
        score_history: dict,
        return_history: list,
        method: str = "ic_weighted",
        shrinkage: float = 0.3,
        lookback: int = 200,
    ) -> bool:
        """Calibrate ensemble weights adaptively using Rust accelerator.

        Updates ridge_w / lgbm_w in-place if calibration succeeds.

        Returns True if weights were updated, False otherwise.
        """
        try:
            result = _rust_ensemble_calibrate(
                method, score_history, return_history, shrinkage, lookback,
            )
        except Exception:
            logger.debug("rust_adaptive_ensemble_calibrate failed", exc_info=True)
            return False
        if result is None:
            return False
        ridge_w = result.get("ridge", self._ridge_w)
        lgbm_w = result.get("lgbm", self._lgbm_w)
        # Sanity: weights must be non-negative and sum to ~1
        total = ridge_w + lgbm_w
        if total <= 0:
            return False
        self._ridge_w = ridge_w / total
        self._lgbm_w = lgbm_w / total
        logger.info(
            "Ensemble weights calibrated: ridge=%.3f lgbm=%.3f (method=%s)",
            self._ridge_w, self._lgbm_w, method,
        )
        return True


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

        # Z-clamp: |z| > 3.5 with no position -> cap to deadzone + 0.5
        # Prevents post-warmup z-score spikes from triggering trades
        if abs(z) > 3.5 and current_signal == 0:
            old_z = z
            cap = self.deadzone + 0.5  # just above deadzone, not extreme
            z = cap if z > 0 else -cap
            logger.info(
                "%s Z_CLAMP: |z|=%.1f capped to %.1f "
                "(extreme z with no position -> likely unreliable, dz=%.1f)",
                self._symbol, abs(old_z), abs(z), self.deadzone,
            )

        # Regime filter: widen deadzone by 50% instead of blocking all signals.
        # Position sizing already applies 0.6x discount when regime_active=False,
        # so we only raise the bar slightly — not block outright.
        effective_dz = self.deadzone * 1.5 if not regime_ok else self.deadzone

        signal = int(self._bridge.apply_constraints(
            self._symbol, pred, hour_key,
            deadzone=effective_dz,
            min_hold=self._min_hold,
            max_hold=self._max_hold,
            long_only=self._long_only,
        ))

        return signal, z

    @staticmethod
    def feature_signal(
        momentum: float,
        volatility: float,
        vwap_ratio: float,
        *,
        momentum_threshold: float = 0.001,
        vol_penalty_factor: float = 2.0,
        vwap_weight: float = 0.3,
    ) -> tuple[int, float, float]:
        """Compute a feature-based signal via Rust fast-path.

        Useful as an auxiliary signal for decision modules that need a quick
        directional read from raw features without running the full ML ensemble.

        Args:
            momentum: price momentum (e.g. ret_1 or ma_cross)
            volatility: realized volatility (e.g. vol_20)
            vwap_ratio: (close - vwap) / close deviation

        Returns:
            (side, score, confidence) where side is +1/-1/0,
            score is the raw signal strength, and confidence is [0, 1].
        """
        side_str, score, confidence = _rust_feature_signal(
            momentum, volatility, vwap_ratio,
            momentum_threshold=momentum_threshold,
            vol_penalty_factor=vol_penalty_factor,
            vwap_weight=vwap_weight,
        )
        _SIDE_MAP = {"buy": 1, "sell": -1, "flat": 0}
        side = _SIDE_MAP.get(side_str, 0)
        return side, float(score), float(confidence)
