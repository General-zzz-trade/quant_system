"""Batch predictor: computes predictions using batch features for signal accuracy.

The incremental RustFeatureEngine produces different feature values than
batch_feature_engine (especially for macro/onchain/cross-asset features),
causing predictions to diverge ~100x and z-scores to point wrong direction.

This module loads kline CSV, computes batch features for the latest bar,
and generates the correct prediction to feed the z-score buffer.

Every artifact (config.json and pickled models) is HMAC-verified before it is
parsed; artifact names from config.json must stay inside the model directory.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = Path("data_files")
MODEL_DIR = Path("models_v8")

_cache: dict[str, dict[str, Any]] = {}
_CACHE_TTL = 120  # 2 min — shorter for responsive intra-bar z-score preview


def invalidate_cache(symbol: str | None = None) -> None:
    """Clear cached predictions. Called when new bar data arrives."""
    if symbol is None:
        _cache.clear()
    else:
        _cache.pop(symbol, None)


def _fix_oi(symbol: str) -> None:
    try:
        from scripts.run_full_backtest import _fix_oi_file
        _fix_oi_file(symbol)
    except Exception:
        pass


def predict_latest(symbol: str, model_dir_name: str, *, as_of_ms: int | None = None) -> float | None:
    """Compute batch prediction for the latest bar of *symbol*."""
    from runner.data_contract import inspect_batch_inputs
    inputs = inspect_batch_inputs(symbol, DATA_DIR, as_of_ms=as_of_ms)
    if not inputs["ready"]:
        logger.warning("Batch inputs unavailable for %s: %s", symbol, inputs["reason"])
        return None
    now = time.monotonic()
    cached = _cache.get(symbol)
    identity = (model_dir_name, inputs["as_of_ms"], inputs["signature"])
    if cached and (now - cached["ts"]) < _CACHE_TTL and cached.get("identity") == identity:
        return cached["pred"]

    try:
        csv_path = DATA_DIR / f"{symbol}_1h.csv"
        model_path = MODEL_DIR / model_dir_name
        config_path = model_path / "config.json"
        if not csv_path.exists() or not config_path.exists():
            return None

        from infra.model_signing import load_verified_json, load_verified_pickle, resolve_artifact
        config = load_verified_json(config_path)

        df = pd.read_csv(csv_path)
        df = df[df["open_time"] <= inputs["as_of_ms"]].sort_values("open_time").tail(500).reset_index(drop=True)
        from features.batch_feature_engine import compute_features_batch
        feat_df = compute_features_batch(symbol, df)
        if len(feat_df) == 0:
            return None

        ridge_w = config.get("ridge_weight", 0.6)
        lgbm_w = config.get("lgbm_weight", 0.4)

        # Regime detection: use MA50 slope to pick bull/bear/default models
        horizon_models = config["horizon_models"]
        regime_cfg = config.get("regime_models")
        if regime_cfg and config.get("regime_enabled"):
            closes = df["close"].values
            if len(closes) >= 74:
                ma50 = pd.Series(closes).rolling(50).mean().values
                slope = (ma50[-1] / ma50[-25] - 1) if ma50[-25] > 0 else 0
                threshold = regime_cfg.get("regime_threshold", 0.002)
                if slope < -threshold and "bear" in regime_cfg:
                    bear_models = regime_cfg["bear"]
                    horizon_models = [bear_models[str(h)] for h in sorted(bear_models) if str(h) in bear_models]
                    logger.debug("Regime: BEAR (slope=%.4f), using bear models", slope)
                elif slope > threshold and "bull" in regime_cfg:
                    bull_models = regime_cfg["bull"]
                    horizon_models = [bull_models[str(h)] for h in sorted(bull_models) if str(h) in bull_models]
                    logger.debug("Regime: BULL (slope=%.4f), using bull models", slope)

        preds = []
        for hm in horizon_models:
            # LGBM prediction
            lgbm_path = resolve_artifact(model_path, hm["lgbm"])
            if not lgbm_path.exists():
                continue
            d = load_verified_pickle(lgbm_path)
            lgbm_model = d["model"] if isinstance(d, dict) else d
            X = np.zeros((1, len(hm["features"])))
            for j, fname in enumerate(hm["features"]):
                if fname in feat_df.columns:
                    val = feat_df[fname].iloc[-1]
                    X[0, j] = 0.0 if np.isnan(val) else val
            lgbm_pred = lgbm_model.predict(X)[0]

            # Ridge prediction (if available)
            ridge_path_name = hm.get("ridge")
            ridge_pred = None
            if ridge_path_name:
                ridge_path = resolve_artifact(model_path, ridge_path_name)
                if ridge_path.exists():
                    rd = load_verified_pickle(ridge_path)
                    ridge_model = rd["model"] if isinstance(rd, dict) else rd
                    rf = rd.get("features") or hm.get("ridge_features") or hm["features"]
                    X_r = np.zeros((1, len(rf)))
                    for j, fname in enumerate(rf):
                        if fname in feat_df.columns:
                            val = feat_df[fname].iloc[-1]
                            X_r[0, j] = 0.0 if np.isnan(val) else val
                    ridge_pred = ridge_model.predict(X_r)[0]

            # Ensemble: Ridge(60%) + LGBM(40%), fallback to LGBM-only
            if ridge_pred is not None:
                pred_hm = ridge_pred * ridge_w + lgbm_pred * lgbm_w
            else:
                pred_hm = lgbm_pred
            preds.append(pred_hm)

        if not preds:
            return None
        pred = float(np.mean(preds))
        rechecked = inspect_batch_inputs(symbol, DATA_DIR, as_of_ms=inputs["as_of_ms"])
        if not np.isfinite(pred) or not rechecked["ready"] or rechecked["signature"] != inputs["signature"]:
            return None
        _cache[symbol] = {"pred": pred, "ts": now, "identity": identity}
        return pred
    except Exception:
        logger.debug("Batch prediction failed for %s", symbol, exc_info=True)
        return None
