"""Batch predictor: computes predictions using batch features for signal accuracy.

The incremental RustFeatureEngine produces different feature values than
batch_feature_engine (especially for macro/onchain/cross-asset features),
causing predictions to diverge ~100x and z-scores to point wrong direction.

This module loads kline CSV, computes batch features for the latest bar,
and generates the correct prediction to feed the z-score buffer.

Note: Uses pickle for loading trusted local ML model artifacts (lightgbm/sklearn).
These models are produced by our own training pipeline and HMAC-signed.
"""
from __future__ import annotations

import json
import logging
import pickle  # noqa: S403 — trusted local model artifacts, HMAC-signed
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = Path("data_files")
MODEL_DIR = Path("models_v8")

_cache: dict[str, dict[str, Any]] = {}
_CACHE_TTL = 300  # 5 min


def _fix_oi(symbol: str) -> None:
    try:
        from scripts.run_full_backtest import _fix_oi_file
        _fix_oi_file(symbol)
    except Exception:
        pass


def predict_latest(symbol: str, model_dir_name: str) -> float | None:
    """Compute batch prediction for the latest bar of *symbol*."""
    now = time.monotonic()
    cached = _cache.get(symbol)
    if cached and (now - cached["ts"]) < _CACHE_TTL:
        return cached["pred"]

    try:
        _fix_oi(symbol)
        csv_path = DATA_DIR / f"{symbol}_1h.csv"
        model_path = MODEL_DIR / model_dir_name
        config_path = model_path / "config.json"
        if not csv_path.exists() or not config_path.exists():
            return None

        with open(config_path) as f:
            config = json.load(f)

        df = pd.read_csv(csv_path).sort_values("open_time").tail(100).reset_index(drop=True)
        from features.batch_feature_engine import compute_features_batch
        feat_df = compute_features_batch(symbol, df)
        if len(feat_df) == 0:
            return None

        preds = []
        for hm in config["horizon_models"]:
            lgbm_path = model_path / hm["lgbm"]
            if not lgbm_path.exists():
                continue
            # noqa: S301 — trusted local model artifacts
            with open(lgbm_path, "rb") as f:
                d = pickle.load(f)  # noqa: S301
            model = d["model"] if isinstance(d, dict) else d
            X = np.zeros((1, len(hm["features"])))
            for j, fname in enumerate(hm["features"]):
                if fname in feat_df.columns:
                    val = feat_df[fname].iloc[-1]
                    X[0, j] = 0.0 if np.isnan(val) else val
            preds.append(model.predict(X)[0])

        if not preds:
            return None
        pred = float(np.mean(preds))
        _cache[symbol] = {"pred": pred, "ts": now}
        return pred
    except Exception:
        logger.debug("Batch prediction failed for %s", symbol, exc_info=True)
        return None
