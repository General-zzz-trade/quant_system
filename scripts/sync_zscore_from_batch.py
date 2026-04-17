#!/usr/bin/env python3
"""Pre-compute z-score checkpoints from batch predictions.

Ensures live z-score buffer matches backtest predictions, preventing
signal direction errors caused by incremental vs batch feature divergence.

Run before starting the trading service:
    python3 scripts/sync_zscore_from_batch.py
    sudo systemctl restart binance-alpha.service

Note: Uses pickle for loading trusted local ML model artifacts (lightgbm/sklearn).
These models are produced by our own training pipeline and HMAC-signed.
"""
from __future__ import annotations

import json
import pickle  # noqa: S403 — trusted local model artifacts, HMAC-signed
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/quant_system")

from features.batch_feature_engine import compute_features_batch

DATA_DIR = Path("data_files")
MODEL_DIR = Path("models_v8")
CHECKPOINT_DIR = Path("data/runtime/zscore_checkpoints")

MODELS = {
    "BTCUSDT": "BTCUSDT_gate_v2",
    "ETHUSDT": "ETHUSDT_gate_v2",
}


def _fix_oi(symbol: str) -> None:
    try:
        from scripts.run_full_backtest import _fix_oi_file
        _fix_oi_file(symbol)
    except Exception:
        pass


def sync_one(symbol: str, model_dir_name: str) -> None:
    _fix_oi(symbol)
    csv_path = DATA_DIR / f"{symbol}_1h.csv"
    if not csv_path.exists():
        print(f"  SKIP {symbol}: no data file")
        return

    model_path = MODEL_DIR / model_dir_name
    config_path = model_path / "config.json"
    if not config_path.exists():
        print(f"  SKIP {symbol}: no config.json")
        return

    with open(config_path) as f:
        config = json.load(f)

    df = pd.read_csv(csv_path).sort_values("open_time").reset_index(drop=True)
    feat_df = compute_features_batch(symbol, df)
    n = len(feat_df)

    ridge_w = config.get("ridge_weight", 0.6)
    lgbm_w = config.get("lgbm_weight", 0.4)

    # Regime-conditional: compute per-bar regime and use corresponding model
    regime_cfg = config.get("regime_models")
    regime_enabled = config.get("regime_enabled", False) and regime_cfg
    regime_labels = np.zeros(n)  # 0=default, 1=bull, -1=bear
    if regime_enabled:
        closes_arr = df["close"].values[-n:]
        ma50 = pd.Series(closes_arr).rolling(50).mean().values
        threshold = regime_cfg.get("regime_threshold", 0.002)
        for i in range(74, n):
            slope = (ma50[i] / ma50[i - 24] - 1) if ma50[i - 24] > 0 else 0
            if slope < -threshold:
                regime_labels[i] = -1
            elif slope > threshold:
                regime_labels[i] = 1

    def _load_and_predict(hm_list):
        """Load models from hm_list and return (n,) prediction array."""
        h_preds = []
        for hm in hm_list:
            lgbm_path = model_path / hm["lgbm"]
            if not lgbm_path.exists():
                continue
            with open(lgbm_path, "rb") as f:
                d = pickle.load(f)  # noqa: S301
            lgbm_model = d["model"] if isinstance(d, dict) else d
            X = np.zeros((n, len(hm["features"])))
            for j, fname in enumerate(hm["features"]):
                if fname in feat_df.columns:
                    X[:, j] = feat_df[fname].fillna(0).values
            lgbm_pred = lgbm_model.predict(X)

            ridge_path_name = hm.get("ridge")
            ridge_pred = None
            if ridge_path_name:
                ridge_path = model_path / ridge_path_name
                if ridge_path.exists():
                    with open(ridge_path, "rb") as f:
                        rd = pickle.load(f)  # noqa: S301
                    ridge_model = rd["model"] if isinstance(rd, dict) else rd
                    rf = rd.get("features") or hm.get("ridge_features") or hm["features"]
                    X_r = np.zeros((n, len(rf)))
                    for j, fname in enumerate(rf):
                        if fname in feat_df.columns:
                            X_r[:, j] = feat_df[fname].fillna(0).values
                    ridge_pred = ridge_model.predict(X_r)

            if ridge_pred is not None:
                h_preds.append(ridge_pred * ridge_w + lgbm_pred * lgbm_w)
            else:
                h_preds.append(lgbm_pred)
        if not h_preds:
            return np.zeros(n)
        return np.mean(h_preds, axis=0)

    if regime_enabled:
        # Compute predictions for each regime
        default_pred = _load_and_predict(config["horizon_models"])
        bull_hms = [regime_cfg["bull"][h] for h in sorted(regime_cfg.get("bull", {}))]
        bear_hms = [regime_cfg["bear"][h] for h in sorted(regime_cfg.get("bear", {}))]
        bull_pred = _load_and_predict(bull_hms) if bull_hms else default_pred
        bear_pred = _load_and_predict(bear_hms) if bear_hms else default_pred

        # Per-bar regime-switched prediction
        pred = np.where(regime_labels == 1, bull_pred,
                        np.where(regime_labels == -1, bear_pred, default_pred))
    else:
        pred = _load_and_predict(config["horizon_models"])

    if pred is None or len(pred) == 0:
        print(f"  SKIP {symbol}: no predictions")
        return

    zscore_window = config.get("zscore_window", 720)
    buf = pred[-zscore_window:].tolist()

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = {"zscore_buf": {symbol: buf}}
    path = CHECKPOINT_DIR / f"{symbol}.json"
    with open(path, "w") as f:
        json.dump(checkpoint, f)

    arr = np.array(buf)
    z_last = (buf[-1] - arr.mean()) / max(arr.std(), 1e-10)
    print(f"  {symbol}: {len(buf)} predictions, z={z_last:+.4f}, mean={arr.mean():.6f}, std={arr.std():.6f}")


def main():
    print("Syncing z-score checkpoints from batch predictions...")
    for symbol, model_dir in MODELS.items():
        sync_one(symbol, model_dir)
    print("Done. Restart trading service to load new checkpoints.")


if __name__ == "__main__":
    main()
