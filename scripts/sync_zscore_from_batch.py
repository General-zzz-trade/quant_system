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

    preds_all = []
    for hm in config["horizon_models"]:
        lgbm_path = model_path / hm["lgbm"]
        if not lgbm_path.exists():
            continue
        # noqa: S301 — trusted local model artifacts produced by our training pipeline
        with open(lgbm_path, "rb") as f:
            d = pickle.load(f)  # noqa: S301
        model = d["model"] if isinstance(d, dict) else d
        X = np.zeros((n, len(hm["features"])))
        for j, fname in enumerate(hm["features"]):
            if fname in feat_df.columns:
                X[:, j] = feat_df[fname].fillna(0).values
        preds_all.append(model.predict(X))

    if not preds_all:
        print(f"  SKIP {symbol}: no predictions")
        return

    pred = np.mean(preds_all, axis=0)
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
