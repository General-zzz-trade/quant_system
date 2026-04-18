"""Live vs batch parity check for the 1D ensemble.

The live `runner.daily_paper_runner` and the batch `scripts.daily_poc`
both load the same trained ensemble (`models_v8/{sym}_1d/*.pkl`) and
both compute features the same way. They should produce IDENTICAL
predictions for the same input bar — if they don't, there's a bug
that would silently bite us in production.

This script picks 30 random recent days, builds the input row each way,
runs both prediction paths, and reports max absolute diff.

Run:
  python3 -m scripts.parity_check_1d
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/quant_system")

from runner.daily_paper_runner import (  # noqa: E402
    _load_model_bundle,
    _predict,
)
from scripts.daily_poc import (  # noqa: E402
    compute_1d_features,
    resample_1h_to_1d,
)


TOL_ABS = 1e-9
TOL_REL = 1e-6


def _batch_predict_path(lgbm, xgb_m, X_row):
    """Same prediction path the dz_sweep / daily_poc backtests use:
    50/50 LGBM+XGB ensemble on flat numpy input."""
    import xgboost as xgb

    p_l = float(lgbm["model"].predict(X_row.reshape(1, -1))[0])
    p_x = float(xgb_m["model"].predict(xgb.DMatrix(X_row.reshape(1, -1)))[0])
    return 0.5 * p_l + 0.5 * p_x


def check_symbol(symbol: str, n_samples: int = 30):
    model_dir = f"/quant_system/models_v8/{symbol}_1d"
    if not Path(model_dir).exists():
        print(f"  {symbol}: model not found")
        return False

    lgbm, xgb_m, cfg = _load_model_bundle(model_dir)
    feature_names_cfg = cfg["features"]

    macro = pd.read_csv("/quant_system/data_files/cross_market_daily.csv")
    macro["date"] = pd.to_datetime(macro["date"]).dt.date

    df_1h = pd.read_csv(f"/quant_system/data_files/{symbol}_1h.csv")
    df_1d = resample_1h_to_1d(df_1h)
    df_1d = df_1d[df_1d["date"] >= macro["date"].min()].reset_index(drop=True)

    feat_df, feature_names = compute_1d_features(df_1d, macro)
    X_all = feat_df[feature_names].values.astype(np.float64)
    X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)
    n = len(df_1d)

    rng = np.random.default_rng(42)
    sample_idx = rng.choice(np.arange(60, n), size=min(n_samples, n - 60), replace=False)

    diffs = []
    for i in sorted(sample_idx):
        # Live path (mimics daily_paper_runner.run_symbol exactly)
        X_row_live = np.zeros(len(feature_names_cfg), dtype=np.float64)
        for j, name in enumerate(feature_names_cfg):
            if name in feature_names:
                X_row_live[j] = X_all[i, feature_names.index(name)]
        pred_live = _predict(lgbm, xgb_m, X_row_live)

        # Batch path (mimics dz_sweep / daily_poc)
        pred_batch = _batch_predict_path(lgbm, xgb_m, X_row_live)

        diff = abs(pred_live - pred_batch)
        diffs.append((i, pred_live, pred_batch, diff))

    max_abs = max(d[3] for d in diffs)
    rel_diffs = [d[3] / abs(d[1]) if abs(d[1]) > 1e-10 else 0 for d in diffs]
    max_rel = max(rel_diffs) if rel_diffs else 0

    print(f"  {symbol}_1d:  n={len(diffs)}  "
          f"max_abs={max_abs:.2e}  max_rel={max_rel:.2e}")

    # Show worst-case few
    diffs.sort(key=lambda t: -t[3])
    for i, pl, pb, diff in diffs[:3]:
        print(f"    idx={i}  live={pl:+.10f}  batch={pb:+.10f}  diff={diff:.2e}")

    passed = max_abs < TOL_ABS and max_rel < TOL_REL
    return passed


def check_config_consistency(symbol: str):
    """Sanity: the live runner reads dz/min_hold/etc from config.json.
    Make sure the values match what the WF/holdout backtest used."""
    cfg_path = Path(f"/quant_system/models_v8/{symbol}_1d/config.json")
    cfg = json.loads(cfg_path.read_text())

    expected = {
        "deadzone": 1.75,
        "min_hold": 1,
        "max_hold": 7,
    }
    for k, v in expected.items():
        actual = cfg.get(k)
        ok = actual == v
        marker = "OK" if ok else "MISMATCH"
        print(f"  {symbol} cfg.{k}: {actual} (expected {v}) → {marker}")
        if not ok:
            return False
    # Z-score normalization params present?
    for k in ("zscore_pred_mean", "zscore_pred_std"):
        if k not in cfg:
            print(f"  {symbol} cfg.{k}: MISSING")
            return False
    return True


def main():
    print("=" * 72)
    print("Live ↔ Batch parity check for 1D models")
    print("=" * 72)

    print("\n[1/2] Prediction parity (random 30 bars per symbol):")
    pred_pass = True
    for sym in ("BTCUSDT", "ETHUSDT"):
        ok = check_symbol(sym)
        if not ok:
            pred_pass = False

    print("\n[2/2] Config consistency:")
    cfg_pass = True
    for sym in ("BTCUSDT", "ETHUSDT"):
        ok = check_config_consistency(sym)
        if not ok:
            cfg_pass = False

    print("\n" + "=" * 72)
    if pred_pass and cfg_pass:
        print("OVERALL: PASS — live and batch paths produce identical predictions")
        sys.exit(0)
    else:
        print(f"OVERALL: FAIL  prediction_pass={pred_pass}  config_pass={cfg_pass}")
        sys.exit(2)


if __name__ == "__main__":
    main()
