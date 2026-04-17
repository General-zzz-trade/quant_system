#!/usr/bin/env python3
"""Train production 1D ensemble models for BTC and ETH.

Outputs to models_v8/{BTCUSDT,ETHUSDT}_1d/ matching the 4h layout:
  - lgbm_v8.pkl / xgb_v8.pkl
  - config.json
  - features.json
  - zscore_warmup.json
"""
from __future__ import annotations
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/quant_system")

from scripts.daily_poc import (  # noqa: E402
    compute_1d_features,
    resample_1h_to_1d,
)


PROD_CONFIG = {
    "BTCUSDT": {
        "deadzone": 1.75, "min_hold": 1, "max_hold": 7, "long_only": False,
        "wf_sharpe": 2.76, "wf_ic_pearson": 0.0246, "wf_ic_spearman": 0.0158,
        "wf_trades": 69, "wf_win_rate": 63.8, "wf_max_dd_pct": 1.57,
        "bootstrap_p_positive": 0.996,
        "positive_months": 26, "total_months": 37,
    },
    "ETHUSDT": {
        "deadzone": 1.75, "min_hold": 1, "max_hold": 7, "long_only": False,
        "wf_sharpe": 4.25, "wf_ic_pearson": 0.0364, "wf_ic_spearman": 0.0288,
        "wf_trades": 64, "wf_win_rate": 57.8, "wf_max_dd_pct": 0.75,
        "bootstrap_p_positive": 1.000,
        "positive_months": 23, "total_months": 34,
    },
}
ENSEMBLE_WEIGHTS = [0.5, 0.5]


def _dump_bundle(obj, path):
    """Save model bundle — matches production loader format."""
    import pickle as _pkl  # noqa: S403 — trusted local artefact

    with open(path, "wb") as f:
        _pkl.dump(obj, f)  # noqa: S301


def train_symbol(symbol, macro):
    print("=" * 72)
    print(f"TRAIN 1D PRODUCTION: {symbol}")
    print("=" * 72)

    df_1h = pd.read_csv(f"/quant_system/data_files/{symbol}_1h.csv")
    df_1d = resample_1h_to_1d(df_1h)
    df_1d = df_1d[df_1d["date"] >= macro["date"].min()].reset_index(drop=True)

    feat_df, feature_names = compute_1d_features(df_1d, macro)
    closes = df_1d["close"].values.astype(np.float64)

    log_rets = np.diff(np.log(closes), prepend=np.nan)
    target = np.roll(log_rets, -1)
    target[-1] = np.nan

    X_all = feat_df[feature_names].values.astype(np.float64)
    X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)

    valid = ~np.isnan(target)
    X_tr = X_all[valid]
    y_tr = target[valid]

    print(f"  Training samples: {len(X_tr)} days  Features: {len(feature_names)}")

    import lightgbm as lgb
    import xgboost as xgb

    holdout_start = len(X_tr) - 90
    X_train_final = X_tr[:holdout_start]
    y_train_final = y_tr[:holdout_start]
    X_holdout = X_tr[holdout_start:]
    y_holdout = y_tr[holdout_start:]

    print(f"  Final train: {len(X_train_final)} days  Holdout: {len(X_holdout)} days")

    dtrain = lgb.Dataset(X_train_final, label=y_train_final)
    lgb_params = {
        "objective": "regression", "metric": "rmse",
        "num_leaves": 31, "learning_rate": 0.03,
        "feature_fraction": 0.85, "bagging_fraction": 0.85,
        "bagging_freq": 3, "min_data_in_leaf": 20,
        "verbosity": -1,
    }
    lgb_model = lgb.train(lgb_params, dtrain, num_boost_round=250)

    dtrain_x = xgb.DMatrix(X_train_final, label=y_train_final)
    xgb_params = {
        "objective": "reg:squarederror", "eta": 0.03,
        "max_depth": 5, "subsample": 0.85, "colsample_bytree": 0.85,
        "min_child_weight": 10, "verbosity": 0,
    }
    xgb_model = xgb.train(xgb_params, dtrain_x, num_boost_round=250)

    # Holdout eval
    p_lgb_hold = lgb_model.predict(X_holdout)
    p_xgb_hold = xgb_model.predict(xgb.DMatrix(X_holdout))
    ensemble_hold = 0.5 * p_lgb_hold + 0.5 * p_xgb_hold
    good = ~np.isnan(y_holdout)
    holdout_ic = float(np.corrcoef(ensemble_hold[good], y_holdout[good])[0, 1])
    print(f"  Holdout IC (90 days): {holdout_ic:.4f}")

    # Full-sample preds for z-score calibration
    p_lgb_full = lgb_model.predict(X_all)
    p_xgb_full = xgb_model.predict(xgb.DMatrix(X_all))
    ensemble_full = 0.5 * p_lgb_full + 0.5 * p_xgb_full

    pred_std = float(np.nanstd(ensemble_full))
    pred_mean = float(np.nanmean(ensemble_full))
    print(f"  Pred stats: mean={pred_mean:.5f}  std={pred_std:.5f}")

    z_full = (ensemble_full - pred_mean) / max(pred_std, 1e-10)
    cross_175 = int((np.abs(z_full) > 1.75).sum())
    print(f"  Z-score: mean={z_full.mean():+.3f}  std={z_full.std():.3f}  "
          f"|z|>1.75: {cross_175}/{len(z_full)} ({cross_175/len(z_full)*100:.1f}%)")

    out_dir = Path(f"/quant_system/models_v8/{symbol}_1d")
    out_dir.mkdir(parents=True, exist_ok=True)

    _dump_bundle({"model": lgb_model, "features": feature_names},
                 out_dir / "lgbm_v8.pkl")
    _dump_bundle({"model": xgb_model, "features": feature_names},
                 out_dir / "xgb_v8.pkl")
    print(f"  Saved: {out_dir}/lgbm_v8.pkl + xgb_v8.pkl")

    with open(out_dir / "features.json", "w") as f:
        json.dump(feature_names, f, indent=2)

    cfg_in = PROD_CONFIG[symbol]
    cfg = {
        "version": "v8_1d", "symbol": symbol, "timeframe": "1d",
        "ensemble": True, "ensemble_weights": ENSEMBLE_WEIGHTS,
        "ensemble_method": "simple_mean",
        "models": ["lgbm_v8.pkl", "xgb_v8.pkl"],
        "features": feature_names,
        "horizon": 1, "horizon_hours": 24,
        "deadzone": cfg_in["deadzone"], "min_hold": cfg_in["min_hold"],
        "max_hold": cfg_in["max_hold"], "long_only": cfg_in["long_only"],
        "zscore_window": 180, "zscore_warmup": 90,
        "zscore_pred_mean": pred_mean, "zscore_pred_std": pred_std,
        "params": lgb_params, "xgb_params": xgb_params,
        "walk_forward": {
            "sharpe": cfg_in["wf_sharpe"],
            "ic_pearson": cfg_in["wf_ic_pearson"],
            "ic_spearman": cfg_in["wf_ic_spearman"],
            "trades": cfg_in["wf_trades"],
            "win_rate": cfg_in["wf_win_rate"],
            "max_dd_pct": cfg_in["wf_max_dd_pct"],
            "positive_months": cfg_in["positive_months"],
            "total_months": cfg_in["total_months"],
            "bootstrap_p_positive": cfg_in["bootstrap_p_positive"],
        },
        "holdout_ic_90d": holdout_ic,
        "training": {
            "train_days": len(X_train_final),
            "holdout_days": len(X_holdout),
            "total_days_available": len(X_tr),
            "feature_count": len(feature_names),
        },
        "passed": True,
        "override_reason": "1D POC Phase 1 verified: bootstrap P>99%, "
                           "positive months >60%, max DD <2%",
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"  Saved: {out_dir}/config.json")

    warmup_n = 180
    with open(out_dir / "zscore_warmup.json", "w") as f:
        json.dump({
            "predictions": ensemble_full[-warmup_n:].tolist(),
            "dates": [str(d) for d in df_1d["date"].values[-warmup_n:]],
            "pred_mean": pred_mean,
            "pred_std": pred_std,
            "window": 180,
        }, f, indent=2)
    print(f"  Saved: {out_dir}/zscore_warmup.json ({warmup_n} bars)")

    return {
        "symbol": symbol, "holdout_ic": holdout_ic,
        "features": len(feature_names),
        "train_days": len(X_train_final),
        "pred_std": pred_std,
    }


def main():
    macro = pd.read_csv("/quant_system/data_files/cross_market_daily.csv")
    macro["date"] = pd.to_datetime(macro["date"]).dt.date

    results = []
    for sym in ("BTCUSDT", "ETHUSDT"):
        r = train_symbol(sym, macro)
        results.append(r)
        print()

    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    for r in results:
        print(f"  {r['symbol']:>10s}_1d  holdout_IC={r['holdout_ic']:+.4f}  "
              f"features={r['features']}  pred_std={r['pred_std']:.5f}")


if __name__ == "__main__":
    main()
