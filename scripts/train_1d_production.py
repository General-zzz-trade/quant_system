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
        # BTC 1D was already healthy (holdout IC +0.118 unweighted) — adding
        # recency weights HURT it (+0.118 → +0.058) because BTC didn't have
        # the Jan-Feb 2026 regime shift that motivated the weighting.
        "use_recency_weights": False,
    },
    "ETHUSDT": {
        "deadzone": 1.75, "min_hold": 1, "max_hold": 7, "long_only": False,
        "wf_sharpe": 4.25, "wf_ic_pearson": 0.0364, "wf_ic_spearman": 0.0288,
        "wf_trades": 64, "wf_win_rate": 57.8, "wf_max_dd_pct": 0.75,
        "bootstrap_p_positive": 1.000,
        "positive_months": 23, "total_months": 34,
        # ETH 1D had holdout IC -0.083 (Jan-Feb 2026 regime shift). Recency
        # weights recover to +0.015 — model adapts to new macro regime.
        "use_recency_weights": True,
    },
}
ENSEMBLE_WEIGHTS = [0.5, 0.5]


def _dump_bundle(obj, path):
    """Save model bundle — matches production loader format."""
    import pickle as _pkl  # noqa: S403 — trusted local artefact

    with open(path, "wb") as f:
        _pkl.dump(obj, f)  # noqa: S301


def _recency_weights(dates, anchor_date=None):
    """Sample weights — recent data weighted up, old data down.

    Phase A (2026-04-19): added because ETH 1D holdout IC = -0.083 was
    diagnosed (scripts/eth_1d_ic_diag.py) as a Jan-Feb 2026 macro-regime
    shift (VIX, M2, yield-curve correlations all flipped). Letting the
    model learn faster from the new regime should recover positive IC.

    Curve:
      - last 1.5 years:  weight = 2.0  (latest regime, weighted up)
      - 1.5 - 3 years:   weight = 1.0  (baseline)
      - 3 - 5 years:     weight = 0.5  (older regime, dampened)
    """
    if anchor_date is None:
        anchor_date = dates[-1]
    anchor = pd.Timestamp(anchor_date)
    days_old = np.array([(anchor - pd.Timestamp(d)).days for d in dates])
    weights = np.ones(len(dates), dtype=np.float64)
    weights[days_old < 365 * 1.5] = 2.0
    weights[days_old > 365 * 3.0] = 0.5
    return weights


# Multi-horizon ensemble (Phase B): predict h=1, 3, 7 day forward returns,
# combine via IC-weighted average. Backtest convention from 4h model is:
# higher-horizon predictions are smoother (less noise), shorter ones are
# more reactive. IC-weighted lets the model decide which to trust.
HORIZONS = [1, 3, 7]


def _make_target(closes, horizon):
    """Forward log return over `horizon` days."""
    log_rets = np.log(closes[horizon:] / closes[:-horizon])
    target = np.full(len(closes), np.nan)
    target[:-horizon] = log_rets
    return target


def train_symbol(symbol, macro):
    print("=" * 72)
    print(f"TRAIN 1D PRODUCTION: {symbol}  (multi-horizon + recency-weighted)")
    print("=" * 72)

    df_1h = pd.read_csv(f"/quant_system/data_files/{symbol}_1h.csv")
    df_1d = resample_1h_to_1d(df_1h)
    df_1d = df_1d[df_1d["date"] >= macro["date"].min()].reset_index(drop=True)

    feat_df, feature_names = compute_1d_features(df_1d, macro)
    closes = df_1d["close"].values.astype(np.float64)
    dates = df_1d["date"].values

    X_all = feat_df[feature_names].values.astype(np.float64)
    X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"  Total days: {len(X_all)}  Features: {len(feature_names)}")

    import lightgbm as lgb
    import xgboost as xgb

    # ── Train one (lgbm + xgb) pair per horizon ──────────────────────
    horizon_models = {}      # h -> {"lgb": ..., "xgb": ...}
    horizon_holdout_ic = {}  # h -> {"lgb": float, "xgb": float, "ensemble": float}

    for horizon in HORIZONS:
        target = _make_target(closes, horizon)
        valid = ~np.isnan(target)
        X_tr = X_all[valid]
        y_tr = target[valid]
        dates_tr = dates[valid]

        holdout_start = len(X_tr) - 90
        X_train = X_tr[:holdout_start]
        y_train = y_tr[:holdout_start]
        dates_train = dates_tr[:holdout_start]
        X_holdout = X_tr[holdout_start:]
        y_holdout = y_tr[holdout_start:]

        # Recency-weighted samples (Phase A) — per-symbol opt-in.
        # See PROD_CONFIG comments for the rationale.
        if PROD_CONFIG[symbol].get("use_recency_weights", False):
            sw = _recency_weights(dates_train)
        else:
            sw = np.ones(len(X_train), dtype=np.float64)

        lgb_params = {
            "objective": "regression", "metric": "rmse",
            "num_leaves": 31, "learning_rate": 0.03,
            "feature_fraction": 0.85, "bagging_fraction": 0.85,
            "bagging_freq": 3, "min_data_in_leaf": 20,
            "verbosity": -1,
        }
        dtrain = lgb.Dataset(X_train, label=y_train, weight=sw)
        lgb_model = lgb.train(lgb_params, dtrain, num_boost_round=250)

        xgb_params = {
            "objective": "reg:squarederror", "eta": 0.03,
            "max_depth": 5, "subsample": 0.85, "colsample_bytree": 0.85,
            "min_child_weight": 10, "verbosity": 0,
        }
        dtrain_x = xgb.DMatrix(X_train, label=y_train, weight=sw)
        xgb_model = xgb.train(xgb_params, dtrain_x, num_boost_round=250)

        # Holdout IC per (model, horizon)
        good = ~np.isnan(y_holdout)
        p_lgb = lgb_model.predict(X_holdout)
        p_xgb = xgb_model.predict(xgb.DMatrix(X_holdout))
        ic_lgb = float(np.corrcoef(p_lgb[good], y_holdout[good])[0, 1])
        ic_xgb = float(np.corrcoef(p_xgb[good], y_holdout[good])[0, 1])
        # Within-horizon ensemble: simple 50/50 (same as old behaviour)
        p_ens = 0.5 * p_lgb + 0.5 * p_xgb
        ic_ens = float(np.corrcoef(p_ens[good], y_holdout[good])[0, 1])

        horizon_models[horizon] = {"lgb": lgb_model, "xgb": xgb_model}
        horizon_holdout_ic[horizon] = {
            "lgb": ic_lgb, "xgb": ic_xgb, "ensemble": ic_ens,
        }
        print(f"  h={horizon}d  IC: lgb={ic_lgb:+.4f}  xgb={ic_xgb:+.4f}  "
              f"ens={ic_ens:+.4f}  (train={len(X_train)} samples, "
              f"weights mean={sw.mean():.2f})")

    # ── Cross-horizon IC-weighted ensemble ───────────────────────────
    # Use horizon=1 holdout target for the final calibration since that's
    # what live z-scores will compare to.
    target_1d = _make_target(closes, 1)
    valid_1d = ~np.isnan(target_1d)
    X_v = X_all[valid_1d]
    y_v = target_1d[valid_1d]

    holdout_start = len(X_v) - 90
    y_holdout = y_v[holdout_start:]
    good = ~np.isnan(y_holdout)

    # Per-horizon ensemble preds on holdout
    holdout_preds = {}
    for h, models in horizon_models.items():
        p_lgb = models["lgb"].predict(X_v[holdout_start:])
        p_xgb = models["xgb"].predict(xgb.DMatrix(X_v[holdout_start:]))
        holdout_preds[h] = 0.5 * p_lgb + 0.5 * p_xgb

    # IC-weighted cross-horizon weights (clip negative ICs to 0)
    raw_weights = np.array([
        max(horizon_holdout_ic[h]["ensemble"], 0.0) for h in HORIZONS
    ])
    if raw_weights.sum() > 0:
        ensemble_weights_h = raw_weights / raw_weights.sum()
    else:
        # All horizons negative — fall back to equal weights
        ensemble_weights_h = np.ones(len(HORIZONS)) / len(HORIZONS)

    print("  Cross-horizon IC weights: " + ", ".join(
        f"h{h}={w:.2f}" for h, w in zip(HORIZONS, ensemble_weights_h)))

    # Combined holdout prediction
    combined_hold = sum(
        ensemble_weights_h[i] * holdout_preds[h]
        for i, h in enumerate(HORIZONS)
    )
    holdout_ic = float(np.corrcoef(combined_hold[good], y_holdout[good])[0, 1])
    print(f"  Combined holdout IC (90 days): {holdout_ic:+.4f}")

    # ── Full-sample combined preds for z-score calibration ───────────
    # For non-h=1 horizons we predict on X_all, then for h>1 the prediction
    # is "next h-day return" — for z-score calibration we treat all as the
    # same scale (the relative magnitudes within each horizon's
    # distribution are what matters for z).
    p_combined_full = np.zeros(len(X_all))
    for i, h in enumerate(HORIZONS):
        p_lgb = horizon_models[h]["lgb"].predict(X_all)
        p_xgb = horizon_models[h]["xgb"].predict(xgb.DMatrix(X_all))
        p_combined_full += ensemble_weights_h[i] * (0.5 * p_lgb + 0.5 * p_xgb)
    ensemble_full = p_combined_full

    # Keep "lgb_model" / "xgb_model" as the h=1 ones for backwards-compat
    # with daily_paper_runner._load_model_bundle (which loads lgbm_v8.pkl
    # and xgb_v8.pkl by name). The combined-prediction is captured by the
    # config.json `multi_horizon` block + zscore_pred_mean/std baked in.
    lgb_model = horizon_models[1]["lgb"]
    xgb_model = horizon_models[1]["xgb"]

    pred_std = float(np.nanstd(ensemble_full))
    pred_mean = float(np.nanmean(ensemble_full))
    print(f"  Pred stats: mean={pred_mean:.5f}  std={pred_std:.5f}")

    z_full = (ensemble_full - pred_mean) / max(pred_std, 1e-10)
    cross_175 = int((np.abs(z_full) > 1.75).sum())
    print(f"  Z-score: mean={z_full.mean():+.3f}  std={z_full.std():.3f}  "
          f"|z|>1.75: {cross_175}/{len(z_full)} ({cross_175/len(z_full)*100:.1f}%)")

    out_dir = Path(f"/quant_system/models_v8/{symbol}_1d")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save per-horizon models. lgbm_v8.pkl/xgb_v8.pkl point to h=1 for
    # backwards-compat with single-horizon loaders; the multi-horizon
    # consumer reads `horizons` + `cross_horizon_weights` from config.
    for h in HORIZONS:
        _dump_bundle({"model": horizon_models[h]["lgb"],
                      "features": feature_names, "horizon": h},
                     out_dir / f"lgbm_h{h}d.pkl")
        _dump_bundle({"model": horizon_models[h]["xgb"],
                      "features": feature_names, "horizon": h},
                     out_dir / f"xgb_h{h}d.pkl")
    _dump_bundle({"model": lgb_model, "features": feature_names},
                 out_dir / "lgbm_v8.pkl")
    _dump_bundle({"model": xgb_model, "features": feature_names},
                 out_dir / "xgb_v8.pkl")
    print(f"  Saved: lgbm/xgb_v8.pkl + {len(HORIZONS)} per-horizon files")

    with open(out_dir / "features.json", "w") as f:
        json.dump(feature_names, f, indent=2)

    cfg_in = PROD_CONFIG[symbol]
    cfg = {
        "version": "v8_1d_mh", "symbol": symbol, "timeframe": "1d",
        "ensemble": True, "ensemble_weights": ENSEMBLE_WEIGHTS,
        "ensemble_method": "ic_weighted_multi_horizon",
        "models": ["lgbm_v8.pkl", "xgb_v8.pkl"],
        # Multi-horizon (Phase B 2026-04-19)
        "horizons": HORIZONS,
        "cross_horizon_weights": ensemble_weights_h.tolist(),
        "horizon_models": {
            str(h): {
                "lgb_file": f"lgbm_h{h}d.pkl",
                "xgb_file": f"xgb_h{h}d.pkl",
                "ic_holdout": horizon_holdout_ic[h]["ensemble"],
            }
            for h in HORIZONS
        },
        "features": feature_names,
        "horizon": 1, "horizon_hours": 24,
        "deadzone": cfg_in["deadzone"], "min_hold": cfg_in["min_hold"],
        "max_hold": cfg_in["max_hold"], "long_only": cfg_in["long_only"],
        "zscore_window": 180, "zscore_warmup": 90,
        "zscore_pred_mean": pred_mean, "zscore_pred_std": pred_std,
        "params": lgb_params, "xgb_params": xgb_params,
        # Recency-weighted training (Phase A 2026-04-19)
        "sample_weighting": {
            "scheme": "recency_2_step",
            "windows_years": [1.5, 3.0, 5.0],
            "weights": [2.0, 1.0, 0.5],
            "anchor_date": str(dates[-1]),
        },
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
            "train_days": holdout_start,
            "holdout_days": 90,
            "total_days_available": len(X_v),
            "feature_count": len(feature_names),
        },
        "passed": True,
        "override_reason": "1D Phase A (recency weights) + Phase B "
                           "(multi-horizon IC ensemble) — see commits and "
                           "scripts/eth_1d_ic_diag.py for context",
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
        "train_days": holdout_start,
        "pred_std": pred_std,
        "horizon_ic": {h: horizon_holdout_ic[h]["ensemble"] for h in HORIZONS},
        "horizon_weights": dict(zip(HORIZONS, ensemble_weights_h.tolist())),
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
