#!/usr/bin/env python3
"""Evaluate LightGBM vs XGBoost ensemble members per symbol."""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

from infra.model_signing import load_verified_pickle
from features.batch_feature_engine import compute_features_batch
from alpha.signal_transform import pred_to_signal as _pred_to_signal


def evaluate_model(symbol, model_idx, model_name, horizon):
    model_dir = Path(f"models_v8/{symbol}_gate_v2")
    with open(model_dir / "config.json") as f:
        cfg = json.load(f)

    feature_names = cfg["features"]
    target_mode = cfg.get("target_mode", "clipped")
    deadzone = cfg.get("deadzone", 0.5)
    min_hold = cfg.get("min_hold", 8)

    # Uses HMAC-verified pickle loading (infra.model_signing)
    pkl_path = model_dir / cfg["models"][model_idx]
    data = load_verified_pickle(pkl_path)
    model = data["model"]

    df = pd.read_csv(f"data_files/{symbol}_1h.csv")
    v11_features = {"spx_overnight_ret", "dxy_change_5d", "vix_zscore_14",
                    "mempool_size_zscore_24", "fee_urgency_ratio",
                    "exchange_supply_zscore_30", "liquidation_cascade_score"}
    needs_v11 = bool(set(feature_names) & v11_features)
    feat_df = compute_features_batch(symbol, df, include_v11=needs_v11)

    if symbol != "BTCUSDT":
        cross_features = {"btc_ret_24", "btc_rsi_14", "btc_mean_reversion_20",
                         "btc_ret_12", "btc_macd_line", "btc_atr_norm_14", "btc_bb_width_20"}
        if set(feature_names) & cross_features:
            from features.batch_cross_asset import build_cross_features_batch
            cross_map = build_cross_features_batch([symbol])
            if cross_map and symbol in cross_map:
                ts_col = "timestamp" if "timestamp" in df.columns else "open_time"
                oos_ts = df[ts_col].values.astype(np.int64)
                cross_aligned = cross_map[symbol].reindex(oos_ts)
                for cname in cross_aligned.columns:
                    if cname in feature_names:
                        feat_df[cname] = cross_aligned[cname].values

    for fname in feature_names:
        if fname not in feat_df.columns:
            feat_df[fname] = np.nan

    closes = feat_df["close"].values.astype(np.float64)
    X = feat_df[feature_names].values.astype(np.float64)

    warmup = 65
    X = X[warmup:]
    closes = closes[warmup:]
    n = len(X)

    import xgboost as xgb
    if isinstance(model, xgb.core.Booster):
        dm = xgb.DMatrix(X)
        y_pred = model.predict(dm)
    else:
        y_pred = model.predict(X)

    # Forward return
    fwd_ret = np.full(n, np.nan)
    fwd_ret[:-horizon] = (closes[horizon:] - closes[:-horizon]) / closes[:-horizon]

    valid = ~(np.isnan(y_pred) | np.isnan(fwd_ret))
    ic, _ = spearmanr(y_pred[valid], fwd_ret[valid])

    # Signal + Sharpe
    signal = _pred_to_signal(y_pred, target_mode=target_mode,
                             deadzone=deadzone, min_hold=min_hold)
    signal = np.clip(signal, 0.0, None)  # long-only for fair comparison

    ret_1bar = np.zeros(n)
    ret_1bar[1:] = (closes[1:] - closes[:-1]) / closes[:-1]
    pnl = signal[:-1] * ret_1bar[1:] - np.abs(np.diff(signal)) * 6e-4
    active = pnl != 0
    if active.sum() > 10:
        sharpe = np.mean(pnl[active]) / np.std(pnl[active], ddof=1) * np.sqrt(8760)
        total_ret = float(np.sum(pnl))
    else:
        sharpe = 0.0
        total_ret = 0.0

    activity = float(np.mean(signal != 0) * 100)

    return {
        "model": model_name,
        "ic": ic,
        "sharpe": sharpe,
        "return": total_ret,
        "activity": activity,
        "y_pred": y_pred,
    }


def main():
    print(f"{'Symbol':<10} {'Model':<8} {'IC':>8} {'Sharpe':>8} {'Return':>10} {'Active%':>8}")
    print("-" * 60)

    for symbol in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
        horizon = 24 if symbol != "SOLUSDT" else 5
        results = []
        for idx, name in [(0, "LightGBM"), (1, "XGBoost")]:
            r = evaluate_model(symbol, idx, name, horizon)
            results.append(r)
            print(f"{symbol:<10} {name:<8} {r['ic']:>+8.4f} {r['sharpe']:>8.2f} "
                  f"{r['return']*100:>+9.1f}% {r['activity']:>7.1f}%")

        # Prediction correlation
        p1, p2 = results[0]["y_pred"], results[1]["y_pred"]
        min_len = min(len(p1), len(p2))
        corr = float(np.corrcoef(p1[:min_len], p2[:min_len])[0, 1])
        print(f"{'':10} {'Corr':<8} {corr:>+8.4f}")

        # Test ensemble weights
        print("\n  Weight sweep (LGB:XGB):")
        print(f"  {'Weights':<12} {'IC':>8} {'Sharpe':>8} {'Return':>10}")
        print(f"  {'-'*42}")

        cfg_path = Path(f"models_v8/{symbol}_gate_v2/config.json")
        with open(cfg_path) as f:
            cfg = json.load(f)
        target_mode = cfg.get("target_mode", "clipped")
        deadzone = cfg.get("deadzone", 0.5)
        min_hold = cfg.get("min_hold", 8)

        p1_full, p2_full = results[0]["y_pred"], results[1]["y_pred"]
        min_n = min(len(p1_full), len(p2_full))
        p1_t, p2_t = p1_full[:min_n], p2_full[:min_n]

        df = pd.read_csv(f"data_files/{symbol}_1h.csv")
        feat_df = compute_features_batch(symbol, df, include_v11=bool(
            set(cfg["features"]) & {"spx_overnight_ret", "mempool_size_zscore_24"}))
        closes = feat_df["close"].values.astype(np.float64)[65:65+min_n]

        fwd_ret = np.full(min_n, np.nan)
        fwd_ret[:-horizon] = (closes[horizon:] - closes[:-horizon]) / closes[:-horizon]

        ret_1bar = np.zeros(min_n)
        ret_1bar[1:] = (closes[1:] - closes[:-1]) / closes[:-1]

        for lgb_w in [1.0, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.0]:
            xgb_w = 1.0 - lgb_w
            blended = lgb_w * p1_t + xgb_w * p2_t

            valid = ~(np.isnan(blended) | np.isnan(fwd_ret))
            ic_b, _ = spearmanr(blended[valid], fwd_ret[valid])

            signal = _pred_to_signal(blended, target_mode=target_mode,
                                     deadzone=deadzone, min_hold=min_hold)
            signal = np.clip(signal, 0.0, None)

            pnl = signal[:-1] * ret_1bar[1:] - np.abs(np.diff(signal)) * 6e-4
            active = pnl != 0
            if active.sum() > 10:
                sh = np.mean(pnl[active]) / np.std(pnl[active], ddof=1) * np.sqrt(8760)
                ret = float(np.sum(pnl))
            else:
                sh = 0.0
                ret = 0.0

            marker = " <-- current" if lgb_w == 0.5 else ""
            print(f"  {lgb_w:.1f}:{xgb_w:.1f}      {ic_b:>+8.4f} {sh:>8.2f} {ret*100:>+9.1f}%{marker}")

        print()


if __name__ == "__main__":
    main()
