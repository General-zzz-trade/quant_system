#!/usr/bin/env python3
"""Backtest with Adaptive Config — simulates live parameter adaptation.

Compares three strategies:
1. Fixed config (current production params)
2. Oracle: per-period best config (walk-forward upper bound, not achievable live)
3. Adaptive: select params from recent N months, apply to next period

The adaptive strategy answers: "If we re-optimized params monthly using
recent data, how would we perform vs fixed and oracle?"

Usage:
    python3 -m scripts.backtest_adaptive --symbol ETHUSDT
    python3 -m scripts.backtest_adaptive --symbol BTCUSDT,ETHUSDT
"""
from __future__ import annotations

import sys
import time
import json
import pickle
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/quant_system")

from features.batch_feature_engine import compute_features_batch
from features.multi_timeframe import compute_4h_features, TF4H_FEATURE_NAMES
from scripts.signal_postprocess import rolling_zscore
from scripts.train_v7_alpha import INTERACTION_FEATURES
from alpha.adaptive_config import AdaptiveConfigSelector, _fast_backtest

COST_BPS_RT = 4
BARS_PER_DAY = 24
BARS_PER_MONTH = BARS_PER_DAY * 30


def load_model_and_predict(symbol: str, df: pd.DataFrame, feat_df: pd.DataFrame):
    """Load model and compute predictions + z-scores for all bars."""
    model_dir = Path(f"models_v8/{symbol}_gate_v2")
    with open(model_dir / "config.json") as f:
        config = json.load(f)

    horizon_models = []
    for hcfg in config.get("horizon_models", []):
        with open(model_dir / hcfg["lgbm"], "rb") as f:
            lgbm_data = pickle.load(f)
        xgb_model = None
        xgb_path = model_dir / hcfg["xgb"]
        if xgb_path.exists():
            with open(xgb_path, "rb") as f:
                xgb_model = pickle.load(f)["model"]
        horizon_models.append({
            "horizon": hcfg["horizon"],
            "lgbm": lgbm_data["model"],
            "xgb": xgb_model,
            "features": lgbm_data["features"],
        })

    n = len(df)
    list(feat_df.columns)

    # Predict per horizon, z-score, average
    preds_by_h = {}
    for hm in horizon_models:
        preds = np.zeros(n)
        for i in range(n):
            x = np.zeros((1, len(hm["features"])))
            for j, fname in enumerate(hm["features"]):
                val = feat_df.iloc[i].get(fname, 0.0)
                x[0, j] = float(val) if isinstance(val, (int, float, np.floating)) else 0.0
            pred = float(hm["lgbm"].predict(x)[0])
            if hm["xgb"] is not None:
                try:
                    import xgboost as xgb
                    xgb_pred = float(hm["xgb"].predict(xgb.DMatrix(x))[0])
                    pred = 0.5 * pred + 0.5 * xgb_pred
                except Exception:
                    pass
            preds[i] = pred
        preds_by_h[hm["horizon"]] = preds

    # Z-score per horizon, then average
    z_all = []
    for h, pred in sorted(preds_by_h.items()):
        z_all.append(rolling_zscore(pred))
    z = np.mean(z_all, axis=0)

    return config, z


def run_adaptive_backtest(symbol: str):
    """Compare fixed vs adaptive vs oracle on rolling periods."""
    data_path = Path(f"data_files/{symbol}_1h.csv")
    if not data_path.exists():
        print(f"  ERROR: {data_path} not found")
        return

    df = pd.read_csv(data_path)
    n = len(df)
    closes = df["close"].values.astype(np.float64)
    ts_col = "open_time" if "open_time" in df.columns else "timestamp"
    timestamps = df[ts_col].values.astype(np.int64)

    print(f"\n  Data: {n:,} 1h bars")
    print("  Computing features...", end=" ", flush=True)
    t0 = time.time()

    _has_v11 = Path("data_files/macro_daily.csv").exists()
    feat_df = compute_features_batch(symbol, df, include_v11=_has_v11)
    tf4h = compute_4h_features(df)
    for col in TF4H_FEATURE_NAMES:
        feat_df[col] = tf4h[col].values
    for int_name, fa, fb in INTERACTION_FEATURES:
        if fa in feat_df.columns and fb in feat_df.columns:
            feat_df[int_name] = feat_df[fa].astype(float) * feat_df[fb].astype(float)
    print(f"done ({time.time()-t0:.1f}s)")

    print("  Computing predictions...", end=" ", flush=True)
    t0 = time.time()
    config, z = load_model_and_predict(symbol, df, feat_df)
    print(f"done ({time.time()-t0:.1f}s)")

    # Fixed config from model
    fixed_dz = config.get("deadzone", 0.3)
    fixed_mh = config.get("min_hold", 12)
    fixed_maxh = config.get("max_hold", 96)
    fixed_lo = config.get("long_only", False)

    # Rolling periods: 3-month test windows
    test_months = 3
    test_bars = BARS_PER_MONTH * test_months
    lookback_months = 6
    lookback_bars = BARS_PER_MONTH * lookback_months
    warmup_bars = BARS_PER_MONTH * 12  # need 12 months before first adaptive selection

    selector = AdaptiveConfigSelector(lookback_months=lookback_months, min_trades=8)
    selector_robust = AdaptiveConfigSelector(lookback_months=lookback_months, min_trades=8)

    print(f"\n  {'Period':>20} {'Fixed':>8} {'Adaptive':>10} {'Robust':>8} {'Oracle':>8} "
          f"{'Adp DZ':>7} {'Adp Hold':>9} {'Adp LO':>7} {'Conf':>6}")
    print(f"  {'-'*20:>20} {'-'*8:>8} {'-'*10:>10} {'-'*8:>8} {'-'*8:>8} "
          f"{'-'*7:>7} {'-'*9:>9} {'-'*7:>7} {'-'*6:>6}")

    fixed_sharpes = []
    adaptive_sharpes = []
    robust_sharpes = []
    oracle_sharpes = []
    adaptive_rets = []
    fixed_rets = []
    robust_rets = []

    period_start = warmup_bars
    while period_start + test_bars <= n:
        period_end = period_start + test_bars
        start_d = pd.Timestamp(timestamps[period_start], unit="ms").strftime("%Y-%m")
        end_d = pd.Timestamp(timestamps[period_end - 1], unit="ms").strftime("%Y-%m")

        z_test = z[period_start:period_end]
        c_test = closes[period_start:period_end]

        # 1. Fixed
        r_fixed = _fast_backtest(z_test, c_test, fixed_dz, fixed_mh, fixed_maxh, fixed_lo)

        # 2. Adaptive: select from lookback, apply to test
        lb_start = max(0, period_start - lookback_bars)
        z_lookback = z[lb_start:period_start]
        c_lookback = closes[lb_start:period_start]
        adp = selector.select(z_lookback, c_lookback)
        r_adaptive = _fast_backtest(z_test, c_test, adp.deadzone, adp.min_hold, adp.max_hold, adp.long_only)

        # 3. Robust adaptive (multi-window)
        z_for_robust = z[:period_start]
        c_for_robust = closes[:period_start]
        rob = selector_robust.select_robust(z_for_robust, c_for_robust)
        r_robust = _fast_backtest(z_test, c_test, rob.deadzone, rob.min_hold, rob.max_hold, rob.long_only)

        # 4. Oracle: best config on test period (hindsight)
        best_oracle = -999
        for dz in [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5]:
            for mh in [8, 12, 24]:
                for mult in [5, 8]:
                    maxh = mh * mult
                    for lo in [True, False]:
                        r = _fast_backtest(z_test, c_test, dz, mh, maxh, lo)
                        if r["sharpe"] > best_oracle and r["trades"] >= 5:
                            best_oracle = r["sharpe"]
                            r_oracle = r
        if best_oracle <= -999:
            r_oracle = {"sharpe": 0, "trades": 0, "return": 0}

        fixed_sharpes.append(r_fixed["sharpe"])
        adaptive_sharpes.append(r_adaptive["sharpe"])
        robust_sharpes.append(r_robust["sharpe"])
        oracle_sharpes.append(r_oracle["sharpe"])
        fixed_rets.append(r_fixed["return"])
        adaptive_rets.append(r_adaptive["return"])
        robust_rets.append(r_robust["return"])

        # Mark winner
        best_live = max(r_fixed["sharpe"], r_adaptive["sharpe"], r_robust["sharpe"])
        f_mark = " *" if r_fixed["sharpe"] == best_live else ""
        a_mark = " *" if r_adaptive["sharpe"] == best_live else ""
        rb_mark = " *" if r_robust["sharpe"] == best_live else ""

        print(
            f"  {start_d+'→'+end_d:>20} "
            f"{r_fixed['sharpe']:>7.2f}{f_mark:1s} "
            f"{r_adaptive['sharpe']:>9.2f}{a_mark:1s} "
            f"{r_robust['sharpe']:>7.2f}{rb_mark:1s} "
            f"{r_oracle['sharpe']:>8.2f} "
            f"{adp.deadzone:>7.1f} "
            f"[{adp.min_hold},{adp.max_hold}]{'':>2} "
            f"{'Y' if adp.long_only else 'N':>4} "
            f"{adp.confidence:>6}"
        )

        period_start = period_end

    # Summary
    n_periods = len(fixed_sharpes)
    if n_periods == 0:
        print("  No periods evaluated!")
        return

    print(f"\n  {'='*75}")
    print(f"  ADAPTIVE CONFIG SUMMARY — {symbol} ({n_periods} periods)")
    print(f"  {'='*75}")

    print(f"\n  {'':>20} {'Fixed':>10} {'Adaptive':>10} {'Robust':>10} {'Oracle':>10}")
    print(f"  {'-'*20:>20} {'-'*10:>10} {'-'*10:>10} {'-'*10:>10} {'-'*10:>10}")
    print(f"  {'Mean Sharpe':>20} {np.mean(fixed_sharpes):>10.2f} {np.mean(adaptive_sharpes):>10.2f} {np.mean(robust_sharpes):>10.2f} {np.mean(oracle_sharpes):>10.2f}")
    print(f"  {'Median Sharpe':>20} {np.median(fixed_sharpes):>10.2f} {np.median(adaptive_sharpes):>10.2f} {np.median(robust_sharpes):>10.2f} {np.median(oracle_sharpes):>10.2f}")
    print(f"  {'Min Sharpe':>20} {np.min(fixed_sharpes):>10.2f} {np.min(adaptive_sharpes):>10.2f} {np.min(robust_sharpes):>10.2f} {np.min(oracle_sharpes):>10.2f}")
    print(f"  {'Sharpe > 0 (%)':>20} {np.mean([s>0 for s in fixed_sharpes])*100:>9.0f}% {np.mean([s>0 for s in adaptive_sharpes])*100:>9.0f}% {np.mean([s>0 for s in robust_sharpes])*100:>9.0f}% {np.mean([s>0 for s in oracle_sharpes])*100:>9.0f}%")
    print(f"  {'Total Return':>20} {sum(fixed_rets):>+9.1f}% {sum(adaptive_rets):>+9.1f}% {sum(robust_rets):>+9.1f}%")

    # Win counts
    adp_wins = sum(1 for a, f in zip(adaptive_sharpes, fixed_sharpes) if a > f)
    rob_wins = sum(1 for r, f in zip(robust_sharpes, fixed_sharpes) if r > f)
    print(f"\n  Adaptive beats Fixed: {adp_wins}/{n_periods} ({adp_wins/n_periods*100:.0f}%)")
    print(f"  Robust beats Fixed:   {rob_wins}/{n_periods} ({rob_wins/n_periods*100:.0f}%)")

    # Config stability from adaptive selector
    if selector.history:
        dzs = [h.deadzone for h in selector.history]
        mhs = [h.min_hold for h in selector.history]
        los = [h.long_only for h in selector.history]
        print("\n  Adaptive config trajectory:")
        print(f"    Deadzone: {dzs}")
        print(f"    Min hold: {mhs}")
        print(f"    Long only: {[int(l) for l in los]}")


def main():
    parser = argparse.ArgumentParser(description="Adaptive Config Backtest")
    parser.add_argument("--symbol", default="ETHUSDT",
                        help="Comma-separated symbols")
    args = parser.parse_args()
    symbols = [s.strip().upper() for s in args.symbol.split(",")]

    print("=" * 75)
    print("  ADAPTIVE CONFIG BACKTEST")
    print("=" * 75)

    for symbol in symbols:
        print(f"\n{'='*75}")
        print(f"  {symbol}")
        print(f"{'='*75}")
        run_adaptive_backtest(symbol)

    print(f"\n{'='*75}")
    print("  DONE")
    print(f"{'='*75}")


if __name__ == "__main__":
    main()
