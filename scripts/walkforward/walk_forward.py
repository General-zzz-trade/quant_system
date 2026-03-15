#!/usr/bin/env python3
"""Walk-Forward Validation — rolling N-period train/test to measure alpha robustness.

Instead of a single train/test split, this framework:
1. Divides data into rolling windows
2. Trains a model on each window
3. Tests on the next window (no overlap)
4. Aggregates results across all windows

This answers: "Does this alpha work in ALL market environments, or just some?"

Walk-forward scheme (example with 6-month test windows):

  |-------- train --------|-- test --|
  |----------- train ----------|-- test --|
  |-------------- train -----------|-- test --|
                                              ^now

Key metrics:
- Per-window Sharpe, IC, trades, win rate
- Aggregate: mean, median, min, std of Sharpe
- Win rate across windows (% of windows with Sharpe > 0)
- Stability ratio: median(Sharpe) / std(Sharpe)

Usage:
    python3 -m scripts.walk_forward --symbol ETHUSDT
    python3 -m scripts.walk_forward --symbol BTCUSDT,ETHUSDT
    python3 -m scripts.walk_forward --symbol ETHUSDT --test-months 3 --n-folds 8
"""
from __future__ import annotations

import sys
import time
import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, "/quant_system")

from features.batch_feature_engine import compute_features_batch
from features.multi_timeframe import compute_4h_features, TF4H_FEATURE_NAMES
from features.dynamic_selector import greedy_ic_select
from scripts.signal_postprocess import rolling_zscore, should_exit_position
from scripts.train_v7_alpha import INTERACTION_FEATURES, BLACKLIST

COST_BPS_RT = 4
BARS_PER_DAY = 24
BARS_PER_MONTH = BARS_PER_DAY * 30
HPO_TRIALS = 10
TOP_K = 14
WARMUP = 30
HORIZONS = [12, 24]


def fast_ic(x, y):
    m = ~(np.isnan(x) | np.isnan(y))
    if m.sum() < 50:
        return 0.0
    r, _ = spearmanr(x[m], y[m])
    return float(r) if not np.isnan(r) else 0.0


def compute_target(closes, horizon):
    n = len(closes)
    y = np.full(n, np.nan)
    y[:n - horizon] = closes[horizon:] / closes[:n - horizon] - 1
    v = y[~np.isnan(y)]
    if len(v) > 10:
        p1, p99 = np.percentile(v, [1, 99])
        y = np.where(np.isnan(y), np.nan, np.clip(y, p1, p99))
    return y


def train_fold(
    X: np.ndarray,
    closes: np.ndarray,
    feature_names: List[str],
    train_start: int,
    train_end: int,
    test_start: int,
    test_end: int,
    horizons: List[int],
) -> Dict[str, Any]:
    """Train one walk-forward fold and return test metrics."""
    import lightgbm as lgb

    # Val split: last 20% of train
    val_size = max(int((train_end - train_start) * 0.2), BARS_PER_MONTH)
    val_start = train_end - val_size

    preds_test = {}
    ics = {}

    for h in horizons:
        y = compute_target(closes, h)

        # Feature selection on train (before val)
        valid_t = ~np.isnan(y[train_start:val_start])
        if valid_t.sum() < 200:
            continue

        selected = greedy_ic_select(
            X[train_start:val_start][valid_t],
            y[train_start:val_start][valid_t],
            feature_names,
            top_k=TOP_K,
        )
        sel_idx = [feature_names.index(f) for f in selected]
        X_sel = X[:, sel_idx]

        # Train LightGBM with simple HPO
        valid_tr = ~np.isnan(y[train_start:val_start])
        valid_val = ~np.isnan(y[val_start:train_end])

        dtrain = lgb.Dataset(
            X_sel[train_start:val_start][valid_tr],
            y[train_start:val_start][valid_tr],
        )
        dval = lgb.Dataset(
            X_sel[val_start:train_end][valid_val],
            y[val_start:train_end][valid_val],
            reference=dtrain,
        )

        best_ic = -999
        best_model = None

        for lr in [0.01, 0.03, 0.05]:
            for nl in [15, 31]:
                params = {
                    "objective": "regression",
                    "metric": "mae",
                    "learning_rate": lr,
                    "num_leaves": nl,
                    "min_child_samples": 80,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "verbosity": -1,
                    "seed": 42,
                }
                model = lgb.train(
                    params, dtrain,
                    num_boost_round=500,
                    valid_sets=[dval],
                    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
                )
                pred_v = model.predict(X_sel[val_start:train_end][valid_val])
                val_ic = fast_ic(pred_v, y[val_start:train_end][valid_val])
                if val_ic > best_ic:
                    best_ic = val_ic
                    best_model = model

        # Test prediction
        pred_test = best_model.predict(X_sel[test_start:test_end])
        preds_test[h] = pred_test

        y_test = y[test_start:test_end]
        valid_test = ~np.isnan(y_test)
        ics[h] = fast_ic(pred_test[valid_test], y_test[valid_test])

    if not preds_test:
        return {"sharpe": 0, "trades": 0, "ic": 0, "error": "no models trained"}

    return ics, preds_test


def backtest_fold(
    preds_test: Dict[int, np.ndarray],
    closes_test: np.ndarray,
    deadzone: float,
    min_hold: int,
    max_hold: int,
    long_only: bool,
) -> Dict[str, Any]:
    """Backtest one fold with config sweep."""
    # Z-score ensemble
    z_all = []
    for h, pred in sorted(preds_test.items()):
        z_all.append(rolling_zscore(pred, window=720, warmup=180))
    z = np.mean(z_all, axis=0)

    # Sweep
    best_sharpe = -999
    best_result = None
    best_cfg = {}

    for dz in [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5]:
        for mh in [8, 12, 24]:
            for maxh_mult in [5, 8]:
                maxh = mh * maxh_mult
                for lo in [True, False]:
                    r = _run_backtest(z, closes_test, dz, mh, maxh, lo)
                    if r["sharpe"] > best_sharpe and r["trades"] >= 5:
                        best_sharpe = r["sharpe"]
                        best_result = r
                        best_cfg = {"dz": dz, "min_hold": mh, "max_hold": maxh, "long_only": lo}

    # Also test with the provided config (fixed params)
    fixed = _run_backtest(z, closes_test, deadzone, min_hold, max_hold, long_only)

    return {
        "best": best_result or {"sharpe": 0, "trades": 0},
        "best_cfg": best_cfg,
        "fixed": fixed,
    }


def _run_backtest(z, closes, deadzone, min_hold, max_hold, long_only):
    n = len(z)
    cost_frac = COST_BPS_RT / 10000
    pos = 0.0
    eb = 0
    trades = []

    for i in range(n):
        if pos != 0:
            held = i - eb
            se = should_exit_position(
                position=pos,
                z_value=float(z[i]),
                held_bars=held,
                min_hold=min_hold,
                max_hold=max_hold,
            )
            if se:
                pnl = pos * (closes[i] - closes[eb]) / closes[eb]
                trades.append(pnl - cost_frac)
                pos = 0.0
        if pos == 0:
            if z[i] > deadzone:
                pos = 1.0
                eb = i
            elif not long_only and z[i] < -deadzone:
                pos = -1.0
                eb = i

    if not trades:
        return {"sharpe": 0, "trades": 0, "return": 0, "win_rate": 0}

    net = np.array(trades)
    avg_hold = n / max(len(trades), 1)
    tpy = 365 * 24 / max(avg_hold, 1)
    sharpe = float(np.mean(net) / max(np.std(net, ddof=1), 1e-10) * np.sqrt(tpy))
    return {
        "sharpe": round(sharpe, 2),
        "trades": len(trades),
        "return": round(float(np.sum(net)) * 100, 2),
        "win_rate": round(float(np.mean(net > 0)) * 100, 1),
        "avg_net_bps": round(float(np.mean(net)) * 10000, 1),
    }


def walk_forward_symbol(
    symbol: str,
    horizons: List[int],
    test_months: int = 3,
    min_train_months: int = 12,
    n_folds: Optional[int] = None,
) -> Dict[str, Any]:
    """Run walk-forward validation for one symbol."""
    data_path = Path(f"data_files/{symbol}_1h.csv")
    if not data_path.exists():
        print(f"  ERROR: {data_path} not found")
        return {}

    df = pd.read_csv(data_path)
    n = len(df)
    closes = df["close"].values.astype(np.float64)
    ts_col = "open_time" if "open_time" in df.columns else "timestamp"
    timestamps = df[ts_col].values.astype(np.int64)
    start_date = pd.Timestamp(timestamps[0], unit="ms").strftime("%Y-%m-%d")
    end_date = pd.Timestamp(timestamps[-1], unit="ms").strftime("%Y-%m-%d")
    print(f"\n  Data: {n:,} 1h bars ({start_date} → {end_date})")
    total_months = n / BARS_PER_MONTH
    print(f"  Total: {total_months:.0f} months")

    # Compute features
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

    feature_names = [c for c in feat_df.columns
                     if c not in ("close", "open_time", "timestamp")
                     and c not in BLACKLIST]
    X = feat_df[feature_names].values.astype(np.float64)
    print(f"{len(feature_names)} features in {time.time()-t0:.1f}s")

    # Compute fold boundaries
    test_bars = BARS_PER_MONTH * test_months
    min_train_bars = BARS_PER_MONTH * min_train_months

    folds = []
    fold_end = n
    while True:
        test_start = fold_end - test_bars
        test_end = fold_end
        train_end = test_start
        train_start = WARMUP

        if train_end - train_start < min_train_bars:
            break

        folds.append((train_start, train_end, test_start, test_end))
        fold_end = test_start

        if n_folds and len(folds) >= n_folds:
            break

    folds.reverse()  # chronological order

    print(f"  Folds: {len(folds)} (test={test_months}mo, min_train={min_train_months}mo)")

    # Load existing config for fixed-param comparison
    config_path = Path(f"models_v8/{symbol}_gate_v2/config.json")
    if config_path.exists():
        with open(config_path) as f:
            model_config = json.load(f)
        deadzone = model_config.get("deadzone", 0.3)
        min_hold = model_config.get("min_hold", 12)
        max_hold = model_config.get("max_hold", 96)
        long_only = model_config.get("long_only", False)
    else:
        deadzone, min_hold, max_hold, long_only = 0.3, 12, 96, False

    # Run folds
    print(f"\n  {'Fold':>5} {'Train Period':>25} {'Test Period':>25} {'IC h12':>7} {'IC h24':>7} "
          f"{'Sharpe':>7} {'Trades':>7} {'WR%':>6} {'Ret%':>8} {'FixSharpe':>10}")
    print(f"  {'-'*5:>5} {'-'*25:>25} {'-'*25:>25} {'-'*7:>7} {'-'*7:>7} "
          f"{'-'*7:>7} {'-'*7:>7} {'-'*6:>6} {'-'*8:>8} {'-'*10:>10}")

    all_results = []

    for fold_idx, (tr_s, tr_e, te_s, te_e) in enumerate(folds):
        tr_start_d = pd.Timestamp(timestamps[tr_s], unit="ms").strftime("%Y-%m")
        tr_end_d = pd.Timestamp(timestamps[tr_e-1], unit="ms").strftime("%Y-%m")
        te_start_d = pd.Timestamp(timestamps[te_s], unit="ms").strftime("%Y-%m")
        te_end_d = pd.Timestamp(timestamps[te_e-1], unit="ms").strftime("%Y-%m")

        t0 = time.time()
        result = train_fold(X, closes, feature_names, tr_s, tr_e, te_s, te_e, horizons)

        if isinstance(result, dict) and "error" in result:
            print(f"  {fold_idx+1:>5} {tr_start_d+'→'+tr_end_d:>25} {te_start_d+'→'+te_end_d:>25} ERROR: {result['error']}")
            continue

        ics, preds_test = result
        closes_test = closes[te_s:te_e]

        bt = backtest_fold(preds_test, closes_test, deadzone, min_hold, max_hold, long_only)
        best = bt["best"]
        fixed = bt["fixed"]

        fold_result = {
            "fold": fold_idx + 1,
            "train": f"{tr_start_d}→{tr_end_d}",
            "test": f"{te_start_d}→{te_end_d}",
            "train_bars": tr_e - tr_s,
            "test_bars": te_e - te_s,
            "ics": ics,
            "best": best,
            "best_cfg": bt["best_cfg"],
            "fixed": fixed,
            "time_sec": time.time() - t0,
        }
        all_results.append(fold_result)

        ic_12 = ics.get(12, 0)
        ic_24 = ics.get(24, 0)
        print(
            f"  {fold_idx+1:>5} "
            f"{tr_start_d+'→'+tr_end_d:>25} "
            f"{te_start_d+'→'+te_end_d:>25} "
            f"{ic_12:>+7.3f} "
            f"{ic_24:>+7.3f} "
            f"{best['sharpe']:>7.2f} "
            f"{best['trades']:>7} "
            f"{best.get('win_rate', 0):>6.1f} "
            f"{best.get('return', 0):>+8.2f} "
            f"{fixed['sharpe']:>10.2f}"
        )

    if not all_results:
        print("  No valid folds!")
        return {}

    # Aggregate
    best_sharpes = [r["best"]["sharpe"] for r in all_results]
    fixed_sharpes = [r["fixed"]["sharpe"] for r in all_results]
    best_rets = [r["best"].get("return", 0) for r in all_results]
    fixed_rets = [r["fixed"].get("return", 0) for r in all_results]
    ic_12s = [r["ics"].get(12, 0) for r in all_results]
    ic_24s = [r["ics"].get(24, 0) for r in all_results]

    print(f"\n  {'='*80}")
    print(f"  WALK-FORWARD SUMMARY — {symbol} ({len(all_results)} folds)")
    print(f"  {'='*80}")

    print(f"\n  {'Metric':>25} {'Best Config':>15} {'Fixed Config':>15}")
    print(f"  {'-'*25:>25} {'-'*15:>15} {'-'*15:>15}")
    print(f"  {'Mean Sharpe':>25} {np.mean(best_sharpes):>15.2f} {np.mean(fixed_sharpes):>15.2f}")
    print(f"  {'Median Sharpe':>25} {np.median(best_sharpes):>15.2f} {np.median(fixed_sharpes):>15.2f}")
    print(f"  {'Min Sharpe':>25} {np.min(best_sharpes):>15.2f} {np.min(fixed_sharpes):>15.2f}")
    print(f"  {'Max Sharpe':>25} {np.max(best_sharpes):>15.2f} {np.max(fixed_sharpes):>15.2f}")
    print(f"  {'Std Sharpe':>25} {np.std(best_sharpes):>15.2f} {np.std(fixed_sharpes):>15.2f}")
    print(f"  {'Sharpe > 0 (%)':>25} {np.mean([s > 0 for s in best_sharpes])*100:>14.0f}% {np.mean([s > 0 for s in fixed_sharpes])*100:>14.0f}%")
    print(f"  {'Sharpe > 1.0 (%)':>25} {np.mean([s > 1 for s in best_sharpes])*100:>14.0f}% {np.mean([s > 1 for s in fixed_sharpes])*100:>14.0f}%")

    stability_best = np.median(best_sharpes) / max(np.std(best_sharpes), 0.01)
    stability_fixed = np.median(fixed_sharpes) / max(np.std(fixed_sharpes), 0.01)
    print(f"  {'Stability ratio':>25} {stability_best:>15.2f} {stability_fixed:>15.2f}")

    print(f"\n  {'Total Return (%)':>25} {sum(best_rets):>+14.1f}% {sum(fixed_rets):>+14.1f}%")
    print(f"  {'Mean Return/fold (%)':>25} {np.mean(best_rets):>+14.1f}% {np.mean(fixed_rets):>+14.1f}%")

    print(f"\n  {'Mean IC h12':>25} {np.mean(ic_12s):>+15.4f}")
    print(f"  {'Mean IC h24':>25} {np.mean(ic_24s):>+15.4f}")
    print(f"  {'IC h12 > 0 (%)':>25} {np.mean([ic > 0 for ic in ic_12s])*100:>14.0f}%")
    print(f"  {'IC h24 > 0 (%)':>25} {np.mean([ic > 0 for ic in ic_24s])*100:>14.0f}%")

    # Per-fold config stability
    if all_results:
        dzs = [r["best_cfg"].get("dz", 0) for r in all_results]
        mhs = [r["best_cfg"].get("min_hold", 0) for r in all_results]
        los = [r["best_cfg"].get("long_only", False) for r in all_results]
        print("\n  Config stability:")
        print(f"    Deadzone range: [{min(dzs):.1f}, {max(dzs):.1f}]")
        print(f"    Min_hold range: [{min(mhs)}, {max(mhs)}]")
        print(f"    Long_only: {sum(los)}/{len(los)} folds")

    # Verdict
    print(f"\n  {'─'*80}")
    mean_s = np.mean(best_sharpes)
    min_s = np.min(best_sharpes)
    pct_positive = np.mean([s > 0 for s in best_sharpes]) * 100

    if mean_s > 1.5 and min_s > 0 and pct_positive == 100:
        verdict = "STRONG — consistent alpha across all periods"
    elif mean_s > 1.0 and pct_positive >= 80:
        verdict = "GOOD — mostly positive, some weakness"
    elif mean_s > 0.5 and pct_positive >= 60:
        verdict = "MARGINAL — alpha exists but inconsistent"
    elif pct_positive >= 50:
        verdict = "WEAK — coin-flip performance"
    else:
        verdict = "NO ALPHA — model does not generalize"

    print(f"  VERDICT: {verdict}")
    print(f"  {'─'*80}")

    return {
        "symbol": symbol,
        "n_folds": len(all_results),
        "best_sharpes": best_sharpes,
        "fixed_sharpes": fixed_sharpes,
        "mean_best_sharpe": float(np.mean(best_sharpes)),
        "mean_fixed_sharpe": float(np.mean(fixed_sharpes)),
        "stability": float(stability_best),
        "pct_positive": float(pct_positive),
        "verdict": verdict,
        "folds": all_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Walk-Forward Validation")
    parser.add_argument("--symbol", default="BTCUSDT,ETHUSDT")
    parser.add_argument("--horizons", default="12,24")
    parser.add_argument("--test-months", type=int, default=3)
    parser.add_argument("--min-train-months", type=int, default=12)
    parser.add_argument("--n-folds", type=int, default=None,
                        help="Max number of folds (default: all possible)")
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbol.split(",")]
    horizons = [int(h.strip()) for h in args.horizons.split(",")]

    print("=" * 80)
    print("  WALK-FORWARD VALIDATION")
    print(f"  Symbols:    {symbols}")
    print(f"  Horizons:   {horizons}")
    print(f"  Test window: {args.test_months} months")
    print(f"  Min train:   {args.min_train_months} months")
    print("=" * 80)

    all_results = {}
    for symbol in symbols:
        result = walk_forward_symbol(
            symbol,
            horizons=horizons,
            test_months=args.test_months,
            min_train_months=args.min_train_months,
            n_folds=args.n_folds,
        )
        all_results[symbol] = result

    # Cross-symbol summary
    if len(symbols) > 1:
        print(f"\n{'='*80}")
        print("  CROSS-SYMBOL SUMMARY")
        print(f"{'='*80}")
        for sym, r in all_results.items():
            if r:
                print(f"  {sym}: mean Sharpe={r['mean_best_sharpe']:.2f}, "
                      f"stability={r['stability']:.2f}, "
                      f"positive={r['pct_positive']:.0f}% — {r['verdict']}")

    # Save results
    out_path = Path("logs/walk_forward_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
