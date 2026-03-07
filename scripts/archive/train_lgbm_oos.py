#!/usr/bin/env python3
"""Train LGBM with enriched features — proper OOS validation.

Train on first 70% of data, test on last 30%. No walk-forward.
Measures true out-of-sample direction accuracy and runs backtest on OOS period.

Usage:
    python3 -m scripts.train_lgbm_oos --all
    python3 -m scripts.train_lgbm_oos --symbol BTCUSDT
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

from alpha.models.lgbm_alpha import LGBMAlphaModel
from features.enriched_computer import EnrichedFeatureComputer, ENRICHED_FEATURE_NAMES

logger = logging.getLogger(__name__)

TARGET_HORIZON = 5
TRAIN_RATIO = 0.70


def compute_enriched_features(df: pd.DataFrame, symbol: str = "BTCUSDT") -> pd.DataFrame:
    from datetime import datetime, timezone
    computer = EnrichedFeatureComputer()
    records = []
    for _, row in df.iterrows():
        close = float(row["close"])
        volume = float(row.get("volume", 0))
        high = float(row.get("high", close))
        low = float(row.get("low", close))
        open_ = float(row.get("open", close))
        trades = float(row.get("trades", 0) or 0)
        taker_buy_volume = float(row.get("taker_buy_volume", 0) or 0)
        quote_volume = float(row.get("quote_volume", 0) or 0)

        ts_raw = row.get("timestamp") or row.get("open_time", "")
        hour, dow = -1, -1
        if ts_raw:
            try:
                ts_ms = int(ts_raw)
                dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
                hour, dow = dt.hour, dt.weekday()
            except (ValueError, OSError):
                pass

        feats = computer.on_bar(
            symbol, close=close, volume=volume, high=high, low=low,
            open_=open_, hour=hour, dow=dow,
            trades=trades, taker_buy_volume=taker_buy_volume,
            quote_volume=quote_volume,
        )
        records.append(feats)
    return pd.DataFrame(records)


def run_oos_for_symbol(symbol: str, out_base: Path) -> dict | None:
    csv_path = Path(f"data_files/{symbol}_1h.csv")
    if not csv_path.exists():
        print(f"  SKIP {symbol}: CSV not found")
        return None

    print(f"\n{'='*60}")
    print(f"  {symbol} — OOS Validation (enriched features)")
    print(f"{'='*60}")

    df = pd.read_csv(csv_path)
    n_total = len(df)
    print(f"  Total bars: {n_total}")

    # Compute features
    print(f"  Computing {len(ENRICHED_FEATURE_NAMES)} features...")
    feat_df = compute_enriched_features(df, symbol)

    # Target: forward return
    target = df["close"].shift(-TARGET_HORIZON) / df["close"] - 1.0

    # Align — sparse features (OI/LS/funding deep) allowed NaN, LGBM handles natively
    X = feat_df[list(ENRICHED_FEATURE_NAMES)]
    all_nan_cols = X.columns[X.isna().all()]
    if len(all_nan_cols) > 0:
        print(f"  Dropping {len(all_nan_cols)} all-NaN features: {list(all_nan_cols)}")
        X = X.drop(columns=all_nan_cols)
    SPARSE_FEATURES = {
        "oi_change_pct", "oi_change_ma8", "oi_close_divergence",
        "ls_ratio", "ls_ratio_zscore_24", "ls_extreme",
    }
    core_cols = [c for c in X.columns if c not in SPARSE_FEATURES]
    y = target
    mask = X[core_cols].notna().all(axis=1) & y.notna()
    X_clean = X[mask].values
    y_clean = y[mask].values
    clean_indices = np.where(mask.values)[0]

    print(f"  Valid samples: {len(X_clean)} (dropped {n_total - len(X_clean)} warmup/tail)")

    # Split
    split_idx = int(len(X_clean) * TRAIN_RATIO)
    X_train = X_clean[:split_idx]
    y_train = y_clean[:split_idx]
    X_test = X_clean[split_idx:]
    y_test = y_clean[split_idx:]

    train_start_bar = clean_indices[0]
    train_end_bar = clean_indices[split_idx - 1]
    test_start_bar = clean_indices[split_idx]
    test_end_bar = clean_indices[-1]

    print(f"  Train: {len(X_train)} bars (rows {train_start_bar}-{train_end_bar})")
    print(f"  Test:  {len(X_test)} bars (rows {test_start_bar}-{test_end_bar})")

    if "timestamp" in df.columns or "ts" in df.columns:
        ts_col = "timestamp" if "timestamp" in df.columns else "ts"
        print(f"  Train period: {df[ts_col].iloc[train_start_bar]} → {df[ts_col].iloc[train_end_bar]}")
        print(f"  Test period:  {df[ts_col].iloc[test_start_bar]} → {df[ts_col].iloc[test_end_bar]}")

    # Train with regularization
    active_features = tuple(X.columns)
    model = LGBMAlphaModel(name="lgbm_alpha", feature_names=active_features)
    model.fit(
        X_train, y_train,
        params={
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.03,
            "num_leaves": 31,
            "min_child_samples": 50,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "regression",
            "verbosity": -1,
        },
    )

    # Save OOS model
    out_dir = out_base / symbol
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(out_dir / "lgbm_oos.pkl")

    # Evaluate on test set
    from scripts.oos_eval import compute_1bar_returns, evaluate_oos, print_evaluation

    y_pred = model._model.predict(X_test)

    # 1-bar returns from close prices
    closes_all = df["close"].values
    test_orig_idx = clean_indices[split_idx:]
    ret_1bar = compute_1bar_returns(closes_all, test_orig_idx)

    eval_result = evaluate_oos(y_pred, y_test, ret_1bar)

    # Feature importance
    importances = model._model.feature_importances_
    feat_imp = sorted(zip(active_features, importances), key=lambda x: -x[1])

    print_evaluation(eval_result, label="OOS")
    print(f"\n  Top 10 features by importance:")
    for feat, imp in feat_imp[:10]:
        print(f"    {feat:<25s} {imp:>6d}")

    pq = eval_result["prediction_quality"]
    best_thr = eval_result["best_threshold"]
    live_row = next((r for r in eval_result["threshold_scan"] if r["threshold"] == 0.001), eval_result["threshold_scan"][0])

    result = {
        "symbol": symbol,
        "n_features": len(active_features),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "oos_direction_accuracy": pq["direction_accuracy"],
        "oos_mse": pq["mse"],
        "oos_ic": pq["ic"],
        "oos_net_return": live_row["net_return"],
        "oos_gross_return": live_row["gross_return"],
        "oos_total_costs": live_row["total_costs"],
        "oos_sharpe_annual": live_row["sharpe_annual"],
        "oos_win_rate": live_row["win_rate"],
        "oos_n_trades": live_row["n_trades"],
        "oos_best_threshold": best_thr,
        "threshold_scan": eval_result["threshold_scan"],
        "top_features": [{"name": f, "importance": int(i)} for f, i in feat_imp[:10]],
    }

    with open(out_dir / "oos_results.json", "w") as f:
        json.dump(result, f, indent=2)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="LGBM OOS validation with enriched features")
    parser.add_argument("--symbol", help="Single symbol")
    parser.add_argument("--all", action="store_true", help="All 3 symbols")
    parser.add_argument("--out", default="models", help="Output base directory")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    symbols = []
    if args.all:
        symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    elif args.symbol:
        symbols = [args.symbol.upper()]
    else:
        parser.print_help()
        return

    out_base = Path(args.out)
    results = {}
    for sym in symbols:
        r = run_oos_for_symbol(sym, out_base)
        if r:
            results[sym] = r

    if results:
        print(f"\n\n{'='*110}")
        print(f"  LGBM OOS Summary (1-bar PnL, with costs, threshold=0.001)")
        print(f"{'='*110}")
        print(f"{'Symbol':<10} {'DirAcc':>8} {'IC':>8} "
              f"{'GrossRet':>10} {'NetRet':>10} {'Costs':>8} {'Sharpe':>8} {'Trades':>7} {'BestThr':>8}")
        print(f"{'-'*110}")
        for sym, r in results.items():
            print(f"{sym:<10} {r['oos_direction_accuracy']*100:>7.1f}% "
                  f"{r['oos_ic']:>8.4f} "
                  f"{r['oos_gross_return']*100:>9.2f}% "
                  f"{r['oos_net_return']*100:>9.2f}% "
                  f"{r['oos_total_costs']*100:>7.2f}% "
                  f"{r['oos_sharpe_annual']:>8.2f} "
                  f"{r['oos_n_trades']:>7d} "
                  f"{r['oos_best_threshold']:>8.4f}")
        print(f"{'='*110}")


if __name__ == "__main__":
    main()
