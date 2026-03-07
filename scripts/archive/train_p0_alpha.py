#!/usr/bin/env python3
"""Train P0 Alpha — enriched + crypto-native + cross-asset features.

Combines:
  1. 26 technical features (enriched)
  2. 7 crypto-native features (time, regime, funding)
  3. 2 cross-asset features (BTC lead-lag for altcoins)

Usage:
    python3 -m scripts.train_p0_alpha --all
    python3 -m scripts.train_p0_alpha --symbol ETHUSDT
    python3 -m scripts.train_p0_alpha --all --oos  (OOS validation only)
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from alpha.models.lgbm_alpha import LGBMAlphaModel
from alpha.training.trainer import ModelTrainer
from features.enriched_computer import EnrichedFeatureComputer, ENRICHED_FEATURE_NAMES
from features.cross_asset_computer import CrossAssetComputer, CROSS_ASSET_FEATURE_NAMES

logger = logging.getLogger(__name__)

TARGET_HORIZON = 5
TRAIN_RATIO = 0.70

# Cross-asset feature names (added for altcoins)
CROSS_ASSET_FEATURES = CROSS_ASSET_FEATURE_NAMES


def _load_funding_schedule(path: Path) -> Dict[int, float]:
    """Load funding rates as {timestamp_ms: rate}."""
    schedule: Dict[int, float] = {}
    if not path.exists():
        return schedule
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = int(row["timestamp"])
            rate = float(row["funding_rate"])
            schedule[ts] = rate
    return schedule


def _find_nearest_funding(ts_ms: int, schedule: Dict[int, float], tolerance_ms: int = 3600_000) -> Optional[float]:
    """Find the most recent funding rate at or before ts_ms."""
    best_ts = None
    best_rate = None
    for fts, rate in schedule.items():
        if fts <= ts_ms and (best_ts is None or fts > best_ts):
            best_ts = fts
            best_rate = rate
    return best_rate


def compute_features_for_symbol(
    df: pd.DataFrame,
    symbol: str,
    funding_schedule: Dict[int, float],
    cross_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Compute full P0 feature set for one symbol."""
    computer = EnrichedFeatureComputer()
    records = []

    # Pre-sort funding schedule keys for efficient lookup
    funding_times = sorted(funding_schedule.keys())
    funding_idx = 0

    for _, row in df.iterrows():
        close = float(row["close"])
        volume = float(row.get("volume", 0))
        high = float(row.get("high", close))
        low = float(row.get("low", close))
        open_ = float(row.get("open", close))
        trades = float(row.get("trades", 0) or 0)
        taker_buy_volume = float(row.get("taker_buy_volume", 0) or 0)
        quote_volume = float(row.get("quote_volume", 0) or 0)

        # Parse timestamp for time features
        ts_raw = row.get("timestamp") or row.get("ts") or row.get("open_time", "")
        hour = -1
        dow = -1
        ts_ms = 0
        if ts_raw:
            try:
                ts_ms = int(ts_raw)
                dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
                hour = dt.hour
                dow = dt.weekday()
            except (ValueError, OSError):
                pass

        # Find applicable funding rate
        funding_rate = None
        while funding_idx < len(funding_times) and funding_times[funding_idx] <= ts_ms:
            funding_rate = funding_schedule[funding_times[funding_idx]]
            funding_idx += 1
        # If we advanced past, the last one found is applicable
        if funding_rate is None and funding_idx > 0:
            funding_rate = funding_schedule[funding_times[funding_idx - 1]]

        feats = computer.on_bar(
            symbol, close=close, volume=volume, high=high, low=low,
            open_=open_, hour=hour, dow=dow, funding_rate=funding_rate,
            trades=trades, taker_buy_volume=taker_buy_volume,
            quote_volume=quote_volume,
        )

        # Add cross-asset features from pre-computed DataFrame
        if cross_df is not None and symbol != "BTCUSDT" and ts_ms in cross_df.index:
            cross_row = cross_df.loc[ts_ms]
            for name in CROSS_ASSET_FEATURE_NAMES:
                val = cross_row.get(name)
                feats[name] = None if pd.isna(val) else float(val)
        elif symbol != "BTCUSDT":
            for name in CROSS_ASSET_FEATURE_NAMES:
                feats[name] = None

        records.append(feats)

    return pd.DataFrame(records)


def compute_cross_asset_features(
    all_dfs: Dict[str, pd.DataFrame],
    all_funding: Dict[str, Dict[int, float]],
) -> Dict[str, pd.DataFrame]:
    """Compute cross-asset features for all symbols by replaying bars in time order.

    Returns dict mapping symbol → DataFrame of cross-asset features indexed by timestamp.
    """
    cross = CrossAssetComputer()

    # Collect all (timestamp, symbol, close, funding_rate) tuples
    events: List[tuple] = []
    for symbol, df in all_dfs.items():
        funding_schedule = all_funding.get(symbol, {})
        funding_times = sorted(funding_schedule.keys())
        f_idx = 0
        for _, row in df.iterrows():
            ts_raw = row.get("timestamp") or row.get("open_time", 0)
            try:
                ts_ms = int(ts_raw)
            except (ValueError, TypeError):
                ts_ms = 0
            close = float(row["close"])

            funding_rate = None
            while f_idx < len(funding_times) and funding_times[f_idx] <= ts_ms:
                funding_rate = funding_schedule[funding_times[f_idx]]
                f_idx += 1
            if funding_rate is None and f_idx > 0:
                funding_rate = funding_schedule[funding_times[f_idx - 1]]

            events.append((ts_ms, symbol, close, funding_rate))

    # Sort by timestamp, then symbol (BTC first for consistent state)
    events.sort(key=lambda x: (x[0], x[1]))

    # Replay and collect features
    records: Dict[str, List[Dict]] = {sym: [] for sym in all_dfs}
    # Group by timestamp
    from itertools import groupby
    for ts_ms, group in groupby(events, key=lambda x: x[0]):
        items = list(group)
        # First: update all symbols at this timestamp
        for _, symbol, close, funding_rate in items:
            cross.on_bar(symbol, close=close, funding_rate=funding_rate)
        # Then: collect features for each non-BTC symbol
        for _, symbol, close, funding_rate in items:
            if symbol != "BTCUSDT":
                feats = cross.get_features(symbol)
                feats["_ts_ms"] = ts_ms
                records[symbol].append(feats)

    result: Dict[str, pd.DataFrame] = {}
    for sym, recs in records.items():
        if recs:
            result[sym] = pd.DataFrame(recs).set_index("_ts_ms")

    return result


def get_feature_names(symbol: str) -> tuple[str, ...]:
    """Get feature names including cross-asset if applicable."""
    if symbol == "BTCUSDT":
        return ENRICHED_FEATURE_NAMES
    return ENRICHED_FEATURE_NAMES + CROSS_ASSET_FEATURES


def run_one(symbol: str, out_base: Path, *, oos_only: bool = False,
            cross_features: Optional[Dict[str, pd.DataFrame]] = None) -> Optional[dict]:
    csv_path = Path(f"data_files/{symbol}_1h.csv")
    funding_path = Path(f"data_files/{symbol}_funding.csv")

    if not csv_path.exists():
        print(f"  SKIP {symbol}: CSV not found")
        return None

    print(f"\n{'='*60}")
    print(f"  {symbol} — P0 Alpha Training")
    print(f"{'='*60}")

    df = pd.read_csv(csv_path)
    print(f"  Bars: {len(df)}")

    # Load funding schedule
    funding_schedule = _load_funding_schedule(funding_path)
    print(f"  Funding rates: {len(funding_schedule)} records")

    # Get cross-asset features for this symbol
    cross_df = None
    if cross_features is not None and symbol in cross_features:
        cross_df = cross_features[symbol]
        print(f"  Cross-asset features: {len(cross_df)} rows")

    # Compute features
    feature_names = get_feature_names(symbol)
    print(f"  Computing {len(feature_names)} features...")
    feat_df = compute_features_for_symbol(df, symbol, funding_schedule, cross_df)

    # Target
    target = df["close"].shift(-TARGET_HORIZON) / df["close"] - 1.0

    # Align — drop features that are entirely NaN (e.g. OI/LS when data not available)
    X = feat_df[list(feature_names)]
    all_nan_cols = X.columns[X.isna().all()]
    if len(all_nan_cols) > 0:
        print(f"  Dropping {len(all_nan_cols)} all-NaN features: {list(all_nan_cols)}")
        X = X.drop(columns=all_nan_cols)
        feature_names = tuple(c for c in feature_names if c not in set(all_nan_cols))
    y = target
    # Sparse features (OI/LS) — allow NaN, LGBM handles natively
    SPARSE_FEATURES = {
        "oi_change_pct", "oi_change_ma8", "oi_close_divergence",
        "ls_ratio", "ls_ratio_zscore_24", "ls_extreme",
    }
    core_cols = [c for c in feature_names if c not in SPARSE_FEATURES]
    mask = X[core_cols].notna().all(axis=1) & y.notna()
    X_clean = X[mask].values.astype(np.float64)
    y_clean = y[mask].values
    clean_idx = np.where(mask.values)[0]

    print(f"  Valid samples: {len(X_clean)} (dropped {len(df) - len(X_clean)})")

    if len(X_clean) < 500:
        print(f"  ERROR: Not enough valid samples ({len(X_clean)})")
        return None

    # Split
    split = int(len(X_clean) * TRAIN_RATIO)
    X_train, X_test = X_clean[:split], X_clean[split:]
    y_train, y_test = y_clean[:split], y_clean[split:]

    if "timestamp" in df.columns:
        train_end = df["timestamp"].iloc[clean_idx[split - 1]]
        test_start = df["timestamp"].iloc[clean_idx[split]]
        dt_train = datetime.fromtimestamp(int(train_end) / 1000, tz=timezone.utc)
        dt_test = datetime.fromtimestamp(int(test_start) / 1000, tz=timezone.utc)
        print(f"  Train: {len(X_train)} bars → {dt_train.strftime('%Y-%m-%d')}")
        print(f"  Test:  {len(X_test)} bars ← {dt_test.strftime('%Y-%m-%d')}")

    # Train
    print(f"  Training LGBM...")
    model = LGBMAlphaModel(name="lgbm_alpha", feature_names=feature_names)
    model.fit(
        X_train, y_train,
        params={
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.02,
            "num_leaves": 31,
            "min_child_samples": 100,
            "reg_alpha": 0.5,
            "reg_lambda": 2.0,
            "subsample": 0.7,
            "colsample_bytree": 0.7,
            "objective": "regression",
            "verbosity": -1,
        },
    )

    # Save
    out_dir = out_base / symbol
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(out_dir / "lgbm_alpha_final.pkl")

    # OOS evaluation
    from scripts.oos_eval import compute_1bar_returns, evaluate_oos, print_evaluation

    y_pred = model._model.predict(X_test)

    # 1-bar returns from close prices
    closes_all = df["close"].values
    test_orig_idx = clean_idx[split:]
    ret_1bar = compute_1bar_returns(closes_all, test_orig_idx)

    eval_result = evaluate_oos(y_pred, y_test, ret_1bar)

    # Feature importance
    importances = model._model.feature_importances_
    feat_imp = sorted(zip(feature_names, importances), key=lambda x: -x[1])

    print_evaluation(eval_result, label="P0 OOS")
    print(f"\n  Top 10 features:")
    for feat, imp in feat_imp[:10]:
        print(f"    {feat:<25s} {imp:>6d}")

    pq = eval_result["prediction_quality"]
    best_thr = eval_result["best_threshold"]
    live_row = next((r for r in eval_result["threshold_scan"] if r["threshold"] == 0.001), eval_result["threshold_scan"][0])

    result = {
        "symbol": symbol,
        "n_features": len(feature_names),
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
        "top_features": [{"name": f, "importance": int(i)} for f, i in feat_imp[:15]],
    }

    with open(out_dir / "p0_oos_results.json", "w") as f:
        json.dump(result, f, indent=2)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train P0 Alpha (crypto-native features)")
    parser.add_argument("--symbol", help="Single symbol")
    parser.add_argument("--all", action="store_true", help="All 3 symbols")
    parser.add_argument("--out", default="models", help="Output directory")
    parser.add_argument("--oos", action="store_true", help="OOS validation only")
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

    # Build cross-asset features if training multiple symbols
    cross_features = None
    if len(symbols) > 1:
        print("\n  Building cross-asset features...")
        all_dfs: Dict[str, pd.DataFrame] = {}
        all_funding: Dict[str, Dict[int, float]] = {}
        for sym in symbols:
            csv_p = Path(f"data_files/{sym}_1h.csv")
            if csv_p.exists():
                all_dfs[sym] = pd.read_csv(csv_p)
                all_funding[sym] = _load_funding_schedule(Path(f"data_files/{sym}_funding.csv"))
        if len(all_dfs) > 1:
            cross_features = compute_cross_asset_features(all_dfs, all_funding)
            print(f"  Cross-asset features ready for {list(cross_features.keys())}")

    results = {}
    for sym in symbols:
        r = run_one(sym, out_base, oos_only=args.oos, cross_features=cross_features)
        if r:
            results[sym] = r

    if results:
        print(f"\n\n{'='*110}")
        print(f"  P0 Alpha OOS Summary (1-bar PnL, with costs, threshold=0.001)")
        print(f"{'='*110}")
        print(f"{'Symbol':<10} {'Feats':>5} {'DirAcc':>8} {'IC':>8} "
              f"{'GrossRet':>10} {'NetRet':>10} {'Costs':>8} {'Sharpe':>8} {'Trades':>7} {'BestThr':>8}")
        print(f"{'-'*110}")
        for sym, r in results.items():
            print(f"{sym:<10} {r['n_features']:>5} {r['oos_direction_accuracy']*100:>7.1f}% "
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
