#!/usr/bin/env python3
"""Train V2 Alpha — all features + dynamic selection + interaction terms.

Combines:
  1. Enriched features (48: original 35 + 8 micro + 5 funding deep)
  2. OI/LS features (6) when available
  3. Cross-asset features (10 for altcoins)
  4. Interaction features (5)
  5. Dynamic feature selection per walk-forward fold

Usage:
    python3 -m scripts.train_v2_alpha --all
    python3 -m scripts.train_v2_alpha --symbol BTCUSDT
    python3 -m scripts.train_v2_alpha --all --top-k 25
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from alpha.models.lgbm_alpha import LGBMAlphaModel
from features.enriched_computer import EnrichedFeatureComputer, ENRICHED_FEATURE_NAMES
from features.cross_asset_computer import CrossAssetComputer, CROSS_ASSET_FEATURE_NAMES
from features.dynamic_selector import rolling_ic_select

logger = logging.getLogger(__name__)

TARGET_HORIZON = 5
TRAIN_RATIO = 0.70

# Interaction feature definitions: (name, feat_a, feat_b)
INTERACTION_FEATURES = [
    ("rsi14_x_vol_regime", "rsi_14", "vol_regime"),
    ("funding_x_taker_imb", "funding_rate", "taker_imbalance"),
    ("btc_ret1_x_beta30", "btc_ret_1", "rolling_beta_30"),
    ("trade_int_x_body", "trade_intensity", "body_ratio"),
    ("oi_chg_x_ret1", "oi_change_pct", "ret_1"),
]


def _load_funding_schedule(path: Path) -> Dict[int, float]:
    schedule: Dict[int, float] = {}
    if not path.exists():
        return schedule
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            schedule[int(row["timestamp"])] = float(row["funding_rate"])
    return schedule


def _load_oi_schedule(path: Path) -> Dict[int, float]:
    schedule: Dict[int, float] = {}
    if not path.exists():
        return schedule
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            schedule[int(row["timestamp"])] = float(row["sum_open_interest"])
    return schedule


def _load_ls_schedule(path: Path) -> Dict[int, float]:
    schedule: Dict[int, float] = {}
    if not path.exists():
        return schedule
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            schedule[int(row["timestamp"])] = float(row["long_short_ratio"])
    return schedule


def compute_features_v2(
    symbol: str,
    cross_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Compute all V2 features for one symbol."""
    csv_path = Path(f"data_files/{symbol}_1h.csv")
    funding_path = Path(f"data_files/{symbol}_funding.csv")
    oi_path = Path(f"data_files/{symbol}_open_interest.csv")
    ls_path = Path(f"data_files/{symbol}_ls_ratio.csv")

    df = pd.read_csv(csv_path)
    funding_schedule = _load_funding_schedule(funding_path)
    oi_schedule = _load_oi_schedule(oi_path)
    ls_schedule = _load_ls_schedule(ls_path)

    funding_times = sorted(funding_schedule.keys())
    oi_times = sorted(oi_schedule.keys())
    ls_times = sorted(ls_schedule.keys())
    f_idx, oi_idx, ls_idx = 0, 0, 0

    comp = EnrichedFeatureComputer()
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
        hour, dow, ts_ms = -1, -1, 0
        if ts_raw:
            try:
                ts_ms = int(ts_raw)
                dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
                hour, dow = dt.hour, dt.weekday()
            except (ValueError, OSError):
                pass

        # Funding rate
        funding_rate = None
        while f_idx < len(funding_times) and funding_times[f_idx] <= ts_ms:
            funding_rate = funding_schedule[funding_times[f_idx]]
            f_idx += 1
        if funding_rate is None and f_idx > 0:
            funding_rate = funding_schedule[funding_times[f_idx - 1]]

        # OI
        open_interest = None
        while oi_idx < len(oi_times) and oi_times[oi_idx] <= ts_ms:
            open_interest = oi_schedule[oi_times[oi_idx]]
            oi_idx += 1
        if open_interest is None and oi_idx > 0:
            open_interest = oi_schedule[oi_times[oi_idx - 1]]

        # LS Ratio
        ls_ratio = None
        while ls_idx < len(ls_times) and ls_times[ls_idx] <= ts_ms:
            ls_ratio = ls_schedule[ls_times[ls_idx]]
            ls_idx += 1
        if ls_ratio is None and ls_idx > 0:
            ls_ratio = ls_schedule[ls_times[ls_idx - 1]]

        feats = comp.on_bar(
            symbol, close=close, volume=volume, high=high, low=low,
            open_=open_, hour=hour, dow=dow, funding_rate=funding_rate,
            trades=trades, taker_buy_volume=taker_buy_volume,
            quote_volume=quote_volume,
            open_interest=open_interest, ls_ratio=ls_ratio,
        )

        # Cross-asset features
        if cross_df is not None and symbol != "BTCUSDT" and ts_ms in cross_df.index:
            cross_row = cross_df.loc[ts_ms]
            for name in CROSS_ASSET_FEATURE_NAMES:
                val = cross_row.get(name)
                feats[name] = None if pd.isna(val) else float(val)
        elif symbol != "BTCUSDT":
            for name in CROSS_ASSET_FEATURE_NAMES:
                feats[name] = None

        records.append(feats)

    feat_df = pd.DataFrame(records)

    # Add interaction features
    for int_name, feat_a, feat_b in INTERACTION_FEATURES:
        if feat_a in feat_df.columns and feat_b in feat_df.columns:
            a = feat_df[feat_a].astype(float)
            b = feat_df[feat_b].astype(float)
            feat_df[int_name] = a * b
        else:
            feat_df[int_name] = np.nan

    feat_df["close"] = df["close"].values
    return feat_df


def get_all_feature_names(symbol: str) -> List[str]:
    """All possible feature names for V2."""
    names = list(ENRICHED_FEATURE_NAMES)
    if symbol != "BTCUSDT":
        names.extend(CROSS_ASSET_FEATURE_NAMES)
    names.extend(name for name, _, _ in INTERACTION_FEATURES)
    return names


def run_one(symbol: str, out_base: Path, *, top_k: int = 25,
            cross_features: Optional[Dict[str, pd.DataFrame]] = None) -> Optional[dict]:
    csv_path = Path(f"data_files/{symbol}_1h.csv")
    if not csv_path.exists():
        print(f"  SKIP {symbol}: CSV not found")
        return None

    print(f"\n{'='*60}")
    print(f"  {symbol} — V2 Alpha Training")
    print(f"{'='*60}")

    cross_df = cross_features.get(symbol) if cross_features else None
    feat_df = compute_features_v2(symbol, cross_df)
    print(f"  Bars: {len(feat_df)}")

    all_names = get_all_feature_names(symbol)
    available = [n for n in all_names if n in feat_df.columns]
    print(f"  Available features: {len(available)}")

    # Target
    target = feat_df["close"].shift(-TARGET_HORIZON) / feat_df["close"] - 1.0

    # Align — drop features that are entirely NaN
    X = feat_df[available]
    all_nan_cols = X.columns[X.isna().all()]
    if len(all_nan_cols) > 0:
        print(f"  Dropping {len(all_nan_cols)} all-NaN features: {list(all_nan_cols)}")
        X = X.drop(columns=all_nan_cols)
        available = [c for c in available if c not in set(all_nan_cols)]
    y = target

    # Sparse features (OI/LS/interactions with OI) — allow NaN, LGBM handles it
    SPARSE_FEATURES = {
        "oi_change_pct", "oi_change_ma8", "oi_close_divergence",
        "ls_ratio", "ls_ratio_zscore_24", "ls_extreme",
        "oi_chg_x_ret1", "btc_ret1_x_beta30",
    }
    core_cols = [c for c in available if c not in SPARSE_FEATURES]
    core_mask = X[core_cols].notna().all(axis=1) & y.notna()
    mask = core_mask
    X_clean = X[mask].values.astype(np.float64)
    y_clean = y[mask].values
    clean_idx = np.where(mask.values)[0]

    # Report NaN coverage for sparse features
    nan_pct = np.isnan(X_clean).mean(axis=0)
    sparse_feats = [(available[i], nan_pct[i]) for i in range(len(available)) if nan_pct[i] > 0.01]
    if sparse_feats:
        print(f"  Sparse features (>1% NaN, LGBM handles natively):")
        for name, pct in sparse_feats:
            print(f"    {name:<30s} {pct*100:.1f}% NaN")

    print(f"  Valid samples: {len(X_clean)}")
    if len(X_clean) < 500:
        print(f"  ERROR: Not enough samples")
        return None

    # Split
    split = int(len(X_clean) * TRAIN_RATIO)
    X_train, X_test = X_clean[:split], X_clean[split:]
    y_train, y_test = y_clean[:split], y_clean[split:]

    # Dynamic feature selection on training data
    print(f"  Running dynamic feature selection (top-{top_k})...")
    selected = rolling_ic_select(X_train, y_train, available, top_k=top_k)
    selected_idx = [available.index(n) for n in selected]
    print(f"  Selected: {selected}")

    X_train_sel = X_train[:, selected_idx]
    X_test_sel = X_test[:, selected_idx]

    # Train
    print(f"  Training LGBM with {len(selected)} features...")
    model = LGBMAlphaModel(name="lgbm_v2_alpha", feature_names=tuple(selected))
    model.fit(
        X_train_sel, y_train,
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
    model.save(out_dir / "lgbm_v2_alpha.pkl")

    # OOS evaluation
    from scripts.oos_eval import compute_1bar_returns, evaluate_oos, print_evaluation

    y_pred = model._model.predict(X_test_sel)

    # 1-bar returns from close prices
    closes_all = feat_df["close"].values
    test_orig_idx = clean_idx[split:]
    ret_1bar = compute_1bar_returns(closes_all, test_orig_idx)

    eval_result = evaluate_oos(y_pred, y_test, ret_1bar)

    importances = model._model.feature_importances_
    feat_imp = sorted(zip(selected, importances), key=lambda x: -x[1])

    print(f"\n  Features used: {len(selected)} / {len(available)}")
    print_evaluation(eval_result, label="V2 OOS")
    print(f"\n  Top features:")
    for feat, imp in feat_imp[:10]:
        print(f"    {feat:<30s} {imp:>6d}")

    pq = eval_result["prediction_quality"]
    best_thr = eval_result["best_threshold"]
    # Pick threshold=0.001 row for JSON (matches live default)
    live_row = next((r for r in eval_result["threshold_scan"] if r["threshold"] == 0.001), eval_result["threshold_scan"][0])

    result = {
        "symbol": symbol,
        "version": "v2",
        "n_features_available": len(available),
        "n_features_selected": len(selected),
        "selected_features": selected,
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

    with open(out_dir / "v2_oos_results.json", "w") as f:
        json.dump(result, f, indent=2)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train V2 Alpha")
    parser.add_argument("--symbol", help="Single symbol")
    parser.add_argument("--all", action="store_true", help="All symbols")
    parser.add_argument("--out", default="models", help="Output directory")
    parser.add_argument("--top-k", type=int, default=25, help="Top-K features to select")
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

    # Build cross-asset features
    cross_features = None
    if len(symbols) > 1:
        from scripts.train_p0_alpha import compute_cross_asset_features
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

    results = {}
    for sym in symbols:
        r = run_one(sym, out_base, top_k=args.top_k, cross_features=cross_features)
        if r:
            results[sym] = r

    if results:
        print(f"\n\n{'='*120}")
        print(f"  V2 Alpha OOS Summary (1-bar PnL, with costs, threshold=0.001)")
        print(f"{'='*120}")
        print(f"{'Symbol':<10} {'Total':>5} {'Used':>4} {'DirAcc':>8} {'IC':>8} "
              f"{'GrossRet':>10} {'NetRet':>10} {'Costs':>8} {'Sharpe':>8} {'Trades':>7} {'BestThr':>8}")
        print(f"{'-'*120}")
        for sym, r in results.items():
            print(f"{sym:<10} {r['n_features_available']:>5} {r['n_features_selected']:>4} "
                  f"{r['oos_direction_accuracy']*100:>7.1f}% "
                  f"{r['oos_ic']:>8.4f} "
                  f"{r['oos_gross_return']*100:>9.2f}% "
                  f"{r['oos_net_return']*100:>9.2f}% "
                  f"{r['oos_total_costs']*100:>7.2f}% "
                  f"{r['oos_sharpe_annual']:>8.2f} "
                  f"{r['oos_n_trades']:>7d} "
                  f"{r['oos_best_threshold']:>8.4f}")
        print(f"{'='*120}")


if __name__ == "__main__":
    main()
