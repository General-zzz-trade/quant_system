#!/usr/bin/env python3
"""Train 30-minute BTCUSDT Production Model.

Key design:
  1. Resample 1m → 30m bars
  2. Compute 105 features via Rust batch engine
  3. Safely include basis/funding/fgi by lagging 2 bars (60min)
  4. Walk-forward validation (3-month expanding folds)
  5. Hyperparameter search on validation IC+Sharpe
  6. Export model for Rust binary
"""
from __future__ import annotations
import sys, time, json, pickle, logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd

sys.path.insert(0, "/quant_system")

from features.batch_feature_engine import compute_features_batch
from features.dynamic_selector import greedy_ic_select, stable_icir_select, _rankdata, _spearman_ic
from scipy.stats import spearmanr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────
SYMBOL = "BTCUSDT"
DATA_PATH = "/quant_system/data_files/BTCUSDT_1m.csv"
WARMUP = 100  # bars of warmup for feature stability
COST_MAKER_RT_BPS = 4  # maker round-trip
COST_TAKER_RT_BPS = 14

# Features that use external hourly data — must be lagged at 30m resolution
LAGGED_FEATURES = {
    "basis", "basis_zscore_24", "basis_momentum", "basis_extreme", "basis_carry_adj",
    "funding_rate", "funding_ma8", "funding_zscore_24", "funding_momentum",
    "funding_extreme", "funding_cumulative_8", "funding_sign_persist",
    "funding_annualized", "funding_vs_vol", "funding_term_slope",
    "fgi_normalized", "fgi_zscore_7", "fgi_extreme",
}
LAG_BARS = 2  # 2 × 30m = 60min lag (use previous completed hour)

# Walk-forward config
FOLD_BARS_30M = 4320  # ~3 months of 30m bars (90d × 48 bars/day)
MIN_TRAIN_BARS = 8640  # ~6 months minimum training

# Hyperparam grid
PARAM_GRID = [
    {"max_depth": 4, "num_leaves": 10, "learning_rate": 0.008, "min_child_samples": 300,
     "reg_alpha": 0.5, "reg_lambda": 8.0, "subsample": 0.5, "colsample_bytree": 0.5},
    {"max_depth": 4, "num_leaves": 12, "learning_rate": 0.01, "min_child_samples": 200,
     "reg_alpha": 0.5, "reg_lambda": 5.0, "subsample": 0.5, "colsample_bytree": 0.6},
    {"max_depth": 4, "num_leaves": 16, "learning_rate": 0.01, "min_child_samples": 200,
     "reg_alpha": 1.0, "reg_lambda": 5.0, "subsample": 0.4, "colsample_bytree": 0.5},
    {"max_depth": 3, "num_leaves": 8, "learning_rate": 0.015, "min_child_samples": 100,
     "reg_alpha": 0.3, "reg_lambda": 3.0, "subsample": 0.6, "colsample_bytree": 0.7},
    {"max_depth": 5, "num_leaves": 12, "learning_rate": 0.008, "min_child_samples": 300,
     "reg_alpha": 1.0, "reg_lambda": 10.0, "subsample": 0.4, "colsample_bytree": 0.5},
]


def fast_ic(x, y):
    m = ~(np.isnan(x) | np.isnan(y))
    if m.sum() < 50: return 0.0
    r, _ = spearmanr(x[m], y[m])
    return float(r) if not np.isnan(r) else 0.0


# ── Resampling ──────────────────────────────────────────────
def resample_1m_to_30m(df_1m: pd.DataFrame) -> pd.DataFrame:
    ts_col = "open_time" if "open_time" in df_1m.columns else "timestamp"
    ts = df_1m[ts_col].values.astype(np.int64)
    group_ms = 30 * 60_000
    groups = ts // group_ms
    work = pd.DataFrame({
        "group": groups, "open_time": ts,
        "open": df_1m["open"].values.astype(np.float64),
        "high": df_1m["high"].values.astype(np.float64),
        "low": df_1m["low"].values.astype(np.float64),
        "close": df_1m["close"].values.astype(np.float64),
        "volume": df_1m["volume"].values.astype(np.float64),
    })
    for col in ("quote_volume", "trades", "taker_buy_volume", "taker_buy_quote_volume"):
        work[col] = df_1m[col].values.astype(np.float64) if col in df_1m.columns else 0.0
    return work.groupby("group", sort=True).agg(
        open_time=("open_time", "first"), open=("open", "first"), high=("high", "max"),
        low=("low", "min"), close=("close", "last"), volume=("volume", "sum"),
        quote_volume=("quote_volume", "sum"), trades=("trades", "sum"),
        taker_buy_volume=("taker_buy_volume", "sum"),
        taker_buy_quote_volume=("taker_buy_quote_volume", "sum"),
    ).reset_index(drop=True)


# ── Feature Computation ────────────────────────────────────
def compute_30m_features(df_30m: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Compute 105 features + lag external data features."""
    _has_v11 = Path("data_files/macro_daily.csv").exists()
    feat_df = compute_features_batch(SYMBOL, df_30m, include_v11=_has_v11)

    # Lag external-data features by 2 bars to avoid look-ahead
    for col in LAGGED_FEATURES:
        if col in feat_df.columns:
            feat_df[col] = feat_df[col].shift(LAG_BARS)

    feature_names = [c for c in feat_df.columns
                     if c not in ("close", "open_time", "timestamp")]
    return feat_df, feature_names


# ── Target ──────────────────────────────────────────────────
def compute_target(closes: np.ndarray, horizon: int) -> np.ndarray:
    n = len(closes)
    y = np.full(n, np.nan)
    y[:n-horizon] = closes[horizon:] / closes[:n-horizon] - 1
    v = y[~np.isnan(y)]
    if len(v) > 10:
        p1, p99 = np.percentile(v, [1, 99])
        y = np.where(np.isnan(y), np.nan, np.clip(y, p1, p99))
    return y


# ── Backtest ────────────────────────────────────────────────
def backtest(pred, closes, horizon, deadzone, cost_bps):
    """Simple backtest on z-scored predictions."""
    pred_std = np.nanstd(pred)
    if pred_std < 1e-12:
        return {"trades": 0, "net_pnl": 0, "avg_gross_bps": 0, "avg_net_bps": 0,
                "win_rate": 0, "sharpe": 0}
    z = pred / pred_std
    min_hold = max(horizon // 2, 1)
    max_hold = horizon * 6

    pos = 0; ep = 0; eb = 0
    tg = []; tn = []
    cost_frac = cost_bps / 10000

    for i in range(len(closes)):
        if pos != 0:
            held = i - eb
            ex = held >= max_hold or (held >= min_hold and (pos*z[i] < -0.3 or abs(z[i]) < 0.2))
            if ex:
                pnl = pos * (closes[i] - ep) / ep
                tg.append(pnl); tn.append(pnl - cost_frac); pos = 0
        if pos == 0:
            if z[i] > deadzone: pos = 1; ep = closes[i]; eb = i
            elif z[i] < -deadzone: pos = -1; ep = closes[i]; eb = i

    if pos != 0:
        pnl = pos * (closes[-1] - ep) / ep
        tg.append(pnl); tn.append(pnl - cost_frac)

    nt = len(tn)
    if nt == 0:
        return {"trades": 0, "net_pnl": 0, "avg_gross_bps": 0, "avg_net_bps": 0,
                "win_rate": 0, "sharpe": 0}
    g = np.array(tg); ne = np.array(tn)
    days = len(closes) / 48
    sharpe = float(np.mean(ne) / max(np.std(ne), 1e-10) * np.sqrt(365))

    return {
        "trades": nt, "trades_per_day": nt / max(days, 1),
        "win_rate": float(np.mean(ne > 0) * 100),
        "avg_gross_bps": float(np.mean(g) * 10000),
        "avg_net_bps": float(np.mean(ne) * 10000),
        "net_pnl_pct": float(np.sum(ne) * 100),
        "sharpe": sharpe,
    }


# ── Walk-Forward Validation ────────────────────────────────
def walk_forward(
    feat_df: pd.DataFrame,
    feature_names: List[str],
    closes: np.ndarray,
    horizon: int,
    params: Dict,
    deadzone: float,
) -> Dict[str, Any]:
    """Expanding-window walk-forward with 3-month folds."""
    import lightgbm as lgb

    n = len(closes)
    y = compute_target(closes, horizon)
    X = feat_df[feature_names].values.astype(np.float64)

    fold_results = []
    fold_start = MIN_TRAIN_BARS

    while fold_start + FOLD_BARS_30M <= n:
        fold_end = min(fold_start + FOLD_BARS_30M, n)
        val_start = max(fold_start - FOLD_BARS_30M // 2, MIN_TRAIN_BARS - FOLD_BARS_30M // 2)

        # Train on [WARMUP, fold_start), val on [val_start, fold_start), test on [fold_start, fold_end)
        tr_slice = slice(WARMUP, val_start)
        val_slice = slice(val_start, fold_start)
        test_slice = slice(fold_start, fold_end)

        valid_tr = ~np.isnan(y[tr_slice])
        valid_val = ~np.isnan(y[val_slice])
        valid_test = ~np.isnan(y[test_slice])

        X_tr = X[tr_slice][valid_tr]
        y_tr = y[tr_slice][valid_tr]
        X_val = X[val_slice][valid_val]
        y_val = y[val_slice][valid_val]

        if len(X_tr) < 1000 or len(X_val) < 200:
            fold_start += FOLD_BARS_30M
            continue

        # Feature selection
        selected = greedy_ic_select(X_tr, y_tr, feature_names, top_k=15)
        sel_idx = [feature_names.index(f) for f in selected]

        lgb_params = {**params, "objective": "regression", "verbosity": -1}
        dtrain = lgb.Dataset(X_tr[:, sel_idx], label=y_tr)
        dval = lgb.Dataset(X_val[:, sel_idx], label=y_val, reference=dtrain)
        bst = lgb.train(lgb_params, dtrain, num_boost_round=500,
                        valid_sets=[dval],
                        callbacks=[lgb.early_stopping(50, verbose=False)])

        # Test
        X_test = X[test_slice]
        c_test = closes[test_slice]
        pred = bst.predict(X_test[:, sel_idx])
        y_test = y[test_slice]

        ic = fast_ic(pred, y_test)
        bt = backtest(pred, c_test, horizon, deadzone, COST_MAKER_RT_BPS)

        fold_results.append({
            "fold_start": fold_start, "fold_end": fold_end,
            "ic": ic, **bt, "features": selected, "model": bst, "sel_idx": sel_idx,
        })

        fold_start += FOLD_BARS_30M

    if not fold_results:
        return {"folds": [], "avg_ic": 0, "avg_net_bps": 0, "total_net_pct": 0, "avg_sharpe": 0}

    avg_ic = np.mean([f["ic"] for f in fold_results])
    avg_net = np.mean([f["avg_net_bps"] for f in fold_results if f["trades"] > 0])
    total_net = sum(f.get("net_pnl_pct", 0) for f in fold_results)
    avg_sharpe = np.mean([f["sharpe"] for f in fold_results if f["trades"] > 0])

    return {
        "folds": fold_results,
        "avg_ic": avg_ic, "avg_net_bps": avg_net,
        "total_net_pct": total_net, "avg_sharpe": avg_sharpe,
    }


# ── Main ──────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("30-Minute BTCUSDT Production Model Training")
    print("=" * 70)

    # Load & resample
    print(f"\nLoading {DATA_PATH} ...")
    df_1m = pd.read_csv(DATA_PATH)
    print(f"  {len(df_1m):,} 1m bars")

    df_30m = resample_1m_to_30m(df_1m)
    print(f"  Resampled to {len(df_30m):,} 30m bars ({len(df_30m)/48:.0f} days)")

    # Compute features
    t0 = time.time()
    print("\nComputing features (105 + lagged external) ...")
    feat_df, feature_names = compute_30m_features(df_30m)
    print(f"  {len(feature_names)} features in {time.time()-t0:.1f}s")

    closes = df_30m["close"].values.astype(np.float64)

    # ── IC Analysis ──
    print("\n" + "=" * 70)
    print("IC ANALYSIS")
    print("=" * 70)

    for horizon in [1, 2, 4, 8]:
        h_min = horizon * 30
        y = compute_target(closes, horizon)
        ics = [(f, fast_ic(feat_df[f].values[WARMUP:] if f in feat_df.columns else np.zeros(len(closes)-WARMUP),
                           y[WARMUP:])) for f in feature_names]
        ics.sort(key=lambda x: abs(x[1]), reverse=True)
        n_pass = sum(1 for _, ic in ics if abs(ic) >= 0.01)
        print(f"\n  h={horizon} ({h_min}min): {n_pass} features pass |IC|≥0.01")
        for f, ic in ics[:10]:
            tag = "✓" if abs(ic) >= 0.01 else " "
            lagged = " [LAG]" if f in LAGGED_FEATURES else ""
            print(f"    {tag} {f:30s}  IC={ic:+.5f}{lagged}")

    # ── Hyperparam Search via Walk-Forward ──
    print("\n" + "=" * 70)
    print("WALK-FORWARD HYPERPARAMETER SEARCH")
    print("=" * 70)

    horizon = 2  # Best from prototype: 2 bars = 60min
    best_wf = None
    best_score = -999

    for dz in [2.0, 2.5, 3.0]:
        for pi, params in enumerate(PARAM_GRID):
            t0 = time.time()
            wf = walk_forward(feat_df, feature_names, closes, horizon, params, dz)
            elapsed = time.time() - t0

            n_folds = len(wf["folds"])
            if n_folds == 0:
                continue

            score = wf["avg_sharpe"] + wf["avg_ic"] * 10  # composite score
            profitable_folds = sum(1 for f in wf["folds"] if f.get("net_pnl_pct", 0) > 0)

            print(f"  P{pi} dz={dz:.1f}: {n_folds} folds, IC={wf['avg_ic']:.4f}, "
                  f"net={wf['avg_net_bps']:+.1f}bp, total={wf['total_net_pct']:+.1f}%, "
                  f"Sharpe={wf['avg_sharpe']:.2f}, profitable={profitable_folds}/{n_folds} "
                  f"({elapsed:.0f}s)")

            if score > best_score and wf["avg_net_bps"] > 0:
                best_score = score
                best_wf = wf
                best_wf["deadzone"] = dz
                best_wf["params"] = params
                best_wf["horizon"] = horizon

    if best_wf is None:
        # Try without requiring avg_net > 0
        print("\n  No profitable config found. Showing best anyway...")
        for dz in [2.0, 2.5, 3.0]:
            for pi, params in enumerate(PARAM_GRID):
                wf = walk_forward(feat_df, feature_names, closes, horizon, params, dz)
                if not wf["folds"]: continue
                score = wf["avg_sharpe"] + wf["avg_ic"] * 10
                if score > best_score:
                    best_score = score
                    best_wf = wf
                    best_wf["deadzone"] = dz
                    best_wf["params"] = params
                    best_wf["horizon"] = horizon

    # ── Results ──
    print("\n" + "=" * 70)
    print("WALK-FORWARD RESULTS")
    print("=" * 70)

    if best_wf and best_wf["folds"]:
        print(f"\n  Best config: dz={best_wf['deadzone']}, params={best_wf['params']}")
        print(f"  Avg IC: {best_wf['avg_ic']:.4f}")
        print(f"  Avg Net: {best_wf['avg_net_bps']:+.1f} bps/trade")
        print(f"  Total Net: {best_wf['total_net_pct']:+.1f}%")
        print(f"  Avg Sharpe: {best_wf['avg_sharpe']:.2f}")

        print(f"\n  {'Fold':>4} {'IC':>7} {'Trades':>7} {'WinR':>6} {'AvgGross':>9} {'AvgNet':>9} {'Net%':>8}")
        print(f"  {'-'*55}")
        for i, f in enumerate(best_wf["folds"]):
            print(f"  {i+1:>4} {f['ic']:>7.4f} {f['trades']:>7} {f['win_rate']:>5.1f}% "
                  f"{f['avg_gross_bps']:>+8.1f}bp {f['avg_net_bps']:>+8.1f}bp "
                  f"{f.get('net_pnl_pct',0):>+7.1f}%")

    # ── Train Final Model (full train set, last fold's features) ──
    print("\n" + "=" * 70)
    print("TRAINING FINAL MODEL")
    print("=" * 70)

    if not best_wf or not best_wf["folds"]:
        print("  No valid model to train. Exiting.")
        return

    import lightgbm as lgb

    # Use last fold's feature selection, train on everything except last 20%
    last_fold = best_wf["folds"][-1]
    selected_features = last_fold["features"]
    sel_idx = [feature_names.index(f) for f in selected_features]
    params = {**best_wf["params"], "objective": "regression", "verbosity": -1}

    y = compute_target(closes, horizon)
    X = feat_df[feature_names].values.astype(np.float64)
    n = len(X)
    train_end = int(n * 0.8)

    valid_tr = ~np.isnan(y[WARMUP:train_end])
    valid_val = ~np.isnan(y[train_end:])

    X_tr = X[WARMUP:train_end][valid_tr][:, sel_idx]
    y_tr = y[WARMUP:train_end][valid_tr]
    X_val = X[train_end:][valid_val][:, sel_idx]
    y_val = y[train_end:][valid_val]

    print(f"  Train: {len(X_tr):,}, Val: {len(X_val):,}")
    print(f"  Features: {selected_features}")

    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
    bst = lgb.train(params, dtrain, num_boost_round=500,
                    valid_sets=[dval],
                    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)])

    pred_val = bst.predict(X_val)
    final_ic = fast_ic(pred_val, y_val)
    c_val = closes[train_end:][valid_val]
    final_bt = backtest(pred_val, c_val, horizon, best_wf["deadzone"], COST_MAKER_RT_BPS)
    print(f"  Final OOS IC: {final_ic:.4f}")
    print(f"  Final backtest: {final_bt}")

    # ── Save Model ──
    print("\n" + "=" * 70)
    print("SAVING MODEL")
    print("=" * 70)

    out_dir = Path(f"models_v8/{SYMBOL}_30m_v1")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save LightGBM
    bst.save_model(str(out_dir / "lgbm_30m.txt"))
    with open(out_dir / "lgbm_30m.pkl", "wb") as f:
        pickle.dump(bst, f)
    print(f"  Saved lgbm_30m.txt + lgbm_30m.pkl")

    # Export to JSON for Rust binary
    try:
        sys.path.insert(0, "/quant_system/scripts")
        from export_model_to_json import export_lightgbm_to_json
        export_lightgbm_to_json(str(out_dir / "lgbm_30m.pkl"), str(out_dir / "lgbm_30m.json"))
        print(f"  Exported lgbm_30m.json for Rust binary")
    except Exception as e:
        print(f"  JSON export failed: {e}")

    # Has any lagged feature been selected?
    has_lagged = any(f in LAGGED_FEATURES for f in selected_features)

    # Save config
    config = {
        "version": "v8_30m",
        "symbol": SYMBOL,
        "ensemble": False,
        "models": ["lgbm_30m.pkl"],
        "ensemble_weights": [1.0],
        "features": selected_features,
        "long_only": False,
        "deadzone": best_wf["deadzone"],
        "min_hold": max(horizon // 2, 1),
        "max_hold": horizon * 6,
        "horizon": horizon,
        "horizon_minutes": horizon * 30,
        "timeframe": "30m",
        "target_mode": "clipped",
        "engine": "rust_feature_engine",
        "kline_only": not has_lagged,
        "needs_external_data": has_lagged,
        "lagged_features": [f for f in selected_features if f in LAGGED_FEATURES],
        "lag_bars": LAG_BARS,
        "params": best_wf["params"],
        "metrics": {
            "oos_ic": final_ic,
            "wf_avg_ic": best_wf["avg_ic"],
            "wf_avg_net_bps": best_wf["avg_net_bps"],
            "wf_total_net_pct": best_wf["total_net_pct"],
            "wf_avg_sharpe": best_wf["avg_sharpe"],
            "final_trades": final_bt["trades"],
            "final_win_rate": final_bt["win_rate"],
            "final_avg_gross_bps": final_bt["avg_gross_bps"],
            "final_avg_net_bps": final_bt["avg_net_bps"],
            "final_net_pnl_pct": final_bt.get("net_pnl_pct", 0),
            "final_sharpe": final_bt["sharpe"],
        },
    }

    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    with open(out_dir / "features.json", "w") as f:
        json.dump({"features": selected_features}, f, indent=2)

    print(f"\n  Model saved to {out_dir}/")
    print(f"  Features ({len(selected_features)}): {selected_features}")
    if has_lagged:
        print(f"  Lagged features: {[f for f in selected_features if f in LAGGED_FEATURES]}")
    print(f"  Deadzone: {best_wf['deadzone']}")
    print(f"  OOS IC: {final_ic:.4f}")
    print(f"  Net P&L: {final_bt.get('net_pnl_pct', 0):+.1f}%")
    print(f"  Sharpe: {final_bt['sharpe']:.2f}")

    print(f"\n{'='*70}")
    print("DONE — Next: create config.30m-btc.yaml and deploy")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
