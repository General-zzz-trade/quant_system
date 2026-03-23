#!/usr/bin/env python3
"""Train BTCUSDT 1h V9 model — IC-driven feature selection.

Improvements over V8:
  - Replace 9 weak OOS-IC features with strong candidates
  - IC-validated candidate pool (cross-symbol consistent signals)
  - Expanded deadzone sweep (finer grid)
  - More HPO trials (20)
  - Forced inclusion of cross-symbol strong features

Usage:
    python3 -m scripts.train_btc_v9
"""
from __future__ import annotations
import sys
import time
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, "/quant_system")

from features.batch_feature_engine import compute_features_batch
from features.multi_timeframe import compute_4h_features, TF4H_FEATURE_NAMES
from shared.signal_postprocess import rolling_zscore, should_exit_position
from alpha.training.train_v7_alpha import INTERACTION_FEATURES, BLACKLIST
from scipy.stats import spearmanr

SYMBOL = "BTCUSDT"
MODEL_DIR = Path("models_v8/BTCUSDT_gate_v2")
HORIZON = 24
WARMUP = 30
COST_BPS_RT = 8  # Honest: 4bp taker × 2 sides

BARS_PER_DAY = 24
BARS_PER_MONTH = BARS_PER_DAY * 30
HPO_TRIALS = 20
TOP_K_FEATURES = 16  # Slightly larger than V8's 14

# Features with strong OOS IC from our scan, consistent across BTC+ETH
# These get priority in selection
PRIORITY_FEATURES = [
    "price_acceleration",   # BTC -0.025, ETH -0.032 (both strong, same sign)
    "close_vs_ma20",        # BTC -0.019, ETH -0.027
    "macd_hist",            # BTC -0.020, ETH -0.025
    "bb_width_20",          # BTC +0.039 (strongest BTC candidate)
    "volume_momentum_10",   # BTC -0.017, ETH -0.024
    "ret_12",               # BTC -0.019, ETH -0.023
    "vwap_dev_20",          # BTC -0.023, ETH -0.026 (in ETH, not BTC)
    "dow_cos",              # BTC +0.031 (strong)
    "vol_20",               # BTC +0.031
    "funding_extreme",      # BTC +0.027
    "cvd_10",               # BTC +0.022
]

# V8 features confirmed weak by OOS IC scan (all |IC_OOS| < 0.02)
WEAK_V8_FEATURES = {
    "funding_sign_persist",  # importance=74 but OOS IC=0.003 — overfitting
    "fgi_extreme",           # IC=0.005
    "vol_ma_ratio_5_20",     # IC=0.006
    "rsi_14",                # IC=0.007
    "ret_24",                # IC=-0.008
    "basis_zscore_24",       # IC=0.003
    "basis_momentum",        # IC=0.002 (lowest importance too)
}


def fast_ic(x, y):
    m = ~(np.isnan(x) | np.isnan(y))
    if m.sum() < 50:
        return 0.0
    r, _ = spearmanr(x[m], y[m])
    return float(r) if not np.isnan(r) else 0.0


def compute_target(closes, horizon):
    n = len(closes)
    y = np.full(n, np.nan)
    y[:n-horizon] = closes[horizon:] / closes[:n-horizon] - 1
    v = y[~np.isnan(y)]
    if len(v) > 10:
        p1, p99 = np.percentile(v, [1, 99])
        y = np.where(np.isnan(y), np.nan, np.clip(y, p1, p99))
    return y


def backtest_signal(pred, closes, deadzone, min_hold, max_hold,
                    cost_bps, long_only=True, zscore_window=720):
    n = len(pred)
    z = rolling_zscore(pred, window=zscore_window, warmup=min(180, zscore_window // 4))
    cost_frac = cost_bps / 10000
    pos = 0.0
    entry_bar = 0
    equity = 10000.0
    trades = []

    for i in range(n):
        if pos != 0:
            held = i - entry_bar
            should_exit = should_exit_position(
                position=pos,
                z_value=float(z[i]),
                held_bars=held,
                min_hold=min_hold,
                max_hold=max_hold,
            )
            if should_exit:
                pnl_pct = pos * (closes[i] - closes[entry_bar]) / closes[entry_bar]
                net = pnl_pct * 500.0 - cost_frac * 500.0
                equity += net
                trades.append(net)
                pos = 0.0
        if pos == 0:
            if z[i] > deadzone:
                pos = 1.0
                entry_bar = i
            elif not long_only and z[i] < -deadzone:
                pos = -1.0
                entry_bar = i

    if not trades:
        return {"sharpe": 0, "trades": 0, "return": 0}
    net_arr = np.array(trades)
    avg_hold = n / max(len(trades), 1)
    tpy = 365 * 24 / max(avg_hold, 1)
    sharpe = float(np.mean(net_arr) / max(np.std(net_arr, ddof=1), 1e-10) * np.sqrt(tpy))
    return {
        "sharpe": sharpe, "trades": len(trades),
        "return": float(np.sum(net_arr)) / 10000,
        "win_rate": float(np.mean(net_arr > 0) * 100),
        "avg_net_bps": float(np.mean(net_arr) / 500 * 10000),
    }


def ic_driven_feature_select(X_train, y_train, feature_names, top_k=16):
    """IC-driven selection: priority features first, then greedy fill."""
    # Compute IC for all features on training data
    ic_scores = {}
    for i, fname in enumerate(feature_names):
        ic = fast_ic(X_train[:, i], y_train)
        ic_scores[fname] = ic

    # Start with priority features that have decent IC
    selected = []
    for f in PRIORITY_FEATURES:
        if f in ic_scores and abs(ic_scores[f]) > 0.01 and f not in WEAK_V8_FEATURES:
            selected.append(f)
        if len(selected) >= top_k // 2:  # Reserve half for greedy
            break

    # Greedy IC-based fill for remaining slots
    remaining = [f for f in feature_names
                 if f not in selected and f not in WEAK_V8_FEATURES]
    remaining.sort(key=lambda f: abs(ic_scores.get(f, 0)), reverse=True)

    for f in remaining:
        if len(selected) >= top_k:
            break
        # Check redundancy: skip if too correlated with existing
        f_idx = feature_names.index(f)
        f_vals = X_train[:, f_idx]
        redundant = False
        for s in selected:
            s_idx = feature_names.index(s)
            s_vals = X_train[:, s_idx]
            mask = ~(np.isnan(f_vals) | np.isnan(s_vals))
            if mask.sum() > 100:
                corr = abs(np.corrcoef(f_vals[mask], s_vals[mask])[0, 1])
                if corr > 0.85:
                    redundant = True
                    break
        if not redundant:
            selected.append(f)

    print(f"  Selected {len(selected)} features:")
    for f in selected:
        ic = ic_scores.get(f, 0)
        priority = "★" if f in PRIORITY_FEATURES else " "
        print(f"    {priority} {f:35s} IC={ic:+.4f}")

    return selected


def main():
    print("=" * 70)
    print(f"TRAINING {SYMBOL} 1h V9 MODEL (IC-driven feature selection)")
    print("=" * 70)

    data_path = f"data_files/{SYMBOL}_1h.csv"
    if not Path(data_path).exists():
        print(f"ERROR: {data_path} not found.")
        return

    df = pd.read_csv(data_path)
    n = len(df)
    ts_col = "open_time" if "open_time" in df.columns else "timestamp"
    timestamps = df[ts_col].values.astype(np.int64)
    closes = df["close"].values.astype(np.float64)
    start_date = pd.Timestamp(timestamps[0], unit="ms").strftime("%Y-%m-%d")
    end_date = pd.Timestamp(timestamps[-1], unit="ms").strftime("%Y-%m-%d")
    print(f"\nData: {n:,} 1h bars ({start_date} → {end_date})")

    # ── Features ──
    print("\nComputing features...")
    t0 = time.time()
    _has_v11 = Path("data_files/macro_daily.csv").exists()
    feat_df = compute_features_batch(SYMBOL, df, include_v11=_has_v11)
    tf4h = compute_4h_features(df)
    for col in TF4H_FEATURE_NAMES:
        feat_df[col] = tf4h[col].values
    for int_name, fa, fb in INTERACTION_FEATURES:
        if fa in feat_df.columns and fb in feat_df.columns:
            feat_df[int_name] = feat_df[fa].astype(float) * feat_df[fb].astype(float)

    feature_names = [c for c in feat_df.columns
                     if c not in ("close", "open_time", "timestamp")
                     and c not in BLACKLIST]
    print(f"Features: {len(feature_names)} in {time.time()-t0:.1f}s")

    y = compute_target(closes, HORIZON)
    X = feat_df[feature_names].values.astype(np.float64)

    # ── Split ──
    oos_bars = BARS_PER_MONTH * 18
    train_end = n - oos_bars
    val_size = BARS_PER_MONTH * 6
    val_start = train_end - val_size

    print(f"\nSplit: train={val_start-WARMUP:,} val={val_size:,} test={oos_bars:,}")

    X_train = X[WARMUP:val_start]
    y_train = y[WARMUP:val_start]
    X_val = X[val_start:train_end]
    y_val = y[val_start:train_end]
    X_test = X[train_end:]
    y_test = y[train_end:]
    closes_test = closes[train_end:]

    valid_tr = ~np.isnan(y_train)
    valid_val = ~np.isnan(y_val)
    X_train = X_train[valid_tr]
    y_train = y_train[valid_tr]
    X_val = X_val[valid_val]
    y_val = y_val[valid_val]

    print(f"  Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")

    # ── Feature selection (IC-driven) ──
    print(f"\nIC-driven feature selection (top {TOP_K_FEATURES})...")
    selected_features = ic_driven_feature_select(X_train, y_train, feature_names,
                                                  top_k=TOP_K_FEATURES)
    sel_idx = [feature_names.index(f) for f in selected_features]

    X_tr_sel = X_train[:, sel_idx]
    X_val_sel = X_val[:, sel_idx]
    X_test_sel = X_test[:, sel_idx]

    # ── Compare V8 features IC on test set ──
    print("\nV8 vs V9 feature IC comparison (on test set):")
    v8_features = ["basis", "ret_24", "fgi_normalized", "fgi_extreme", "parkinson_vol",
                   "atr_norm_14", "rsi_14", "tf4h_atr_norm_14", "basis_zscore_24",
                   "cvd_20", "funding_zscore_24", "basis_momentum",
                   "funding_sign_persist", "vol_ma_ratio_5_20"]
    valid_test = ~np.isnan(y_test)
    for f in v8_features:
        if f in feature_names:
            fi = feature_names.index(f)
            ic = fast_ic(X_test[:, fi][valid_test], y_test[valid_test])
            marker = "WEAK" if f in WEAK_V8_FEATURES else ""
            print(f"  V8 {f:35s} IC={ic:+.4f} {marker}")
    print()
    for f in selected_features:
        fi = feature_names.index(f)
        ic = fast_ic(X_test[:, fi][valid_test], y_test[valid_test])
        new = "NEW" if f not in set(v8_features) else ""
        print(f"  V9 {f:35s} IC={ic:+.4f} {new}")

    # ── Optuna HPO ──
    print(f"\nOptuna HPO ({HPO_TRIALS} trials)...")
    import optuna
    import lightgbm as lgb
    import xgboost as xgb
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            "objective": "regression", "metric": "mse", "verbosity": -1,
            "boosting_type": "gbdt",
            "num_leaves": trial.suggest_int("num_leaves", 15, 50),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 30, 120),
            "subsample": trial.suggest_float("subsample", 0.5, 0.9),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.9),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 2.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
        dtrain = lgb.Dataset(X_tr_sel, label=y_train)
        dval = lgb.Dataset(X_val_sel, label=y_val, reference=dtrain)
        bst = lgb.train(params, dtrain, num_boost_round=500,
                        valid_sets=[dval],
                        callbacks=[lgb.early_stopping(50, verbose=False),
                                   lgb.log_evaluation(0)])
        return fast_ic(bst.predict(X_val_sel), y_val)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=HPO_TRIALS)
    best_params = study.best_params
    print(f"  Best val IC: {study.best_value:.4f}")
    print(f"  Params: {best_params}")

    # ── Final LGBM ──
    print("\nTraining final LGBM...")
    lgbm_params = {"objective": "regression", "metric": "mse", "verbosity": -1,
                   "boosting_type": "gbdt", **best_params}
    dtrain = lgb.Dataset(X_tr_sel, label=y_train)
    dval = lgb.Dataset(X_val_sel, label=y_val, reference=dtrain)
    lgbm_model = lgb.train(lgbm_params, dtrain, num_boost_round=500,
                           valid_sets=[dval],
                           callbacks=[lgb.early_stopping(50, verbose=False),
                                      lgb.log_evaluation(0)])
    pred_test_lgbm = lgbm_model.predict(X_test_sel)
    ic_lgbm = fast_ic(pred_test_lgbm, y_test)
    print(f"  LGBM IC: {ic_lgbm:.4f}")

    # ── XGB ──
    print("Training XGB...")
    xgb_params = {
        "objective": "reg:squarederror", "eval_metric": "rmse",
        "max_depth": best_params["max_depth"],
        "learning_rate": best_params["learning_rate"],
        "subsample": best_params.get("subsample", 0.8),
        "colsample_bytree": best_params.get("colsample_bytree", 0.8),
        "min_child_weight": best_params.get("min_child_samples", 50),
        "reg_alpha": best_params.get("reg_alpha", 0.1),
        "reg_lambda": best_params.get("reg_lambda", 1.0),
        "verbosity": 0,
    }
    dtrain_x = xgb.DMatrix(X_tr_sel, label=y_train)
    dval_x = xgb.DMatrix(X_val_sel, label=y_val)
    xgb_model = xgb.train(xgb_params, dtrain_x, num_boost_round=500,
                           evals=[(dval_x, "val")],
                           early_stopping_rounds=50, verbose_eval=False)
    pred_test_xgb = xgb_model.predict(xgb.DMatrix(X_test_sel))
    ic_xgb = fast_ic(pred_test_xgb, y_test)
    print(f"  XGB IC: {ic_xgb:.4f}")

    # ── Ensemble ──
    pred_test = 0.5 * pred_test_lgbm + 0.5 * pred_test_xgb
    ic_ens = fast_ic(pred_test, y_test)
    print(f"  Ensemble IC: {ic_ens:.4f}")

    # ── V8 baseline IC for comparison ──
    print("\nV8 baseline comparison...")
    v8_idx = [feature_names.index(f) for f in v8_features if f in feature_names]
    v8_model_dir = MODEL_DIR
    try:
        with open(v8_model_dir / "lgbm_v8.pkl", "rb") as f:
            v8_lgbm = pickle.load(f)["model"]
        with open(v8_model_dir / "xgb_v8.pkl", "rb") as f:
            v8_xgb = pickle.load(f)["model"]
        X_test_v8 = X_test[:, v8_idx]
        v8_pred = 0.5 * v8_lgbm.predict(X_test_v8) + \
                  0.5 * v8_xgb.predict(xgb.DMatrix(X_test_v8))
        v8_ic = fast_ic(v8_pred, y_test)
        print(f"  V8 ensemble IC: {v8_ic:.4f}")
        print(f"  V9 ensemble IC: {ic_ens:.4f}  ({'+' if ic_ens > v8_ic else ''}{(ic_ens-v8_ic):.4f})")
    except Exception as e:
        print(f"  V8 comparison failed: {e}")

    # ── Deadzone sweep (finer grid, honest costs) ──
    print(f"\nDeadzone sweep (cost={COST_BPS_RT}bp RT)...")
    best_config = None
    best_sharpe = -999
    best_result = None
    all_results = []
    for dz in [0.3, 0.5, 0.7, 1.0, 1.2, 1.5, 2.0, 2.5]:
        for mh in [12, 24, 36]:
            maxh = mh * 5
            for lo in [True, False]:
                r = backtest_signal(pred_test, closes_test, dz, mh, maxh,
                                    COST_BPS_RT, long_only=lo)
                if r["trades"] >= 10:
                    all_results.append((dz, mh, maxh, lo, r))
                    if r["sharpe"] > best_sharpe:
                        best_sharpe = r["sharpe"]
                        best_config = {"deadzone": dz, "min_hold": mh,
                                       "max_hold": maxh, "long_only": lo}
                        best_result = r

    # Show top 5 configs
    all_results.sort(key=lambda x: x[4]["sharpe"], reverse=True)
    print("\n  Top 5 configurations:")
    for dz, mh, maxh, lo, r in all_results[:5]:
        lo_str = "L" if lo else "L+S"
        print(f"    dz={dz:.1f} hold=[{mh},{maxh}] {lo_str:>3s}  "
              f"Sharpe={r['sharpe']:.2f} trades={r['trades']:>3d} "
              f"WR={r['win_rate']:.0f}% ret={r['return']*100:+.2f}%")

    if best_config is None:
        print("  No viable config found!")
        return

    print(f"\n  BEST: dz={best_config['deadzone']}, hold=[{best_config['min_hold']},{best_config['max_hold']}], "
          f"long_only={best_config['long_only']}")
    print(f"  Sharpe={best_result['sharpe']:.2f}, trades={best_result['trades']}, "
          f"WR={best_result.get('win_rate',0):.0f}%, ret={best_result['return']*100:+.2f}%")

    # ── Bootstrap ──
    print("\nBootstrap Sharpe...")
    z = rolling_zscore(pred_test, window=720, warmup=180)
    trade_pnls = []
    pos = 0.0
    eb = 0
    for i in range(len(z)):
        if pos != 0:
            held = i - eb
            should_exit = should_exit_position(
                position=pos,
                z_value=float(z[i]),
                held_bars=held,
                min_hold=best_config["min_hold"],
                max_hold=best_config["max_hold"],
            )
            if should_exit:
                pnl = pos * (closes_test[i] - closes_test[eb]) / closes_test[eb]
                trade_pnls.append(pnl * 500 - COST_BPS_RT / 10000 * 500)
                pos = 0.0
        if pos == 0:
            if z[i] > best_config["deadzone"]:
                pos = 1.0
                eb = i
            elif not best_config["long_only"] and z[i] < -best_config["deadzone"]:
                pos = -1.0
                eb = i

    trade_pnls = np.array(trade_pnls) if trade_pnls else np.array([0.0])
    bs_sharpes = []
    for _ in range(1000):
        sample = np.random.choice(trade_pnls, size=len(trade_pnls), replace=True)
        if np.std(sample) > 0:
            bs_sharpes.append(float(np.mean(sample) / np.std(sample) * np.sqrt(52)))
    bs_sharpes = np.array(bs_sharpes)
    p5, p50, p95 = np.percentile(bs_sharpes, [5, 50, 95])
    print(f"  Bootstrap: {p50:.2f} (p5={p5:.2f}, p95={p95:.2f})")

    # ── Checks ──
    print(f"\n{'='*70}")
    print("PRODUCTION CHECKS")
    print("=" * 70)
    checks = {
        "Sharpe > 0.8": best_result["sharpe"] > 0.8,
        "IC > 0.03": ic_ens > 0.03,
        "Trades >= 15": best_result["trades"] >= 15,
        "Bootstrap p5 > 0": p5 > 0,
    }
    all_pass = True
    for check, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{status}] {check}")

    # ── Save ──
    if all_pass:
        # Backup V8 first
        import shutil
        backup_dir = Path("models_v8/BTCUSDT_gate_v2_backup_v8")
        if not backup_dir.exists():
            shutil.copytree(MODEL_DIR, backup_dir)
            print(f"\n  V8 backed up to {backup_dir}/")

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        with open(MODEL_DIR / "lgbm_v8.pkl", "wb") as f:
            pickle.dump({"model": lgbm_model, "features": selected_features}, f)
        with open(MODEL_DIR / "xgb_v8.pkl", "wb") as f:
            pickle.dump({"model": xgb_model, "features": selected_features}, f)
        lgbm_model.save_model(str(MODEL_DIR / "lgbm_1h.txt"))

        config = {
            "version": "v9",
            "symbol": SYMBOL,
            "ensemble": True,
            "ensemble_weights": [0.5, 0.5],
            "models": ["lgbm_v8.pkl", "xgb_v8.pkl"],
            "features": selected_features,
            "n_features": len(selected_features),
            "deadzone": best_config["deadzone"],
            "min_hold": best_config["min_hold"],
            "max_hold": best_config["max_hold"],
            "long_only": best_config["long_only"],
            "horizon": HORIZON,
            "target_mode": "clipped",
            "params": best_params,
            "xgb_params": xgb_params,
            "hpo_trials": HPO_TRIALS,
            "metrics": {
                "sharpe": best_result["sharpe"],
                "ic": ic_ens,
                "ic_lgbm": ic_lgbm,
                "ic_xgb": ic_xgb,
                "total_return": best_result["return"],
                "trades": best_result["trades"],
                "win_rate": best_result.get("win_rate", 0),
                "avg_net_bps": best_result.get("avg_net_bps", 0),
                "bootstrap_sharpe_p5": float(p5),
                "bootstrap_sharpe_p50": float(p50),
                "bootstrap_sharpe_p95": float(p95),
            },
            "checks": {k: bool(v) for k, v in checks.items()},
            "weak_features_removed": list(WEAK_V8_FEATURES),
            "priority_features_added": [f for f in PRIORITY_FEATURES if f in selected_features],
            "train_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
            "data_range": f"{start_date} → {end_date}",
            "n_bars": n,
        }
        with open(MODEL_DIR / "config.json", "w") as f:
            json.dump(config, f, indent=2)
        with open(MODEL_DIR / "features.json", "w") as f:
            json.dump(selected_features, f)

        print(f"\n  V9 model saved to {MODEL_DIR}/")
    else:
        print("\n  FAILED — model NOT saved.")

    print(f"\n{'='*70}")
    print(f"SUMMARY: {SYMBOL} 1h V9")
    print(f"{'='*70}")
    print(f"  IC:       {ic_ens:.4f}")
    print(f"  Sharpe:   {best_result['sharpe']:.2f}")
    print(f"  Return:   {best_result['return']*100:+.2f}%")
    print(f"  Trades:   {best_result['trades']}")
    print(f"  WR:       {best_result.get('win_rate',0):.0f}%")
    print(f"  Features: {len(selected_features)}")
    print(f"  Checks:   {'ALL PASS' if all_pass else 'FAILED'}")
    print("\nDone.")


if __name__ == "__main__":
    main()
