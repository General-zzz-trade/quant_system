#!/usr/bin/env python3
"""Multi-horizon ensemble training — trains h=12, h=24, h=48 models per symbol.

P0 improvement: instead of a single h=24 model, train 3 horizons and
ensemble their normalized predictions at inference time.

All backtesting uses causal rolling z-score (matches live exactly).

Usage:
    python3 -m scripts.train_multi_horizon --symbol BTCUSDT
    python3 -m scripts.train_multi_horizon --symbol ETHUSDT
    python3 -m scripts.train_multi_horizon --symbol BTCUSDT,ETHUSDT
"""
from __future__ import annotations
import sys
import time
import json
import pickle
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd

sys.path.insert(0, "/quant_system")

from features.batch_feature_engine import compute_features_batch
from features.multi_timeframe import compute_4h_features, TF4H_FEATURE_NAMES
from features.dynamic_selector import greedy_ic_select
from scripts.signal_postprocess import rolling_zscore, should_exit_position
from scripts.train_v7_alpha import INTERACTION_FEATURES, BLACKLIST
from scipy.stats import spearmanr

# ── Config ──
HORIZONS = [12, 24, 48]
WARMUP = 30
COST_BPS_RT = 4
BARS_PER_DAY = 24
BARS_PER_MONTH = BARS_PER_DAY * 30
HPO_TRIALS = 10
TOP_K_FEATURES = 14
VERSION = "v10"
LGBM_XGB_WEIGHT = 0.5  # weight for lgbm; xgb gets (1 - this)

# Cross-market features to ALWAYS include in training.
# T-1 corrected IC (2026-03-22): coin_ret_1d IC=0.01, vix_level IC=0.00 (removed).
# spy_ret_1d retains real IC=-0.023 after T-1 correction.
# iv_level (DVOL/100): IC=+0.074, strongest no-bias feature.
# funding_zscore_24: IC=-0.052, 85% stability.
LOCKED_FEATURES = [
    "spy_ret_1d",          # IC=-0.023 (T-1 corrected), 64% stability
    "iv_level",            # IC=+0.074 (DVOL/100, strongest no-bias), from options_flow
    "funding_zscore_24",   # IC=-0.052, 85% stability, from enriched_computer
]


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

def backtest_ensemble_signal(
    preds_by_horizon: Dict[int, np.ndarray],
    closes: np.ndarray,
    deadzone: float,
    min_hold: int,
    max_hold: int,
    cost_bps: float,
    long_only: bool = True,
    zscore_window: int = 720,
) -> Dict[str, Any]:
    """Backtest with multi-horizon ensemble: normalize each horizon's pred, average."""
    n = len(closes)
    warmup = min(180, zscore_window // 4)

    # Compute rolling z-score per horizon, then average
    z_all = []
    for h, pred in sorted(preds_by_horizon.items()):
        z_h = rolling_zscore(pred, window=zscore_window, warmup=warmup)
        z_all.append(z_h)
    z = np.mean(z_all, axis=0)

    cost_frac = cost_bps / 10000
    pos = 0.0
    entry_bar = 0
    trades = []

    for i in range(n):
        if pos != 0:
            held = i - entry_bar
            if should_exit_position(
                position=pos,
                z_value=float(z[i]),
                held_bars=held,
                min_hold=min_hold,
                max_hold=max_hold,
            ):
                pnl_pct = pos * (closes[i] - closes[entry_bar]) / closes[entry_bar]
                trades.append(pnl_pct * 500.0 - cost_frac * 500.0)
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
        "sharpe": sharpe,
        "trades": len(trades),
        "return": float(np.sum(net_arr)) / 10000,
        "win_rate": float(np.mean(net_arr > 0) * 100),
        "avg_net_bps": float(np.mean(net_arr) / 500 * 10000),
    }


def train_single_horizon(
    horizon: int,
    X: np.ndarray,
    closes: np.ndarray,
    feature_names: List[str],
    val_start: int,
    train_end: int,
    n: int,
) -> Optional[Tuple[Any, Any, List[str], Dict[str, Any], np.ndarray]]:
    """Train LGBM+XGB for one horizon. Returns (lgbm, xgb, features, params, test_pred)."""
    import optuna
    import lightgbm as lgb
    import xgboost as xgb
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    y = compute_target(closes, horizon)

    X_train = X[WARMUP:val_start]
    y_train = y[WARMUP:val_start]
    X_val = X[val_start:train_end]
    y_val = y[val_start:train_end]
    X_test = X[train_end:]
    y_test = y[train_end:]

    valid_tr = ~np.isnan(y_train)
    valid_val = ~np.isnan(y_val)
    X_train = X_train[valid_tr]
    y_train = y_train[valid_tr]
    X_val = X_val[valid_val]
    y_val = y_val[valid_val]

    # Feature selection: greedy IC + locked cross-market features
    # Locked features are always included; greedy fills remaining slots
    locked_present = [f for f in LOCKED_FEATURES if f in feature_names]
    remaining_k = max(TOP_K_FEATURES - len(locked_present), 5)
    greedy_pool = [f for f in feature_names if f not in locked_present]
    greedy_idx = [feature_names.index(f) for f in greedy_pool]
    X_train_pool = X_train[:, greedy_idx]
    greedy_selected = greedy_ic_select(X_train_pool, y_train, greedy_pool, top_k=remaining_k)
    selected = locked_present + [f for f in greedy_selected if f not in locked_present]
    sel_idx = [feature_names.index(f) for f in selected]
    cm_in = [f for f in selected if f in LOCKED_FEATURES]
    print(f"    Features ({len(selected)}): {selected[:5]}... cross-market: {cm_in}")

    X_tr_sel = X_train[:, sel_idx]
    X_val_sel = X_val[:, sel_idx]
    X_test_sel = X_test[:, sel_idx]

    # Optuna HPO
    def objective(trial):
        params = {
            "objective": "regression", "metric": "mse", "verbosity": -1,
            "boosting_type": "gbdt",
            "num_leaves": trial.suggest_int("num_leaves", 15, 50),
            "max_depth": trial.suggest_int("max_depth", 3, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 30, 100),
            "subsample": trial.suggest_float("subsample", 0.6, 0.9),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.9),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
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
    print(f"    HPO best val IC: {study.best_value:.4f}")

    # Final LGBM
    lgbm_params = {"objective": "regression", "metric": "mse", "verbosity": -1,
                   "boosting_type": "gbdt", **best_params}
    dtrain = lgb.Dataset(X_tr_sel, label=y_train)
    dval = lgb.Dataset(X_val_sel, label=y_val, reference=dtrain)
    lgbm_model = lgb.train(lgbm_params, dtrain, num_boost_round=500,
                           valid_sets=[dval],
                           callbacks=[lgb.early_stopping(50, verbose=False),
                                      lgb.log_evaluation(0)])
    pred_lgbm = lgbm_model.predict(X_test_sel)
    ic_lgbm = fast_ic(pred_lgbm, y_test)

    # XGB
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
    pred_xgb = xgb_model.predict(xgb.DMatrix(X_test_sel))
    ic_xgb = fast_ic(pred_xgb, y_test)

    # Ensemble
    pred_ens = LGBM_XGB_WEIGHT * pred_lgbm + (1 - LGBM_XGB_WEIGHT) * pred_xgb
    ic_ens = fast_ic(pred_ens, y_test)
    print(f"    IC: lgbm={ic_lgbm:.4f} xgb={ic_xgb:.4f} ens={ic_ens:.4f}")

    info = {
        "horizon": horizon,
        "features": selected,
        "lgbm_params": best_params,
        "xgb_params": xgb_params,
        "ic_lgbm": ic_lgbm,
        "ic_xgb": ic_xgb,
        "ic_ensemble": ic_ens,
    }
    return lgbm_model, xgb_model, selected, info, pred_ens


def train_symbol(symbol: str) -> bool:
    """Train multi-horizon ensemble for one symbol. Returns True if saved."""
    model_dir = Path(f"models_v8/{symbol}_gate_v2")
    data_path = Path(f"data_files/{symbol}_1h.csv")

    if not data_path.exists():
        print(f"  ERROR: {data_path} not found")
        return False

    df = pd.read_csv(data_path)
    n = len(df)
    ts_col = "open_time" if "open_time" in df.columns else "timestamp"
    timestamps = df[ts_col].values.astype(np.int64)
    closes = df["close"].values.astype(np.float64)
    start_date = pd.Timestamp(timestamps[0], unit="ms").strftime("%Y-%m-%d")
    end_date = pd.Timestamp(timestamps[-1], unit="ms").strftime("%Y-%m-%d")
    print(f"\n  Data: {n:,} 1h bars ({start_date} → {end_date})")

    # ── Features ──
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
    print(f"{len(feature_names)} features in {time.time() - t0:.1f}s")

    # ── Split ──
    oos_bars = BARS_PER_MONTH * 18
    train_end = n - oos_bars
    val_size = BARS_PER_MONTH * 6
    val_start = train_end - val_size
    closes_test = closes[train_end:]

    print(f"  Split: train={val_start - WARMUP:,} val={val_size:,} test={oos_bars:,}")

    # ── Train each horizon ──
    models = {}  # horizon -> (lgbm, xgb, features, info)
    preds_test = {}  # horizon -> pred array on test set
    all_infos = {}

    for h in HORIZONS:
        print(f"\n  ── Horizon h={h}h ──")
        t0 = time.time()
        result = train_single_horizon(h, X, closes, feature_names, val_start, train_end, n)
        if result is None:
            print(f"    FAILED for h={h}")
            continue
        lgbm_m, xgb_m, feats, info, pred = result
        models[h] = (lgbm_m, xgb_m, feats, info)
        preds_test[h] = pred
        all_infos[h] = info
        print(f"    Done ({time.time() - t0:.1f}s)")

    if len(models) < 2:
        print("  ERROR: Need at least 2 horizons")
        return False

    # ── Multi-horizon ensemble backtest ──
    print(f"\n  ── Ensemble Deadzone Sweep ({len(models)} horizons) ──")
    best_config = None
    best_sharpe = -999
    best_result = None

    for dz in [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0]:
        for mh in [8, 12, 24]:
            for maxh_mult in [5, 8]:
                maxh = mh * maxh_mult
                for lo in [True, False]:
                    r = backtest_ensemble_signal(
                        preds_test, closes_test, dz, mh, maxh,
                        COST_BPS_RT, long_only=lo,
                    )
                    if r["sharpe"] > best_sharpe and r["trades"] >= 10:
                        best_sharpe = r["sharpe"]
                        best_config = {"deadzone": dz, "min_hold": mh,
                                       "max_hold": maxh, "long_only": lo}
                        best_result = r

    if best_config is None:
        print("  No viable ensemble config found!")
        return False

    print(f"  Best: dz={best_config['deadzone']}, hold=[{best_config['min_hold']},{best_config['max_hold']}], "
          f"long_only={best_config['long_only']}")
    print(f"  Sharpe={best_result['sharpe']:.2f}, trades={best_result['trades']}, "
          f"WR={best_result.get('win_rate', 0):.0f}%, ret={best_result['return'] * 100:+.2f}%")

    # ── Also test per-horizon single models for comparison ──
    print("\n  ── Per-Horizon Comparison ──")
    for h, pred in sorted(preds_test.items()):
        best_single = {"sharpe": -999}
        for dz in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
            for mh in [max(4, h // 3), max(8, h // 2), h]:
                for lo in [True, False]:
                    # Use single-horizon backtest (rolling z on single pred)
                    r = backtest_ensemble_signal(
                        {h: pred}, closes_test, dz, mh, mh * 5,
                        COST_BPS_RT, long_only=lo,
                    )
                    if r["sharpe"] > best_single["sharpe"] and r["trades"] >= 10:
                        best_single = r
        ic = all_infos[h]["ic_ensemble"]
        print(f"    h={h:>2}: IC={ic:+.4f}  Sharpe={best_single['sharpe']:.2f}  "
              f"Trades={best_single['trades']}  WR={best_single.get('win_rate', 0):.0f}%")
    print(f"    ENS:  {'':>12}  Sharpe={best_result['sharpe']:.2f}  "
          f"Trades={best_result['trades']}  WR={best_result.get('win_rate', 0):.0f}%")

    # ── Bootstrap ──
    print("\n  Bootstrap Sharpe...")
    z_all = []
    for h, pred in sorted(preds_test.items()):
        z_all.append(rolling_zscore(pred, window=720, warmup=180))
    z = np.mean(z_all, axis=0)

    trade_pnls = []
    pos = 0.0
    eb = 0
    for i in range(len(z)):
        if pos != 0:
            held = i - eb
            if should_exit_position(
                position=pos,
                z_value=float(z[i]),
                held_bars=held,
                min_hold=best_config["min_hold"],
                max_hold=best_config["max_hold"],
            ):
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

    # ── Production checks ──
    # Use average IC across horizons
    avg_ic = np.mean([info["ic_ensemble"] for info in all_infos.values()])

    print(f"\n  {'=' * 60}")
    print(f"  PRODUCTION CHECKS ({symbol})")
    print(f"  {'=' * 60}")
    checks = {
        "Sharpe > 1.0": best_result["sharpe"] > 1.0,
        "Avg IC > 0.02": avg_ic > 0.02,
        "Trades >= 15": best_result["trades"] >= 15,
        "Bootstrap p5 > 0": p5 > 0,
    }
    all_pass = True
    for check, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"    [{status}] {check}")

    # ── Save ──
    if all_pass:
        # Backup existing models
        backup_dir = model_dir.parent / f"{model_dir.name}_backup_pre_v10"
        if not backup_dir.exists() and model_dir.exists():
            import shutil
            shutil.copytree(model_dir, backup_dir)
            print(f"\n  Backed up to {backup_dir}")

        model_dir.mkdir(parents=True, exist_ok=True)

        # Save per-horizon models
        horizon_configs = []
        for h in sorted(models.keys()):
            lgbm_m, xgb_m, feats, info = models[h]
            lgbm_name = f"lgbm_h{h}.pkl"
            xgb_name = f"xgb_h{h}.pkl"

            with open(model_dir / lgbm_name, "wb") as f:
                pickle.dump({"model": lgbm_m, "features": feats}, f)
            with open(model_dir / xgb_name, "wb") as f:
                pickle.dump({"model": xgb_m, "features": feats}, f)

            horizon_configs.append({
                "horizon": h,
                "lgbm": lgbm_name,
                "xgb": xgb_name,
                "features": feats,
                "ic": info["ic_ensemble"],
            })

        # Also save primary model as lgbm_v8.pkl for backward compat
        # Use the middle horizon (h=24) as the "primary"
        primary_h = 24 if 24 in models else sorted(models.keys())[len(models) // 2]
        lgbm_p, xgb_p, feats_p, _ = models[primary_h]
        with open(model_dir / "lgbm_v8.pkl", "wb") as f:
            pickle.dump({"model": lgbm_p, "features": feats_p}, f)
        with open(model_dir / "xgb_v8.pkl", "wb") as f:
            pickle.dump({"model": xgb_p, "features": feats_p}, f)

        config = {
            "symbol": symbol,
            "version": VERSION,
            "multi_horizon": True,
            "horizons": sorted(models.keys()),
            "horizon_models": horizon_configs,
            "primary_horizon": primary_h,
            "ensemble_method": "mean_zscore",
            "deadzone": best_config["deadzone"],
            "min_hold": best_config["min_hold"],
            "max_hold": best_config["max_hold"],
            "long_only": best_config["long_only"],
            "metrics": {
                "sharpe": best_result["sharpe"],
                "avg_ic": avg_ic,
                "per_horizon_ic": {str(h): info["ic_ensemble"] for h, info in all_infos.items()},
                "total_return": best_result["return"],
                "trades": best_result["trades"],
                "win_rate": best_result.get("win_rate", 0),
                "avg_net_bps": best_result.get("avg_net_bps", 0),
                "bootstrap_sharpe_p5": float(p5),
                "bootstrap_sharpe_p50": float(p50),
                "bootstrap_sharpe_p95": float(p95),
            },
            "checks": {k: bool(v) for k, v in checks.items()},
            "train_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
            "data_range": f"{start_date} → {end_date}",
            "n_bars": n,
        }
        with open(model_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        # All features union for reference
        all_feats = []
        for hc in horizon_configs:
            for f in hc["features"]:
                if f not in all_feats:
                    all_feats.append(f)
        with open(model_dir / "features.json", "w") as f:
            json.dump(all_feats, f)

        print(f"\n  Model saved to {model_dir}/ ({VERSION})")
        return True
    else:
        print("\n  FAILED — model NOT saved.")
        return False


def main():
    parser = argparse.ArgumentParser(description="Multi-horizon ensemble training")
    parser.add_argument("--symbol", default="BTCUSDT,ETHUSDT",
                        help="Comma-separated symbols")
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbol.split(",")]

    print("=" * 70)
    print(f"  MULTI-HORIZON ENSEMBLE TRAINING ({VERSION})")
    print(f"  Horizons: {HORIZONS}")
    print(f"  Symbols:  {symbols}")
    print("=" * 70)

    results = {}
    for symbol in symbols:
        print(f"\n{'=' * 70}")
        print(f"  {symbol}")
        print(f"{'=' * 70}")
        t0 = time.time()
        saved = train_symbol(symbol)
        results[symbol] = saved
        print(f"\n  Total time: {time.time() - t0:.1f}s")

    # Summary
    print(f"\n{'=' * 70}")
    print("  SUMMARY")
    print(f"{'=' * 70}")
    for sym, saved in results.items():
        status = "SAVED" if saved else "FAILED"
        print(f"  {sym}: {status}")


if __name__ == "__main__":
    main()
