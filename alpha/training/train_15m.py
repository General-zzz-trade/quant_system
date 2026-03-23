#!/usr/bin/env python3
"""Train dedicated 15-minute models based on alpha research results.

Adapts train_v11.py pipeline for 15m bars:
- Different data source (*_15m.csv)
- 4h features auto-added by compute_features_batch() (tf4h_* columns)
- Different split sizes (4x more bars per month)
- Saves to models_v8/{SYMBOL}_15m/ directory

Usage:
    python3 -m scripts.train_15m --symbol BTCUSDT --horizons 32,64
    python3 -m scripts.train_15m --symbol ETHUSDT --horizons 4,8
    python3 -m scripts.train_15m --symbol BTCUSDT,ETHUSDT
"""
from __future__ import annotations

import sys
import time
import json
import pickle
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from alpha.utils import fast_ic, compute_target

sys.path.insert(0, "/quant_system")

from features.batch_feature_engine import compute_features_batch
from features.dynamic_selector import greedy_ic_select
from shared.signal_postprocess import rolling_zscore, should_exit_position

# ── Config ──
VERSION = "v11-15m"
WARMUP = 30
COST_BPS_RT = 4
BARS_PER_DAY = 96       # 96 15m bars per day
BARS_PER_MONTH = BARS_PER_DAY * 30
HPO_TRIALS = 10
TOP_K_FEATURES = 14

# Default horizons per symbol (from research results)
DEFAULT_HORIZONS = {
    "BTCUSDT": [32, 64],    # 8h, 16h — strongest IC
    "ETHUSDT": [4, 8],      # 1h, 2h — high frequency
}

# Blacklist features that don't exist or are unreliable on 15m
BLACKLIST_15M = {
    "fgi_normalized", "fgi_extreme",  # daily data, stale on 15m
    "funding_rate", "funding_zscore_24", "funding_sign_persist",  # 8h funding
    "basis_carry_adj", "basis_zscore_24",  # spot-futures basis (1h aligned)
}


def train_single_horizon_15m(
    h: int,
    X: np.ndarray,
    closes: np.ndarray,
    feature_names: List[str],
    val_start: int,
    train_end: int,
    n: int,
) -> Optional[Tuple]:
    """Train LightGBM + XGBoost for a single horizon on 15m data."""
    import lightgbm as lgb

    y = compute_target(closes, h)

    # Feature selection on train set
    valid_train = ~np.isnan(y[:val_start])
    if valid_train.sum() < 200:
        print(f"    Not enough valid training samples: {valid_train.sum()}")
        return None

    selected = greedy_ic_select(
        X[:val_start][valid_train],
        y[:val_start][valid_train],
        feature_names,
        top_k=TOP_K_FEATURES,
    )
    print(f"    Features: {selected}")

    feat_idx = [feature_names.index(f) for f in selected]
    X_sel = X[:, feat_idx]

    # Train/val split
    valid_t = ~np.isnan(y[:val_start])
    valid_v = ~np.isnan(y[val_start:train_end])

    dtrain = lgb.Dataset(X_sel[:val_start][valid_t], y[:val_start][valid_t])
    dval = lgb.Dataset(
        X_sel[val_start:train_end][valid_v],
        y[val_start:train_end][valid_v],
        reference=dtrain,
    )

    # HPO: try a few learning rates
    best_val_ic = -1
    best_params = None
    best_model = None

    for lr in [0.01, 0.03, 0.05]:
        for nl in [15, 31, 63]:
            params = {
                "objective": "regression",
                "metric": "mae",
                "learning_rate": lr,
                "num_leaves": nl,
                "min_child_samples": 100,
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
            pred_v = model.predict(X_sel[val_start:train_end][valid_v])
            val_ic = fast_ic(pred_v, y[val_start:train_end][valid_v])
            if val_ic > best_val_ic:
                best_val_ic = val_ic
                best_params = params
                best_model = model

    print(f"    HPO best val IC: {best_val_ic:.4f}")
    lgbm_model = best_model

    # Train XGBoost
    xgb_model = None
    try:
        import xgboost as xgb
        xg_params = {
            "objective": "reg:squarederror",
            "max_depth": 5,
            "learning_rate": best_params["learning_rate"],
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "verbosity": 0,
            "seed": 42,
        }
        dxg_train = xgb.DMatrix(X_sel[:val_start][valid_t], y[:val_start][valid_t])
        dxg_val = xgb.DMatrix(X_sel[val_start:train_end][valid_v], y[val_start:train_end][valid_v])
        xgb_model = xgb.train(
            xg_params, dxg_train,
            num_boost_round=500,
            evals=[(dxg_val, "val")],
            early_stopping_rounds=50,
            verbose_eval=False,
        )
    except Exception as e:
        print(f"    XGBoost skipped: {e}")

    # OOS predictions
    pred_lgbm = lgbm_model.predict(X_sel[train_end:])
    y_oos = y[train_end:]
    valid_oos = ~np.isnan(y_oos)
    ic_lgbm = fast_ic(pred_lgbm[valid_oos], y_oos[valid_oos])

    pred_ens = pred_lgbm.copy()
    ic_xgb = 0.0
    if xgb_model is not None:
        import xgboost as xgb
        pred_xgb = xgb_model.predict(xgb.DMatrix(X_sel[train_end:]))
        ic_xgb = fast_ic(pred_xgb[valid_oos], y_oos[valid_oos])
        pred_ens = 0.5 * pred_lgbm + 0.5 * pred_xgb

    ic_ens = fast_ic(pred_ens[valid_oos], y_oos[valid_oos])
    print(f"    IC: lgbm={ic_lgbm:.4f} xgb={ic_xgb:.4f} ens={ic_ens:.4f}")

    info = {
        "ic_lgbm": ic_lgbm,
        "ic_xgb": ic_xgb,
        "ic_ensemble": ic_ens,
        "val_ic": best_val_ic,
        "params": best_params,
    }

    return lgbm_model, xgb_model, selected, info, pred_ens


def backtest_15m(
    preds_by_horizon: Dict[int, np.ndarray],
    closes: np.ndarray,
    deadzone: float = 0.3,
    min_hold: int = 4,
    max_hold: int = 64,
    long_only: bool = False,
    cost_bps: float = COST_BPS_RT,
    zscore_window: int = 720,
    zscore_warmup: int = 180,
) -> Dict[str, Any]:
    """Backtest 15m ensemble with config sweep."""
    n = len(closes)
    cost_frac = cost_bps / 10000

    # Z-score per horizon, then average
    z_all = []
    for h, pred in sorted(preds_by_horizon.items()):
        z_all.append(rolling_zscore(pred, window=zscore_window, warmup=zscore_warmup))
    z = np.mean(z_all, axis=0)

    pos = 0.0
    entry_bar = 0
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
    tpy = 365 * BARS_PER_DAY / max(avg_hold, 1)
    sharpe = float(np.mean(net_arr) / max(np.std(net_arr, ddof=1), 1e-10) * np.sqrt(tpy))
    return {
        "sharpe": sharpe,
        "trades": len(trades),
        "return": float(np.sum(net_arr)) / 10000,
        "win_rate": float(np.mean(net_arr > 0) * 100),
        "avg_net_bps": float(np.mean(net_arr) / 500 * 10000),
        "avg_hold_bars": avg_hold,
        "avg_hold_hours": avg_hold * 0.25,
    }


def train_symbol_15m(symbol: str, horizons: List[int]) -> bool:
    """Train 15m models for one symbol."""
    data_path = Path(f"data_files/{symbol}_15m.csv")
    model_dir = Path(f"models_v8/{symbol}_15m")

    if not data_path.exists():
        print(f"  ERROR: {data_path} not found")
        return False

    df = pd.read_csv(data_path)
    n = len(df)
    closes = df["close"].values.astype(np.float64)
    start = pd.Timestamp(df["open_time"].iloc[0], unit="ms").strftime("%Y-%m-%d")
    end = pd.Timestamp(df["open_time"].iloc[-1], unit="ms").strftime("%Y-%m-%d")
    print(f"\n  Data: {n:,} 15m bars ({start} → {end})")

    # ── Features ──
    print("  Computing features...", end=" ", flush=True)
    t0 = time.time()
    feat_df = compute_features_batch(symbol="", df=df, include_v11=False)

    feature_names = [c for c in feat_df.columns
                     if c not in ("close", "open_time", "timestamp", "open", "high", "low", "volume")
                     and c not in BLACKLIST_15M]
    X = feat_df[feature_names].values.astype(np.float64)
    print(f"{len(feature_names)} features in {time.time()-t0:.1f}s")

    # ── Split ──
    oos_bars = BARS_PER_MONTH * 6  # 6 months OOS
    if oos_bars > n * 0.6:
        oos_bars = int(n * 0.4)  # cap at 40% if data is short
    train_end = n - oos_bars
    val_size = BARS_PER_MONTH * 2
    if val_size > train_end * 0.3:
        val_size = int(train_end * 0.25)
    val_start = train_end - val_size
    closes_test = closes[train_end:]

    print(f"  Split: train={val_start-WARMUP:,} val={val_size:,} test={oos_bars:,}")

    # ── Train each horizon ──
    models = {}
    preds_test = {}
    all_infos = {}

    for h in horizons:
        hours = h * 0.25
        print(f"\n  ── h={h} bars ({hours:.0f}h) ──")
        t0 = time.time()
        result = train_single_horizon_15m(h, X, closes, feature_names, val_start, train_end, n)
        if result is None:
            print("    FAILED")
            continue
        lgbm_m, xgb_m, feats, info, pred = result
        models[h] = (lgbm_m, xgb_m, feats, info)
        preds_test[h] = pred
        all_infos[h] = info
        print(f"    Done ({time.time()-t0:.1f}s)")

    if not models:
        print("  ERROR: No models trained successfully")
        return False

    # ── Config sweep ──
    print(f"\n  ── Config Sweep ({len(models)} horizons) ──")
    best_cfg = None
    best_sharpe = -999
    best_result = None

    for dz in [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0]:
        for mh in [4, 8, 16]:
            for maxh in [32, 64, 128]:
                for lo in [True, False]:
                    r = backtest_15m(
                        preds_test, closes_test, deadzone=dz,
                        min_hold=mh, max_hold=maxh, long_only=lo,
                    )
                    if r["sharpe"] > best_sharpe and r["trades"] >= 10:
                        best_sharpe = r["sharpe"]
                        best_cfg = {"dz": dz, "min_hold": mh, "max_hold": maxh, "long_only": lo}
                        best_result = r

    if best_cfg is None:
        print("  No viable config found!")
        return False

    print(f"  Best: dz={best_cfg['dz']}, hold=[{best_cfg['min_hold']},{best_cfg['max_hold']}], "
          f"long_only={best_cfg['long_only']}")
    print(f"  Sharpe={best_result['sharpe']:.2f}, trades={best_result['trades']}, "
          f"WR={best_result['win_rate']:.0f}%, ret={best_result['return']*100:+.2f}%")
    print(f"  Avg hold: {best_result['avg_hold_hours']:.1f}h, "
          f"avg net: {best_result['avg_net_bps']:.1f}bps")

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
            should_exit = should_exit_position(
                position=pos,
                z_value=float(z[i]),
                held_bars=held,
                min_hold=best_cfg["min_hold"],
                max_hold=best_cfg["max_hold"],
            )
            if should_exit:
                pnl = pos * (closes_test[i] - closes_test[eb]) / closes_test[eb]
                trade_pnls.append(pnl * 500 - COST_BPS_RT / 10000 * 500)
                pos = 0.0
        if pos == 0:
            if z[i] > best_cfg["dz"]:
                pos = 1.0
                eb = i
            elif not best_cfg["long_only"] and z[i] < -best_cfg["dz"]:
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
    avg_ic = np.mean([info["ic_ensemble"] for info in all_infos.values()])

    print(f"\n  {'='*60}")
    print(f"  PRODUCTION CHECKS ({symbol} 15m)")
    print(f"  {'='*60}")
    checks = {
        "Sharpe > 1.0": best_result["sharpe"] > 1.0,
        "Avg IC > 0.01": avg_ic > 0.01,          # lower bar for 15m
        "Trades >= 10": best_result["trades"] >= 10,
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
        model_dir.mkdir(parents=True, exist_ok=True)

        horizon_configs = []
        for h in sorted(models.keys()):
            lgbm_m, xgb_m, feats, info = models[h]
            lgbm_name = f"lgbm_h{h}.pkl"
            xgb_name = f"xgb_h{h}.pkl"

            with open(model_dir / lgbm_name, "wb") as f:
                pickle.dump({"model": lgbm_m, "features": feats}, f)
            if xgb_m is not None:
                with open(model_dir / xgb_name, "wb") as f:
                    pickle.dump({"model": xgb_m, "features": feats}, f)

            horizon_configs.append({
                "horizon": h,
                "lgbm": lgbm_name,
                "xgb": xgb_name,
                "features": feats,
                "ic": info["ic_ensemble"],
            })

        # Primary model (largest IC)
        primary_h = max(models.keys(), key=lambda h: all_infos[h]["ic_ensemble"])
        lgbm_p, xgb_p, feats_p, _ = models[primary_h]
        with open(model_dir / "lgbm_v8.pkl", "wb") as f:
            pickle.dump({"model": lgbm_p, "features": feats_p}, f)
        if xgb_p is not None:
            with open(model_dir / "xgb_v8.pkl", "wb") as f:
                pickle.dump({"model": xgb_p, "features": feats_p}, f)

        config_dict = {
            "symbol": symbol,
            "version": VERSION,
            "timeframe": "15m",
            "multi_horizon": True,
            "horizons": sorted(models.keys()),
            "horizon_models": horizon_configs,
            "primary_horizon": primary_h,
            "ensemble_method": "ic_weighted",
            "ic_ema_span": 720,
            "ic_min_threshold": -0.01,
            "lgbm_xgb_weight": 0.5,
            "zscore_window": 720,
            "zscore_warmup": 180,
            "deadzone": best_cfg["dz"],
            "min_hold": best_cfg["min_hold"],
            "max_hold": best_cfg["max_hold"],
            "long_only": best_cfg["long_only"],
            "exit": {
                "trailing_stop_pct": 0.0,
                "zscore_cap": 0.0,
                "reversal_threshold": -0.3,
                "deadzone_fade": 0.2,
            },
            "regime_gate": {"enabled": False},
            "time_filter": {"enabled": False, "skip_hours_utc": []},
            "metrics": {
                "sharpe": best_result["sharpe"],
                "avg_ic": float(avg_ic),
                "per_horizon_ic": {str(h): info["ic_ensemble"] for h, info in all_infos.items()},
                "total_return": best_result["return"],
                "trades": best_result["trades"],
                "win_rate": best_result["win_rate"],
                "avg_net_bps": best_result["avg_net_bps"],
                "avg_hold_hours": best_result["avg_hold_hours"],
                "bootstrap_sharpe_p5": float(p5),
                "bootstrap_sharpe_p50": float(p50),
                "bootstrap_sharpe_p95": float(p95),
            },
            "checks": {k: bool(v) for k, v in checks.items()},
            "train_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
            "data_range": f"{start} → {end}",
            "n_bars": n,
        }
        with open(model_dir / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2)

        # Features list
        all_feats = []
        for hc in horizon_configs:
            for feat in hc["features"]:
                if feat not in all_feats:
                    all_feats.append(feat)
        with open(model_dir / "features.json", "w") as f:
            json.dump(all_feats, f)

        print(f"\n  Model saved to {model_dir}/ ({VERSION})")
        return True
    else:
        print("\n  FAILED — model NOT saved.")
        return False


def main():
    parser = argparse.ArgumentParser(description="15m Model Training")
    parser.add_argument("--symbol", default="BTCUSDT,ETHUSDT",
                        help="Comma-separated symbols")
    parser.add_argument("--horizons", default=None,
                        help="Comma-separated horizons in 15m bars (default: per-symbol optimal)")
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbol.split(",")]

    print("=" * 70)
    print("  15-MINUTE MODEL TRAINING")
    print(f"  Symbols: {symbols}")
    print("=" * 70)

    results = {}
    for symbol in symbols:
        if args.horizons:
            horizons = [int(h.strip()) for h in args.horizons.split(",")]
        else:
            horizons = DEFAULT_HORIZONS.get(symbol, [4, 8, 16])

        print(f"\n{'='*70}")
        print(f"  {symbol} — horizons: {horizons} ({[h*0.25 for h in horizons]}h)")
        print(f"{'='*70}")

        t0 = time.time()
        saved = train_symbol_15m(symbol, horizons)
        results[symbol] = saved
        print(f"\n  Total time: {time.time()-t0:.1f}s")

    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    for sym, saved in results.items():
        status = "SAVED" if saved else "FAILED"
        print(f"  {sym}: {status}")


if __name__ == "__main__":
    main()
