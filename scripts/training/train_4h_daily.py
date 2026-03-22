#!/usr/bin/env python3
"""Train 4h and daily alpha models using the V11 pipeline.

Resamples 1h data → 4h / daily, computes features, trains Ridge+LGBM
ensemble with walk-forward validation.

Note: Uses pickle for model serialization (sklearn/lightgbm standard).

Usage:
    python3 -m scripts.training.train_4h_daily --symbol BTCUSDT --interval 4h
    python3 -m scripts.training.train_4h_daily --symbol ETHUSDT --interval 1d
    python3 -m scripts.training.train_4h_daily --all
"""
from __future__ import annotations

import sys
import time
import json
import pickle  # noqa: S403 — required for sklearn/lightgbm model serialization
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

sys.path.insert(0, "/quant_system")

from features.batch_feature_engine import compute_features_batch
from scripts.train_multi_horizon import (
    rolling_zscore,
    train_single_horizon,
    compute_target,
    WARMUP,
    COST_BPS_RT,
    LOCKED_FEATURES,
    TOP_K_FEATURES,
)
from features.dynamic_selector import greedy_ic_select
from scipy.stats import spearmanr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CROSS_MARKET_PATH = Path("data_files/cross_market_daily.csv")

INTERVAL_CONFIG = {
    "4h": {
        "resample_rule": "4h",
        "bars_per_day": 6,
        "default_horizons": {"BTCUSDT": [6, 12, 24], "ETHUSDT": [6, 12]},
        "dz_sweep": [0.3, 0.5, 0.8, 1.0, 1.5, 2.0],
        "mh_sweep": [3, 6, 12],
        "maxh_mult": [4, 6, 8],
        "zscore_window": 180,
        "zscore_warmup": 45,
        "sma_window": 120,
        "oos_days": 180,
        "val_days": 90,
        "model_suffix": "_4h",
    },
    "1d": {
        "resample_rule": "1D",
        "bars_per_day": 1,
        "default_horizons": {"BTCUSDT": [1, 2, 5], "ETHUSDT": [1, 2, 3]},
        "dz_sweep": [0.3, 0.5, 0.8, 1.0],
        "mh_sweep": [1, 2, 3],
        "maxh_mult": [3, 5, 10],
        "zscore_window": 60,
        "zscore_warmup": 20,
        "sma_window": 20,
        "oos_days": 365,
        "val_days": 180,
        "model_suffix": "_1d",
    },
}


def resample_1h(df_1h: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample 1h OHLCV data to target interval."""
    df = df_1h.copy()
    df["datetime"] = pd.to_datetime(df["open_time"], unit="ms")
    df = df.set_index("datetime")

    agg = {"open_time": "first", "open": "first", "high": "max",
           "low": "min", "close": "last", "volume": "sum"}
    for col in ["quote_volume", "taker_buy_volume", "taker_buy_quote_volume", "trades"]:
        if col in df.columns:
            agg[col] = "sum"

    out = df.resample(rule).agg(agg).dropna(subset=["close"])
    return out.reset_index(drop=True)


def add_cross_market_features(feat_df: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Merge cross-market daily features."""
    if not CROSS_MARKET_PATH.exists():
        logger.warning("Cross-market data not found: %s", CROSS_MARKET_PATH)
        return feat_df

    cm = pd.read_csv(CROSS_MARKET_PATH, parse_dates=["date"])
    cm["date"] = cm["date"].dt.date
    bar_dates = pd.to_datetime(df["open_time"], unit="ms").dt.date
    cm_indexed = cm.set_index("date")
    # T-1 shift: bar on date D uses cross-market data from D-1 (prevents look-ahead)
    cm_indexed.index = [d + pd.Timedelta(days=1) for d in cm_indexed.index]

    for col in cm_indexed.columns:
        mapped = bar_dates.map(lambda d, c=col: cm_indexed[c].get(d, np.nan))
        feat_df[col] = mapped.ffill().values

    logger.info("Added %d cross-market features", len(cm_indexed.columns))
    return feat_df


def _train_ridge_horizon(
    horizon: int,
    X: np.ndarray,
    closes: np.ndarray,
    feature_names: list[str],
    val_start: int,
    train_end: int,
    n: int,
):
    """Train Ridge-only for one horizon. Returns same tuple format as train_single_horizon.

    Ridge OOS IC=0.035 vs LGBM IC=-0.022 on 4h data — LGBM overfits.
    Returns (lgbm_m, ridge_m, features, info, test_pred) where lgbm_m=ridge_m.
    """
    y = compute_target(closes, horizon)

    X_train = X[WARMUP:val_start]
    y_train = y[WARMUP:val_start]
    X_val = X[val_start:train_end]
    y_val = y[val_start:train_end]
    X_test = X[train_end:]

    valid_tr = ~np.isnan(y_train)
    valid_val = ~np.isnan(y_val)
    X_train = X_train[valid_tr]
    y_train = y_train[valid_tr]
    X_val = X_val[valid_val]
    y_val = y_val[valid_val]

    if len(X_train) < 100 or len(X_val) < 20:
        return None

    # Feature selection: greedy IC + locked cross-market features
    locked_present = [f for f in LOCKED_FEATURES if f in feature_names]
    remaining_k = max(TOP_K_FEATURES - len(locked_present), 5)
    greedy_pool = [f for f in feature_names if f not in locked_present]
    greedy_idx = [feature_names.index(f) for f in greedy_pool]
    X_train_pool = X_train[:, greedy_idx]
    greedy_selected = greedy_ic_select(X_train_pool, y_train, greedy_pool, top_k=remaining_k)
    selected = locked_present + [f for f in greedy_selected if f not in locked_present]
    sel_idx = [feature_names.index(f) for f in selected]
    logger.info("    Ridge features (%d): %s...", len(selected), selected[:5])

    X_tr_sel = X_train[:, sel_idx]
    X_val_sel = X_val[:, sel_idx]
    X_test_sel = X_test[:, sel_idx]

    # Replace NaN with 0 for Ridge
    X_tr_sel = np.nan_to_num(X_tr_sel, nan=0.0)
    X_val_sel = np.nan_to_num(X_val_sel, nan=0.0)
    X_test_sel = np.nan_to_num(X_test_sel, nan=0.0)

    # Train Ridge(alpha=1.0)
    ridge_m = Ridge(alpha=1.0)
    ridge_m.fit(X_tr_sel, y_train)

    # Evaluate on validation set
    val_pred = ridge_m.predict(X_val_sel)
    ic_val, _ = spearmanr(val_pred, y_val)
    ic_val = float(ic_val) if not np.isnan(ic_val) else 0.0

    # Test predictions
    test_pred = ridge_m.predict(X_test_sel)

    info = {
        "ic_ridge": ic_val,
        "ic_ensemble": ic_val,  # backward compat
        "model_type": "ridge_only",
        "alpha": 1.0,
        "n_features": len(selected),
    }

    # Return: (lgbm_m, xgb_m, feats, info, pred)
    # For Ridge-only, both slots hold the Ridge model
    return ridge_m, ridge_m, selected, info, test_pred


def train_symbol(symbol: str, interval: str, dry_run: bool = False) -> dict:
    """Train a model for one symbol at the specified interval."""
    cfg = INTERVAL_CONFIG[interval]
    model_dir = Path(f"models_v8/{symbol}{cfg['model_suffix']}")
    data_path = Path(f"data_files/{symbol}_1h.csv")

    if not data_path.exists():
        logger.error("%s not found", data_path)
        return {"status": "error", "reason": "data_missing"}

    logger.info("Loading %s...", data_path)
    df_1h = pd.read_csv(data_path)
    df = resample_1h(df_1h, cfg["resample_rule"])
    n = len(df)
    bars_per_day = cfg["bars_per_day"]
    n_days = n / bars_per_day
    logger.info("%s %s: %d bars (%.0f days)", symbol, interval, n, n_days)

    if n < 500:
        logger.error("Insufficient data: %d bars", n)
        return {"status": "error", "reason": "insufficient_data"}

    # Features
    logger.info("Computing features...")
    t0 = time.time()
    feat_df = compute_features_batch(symbol, df)
    feat_df = add_cross_market_features(feat_df, df)

    # Interaction features
    for name, fa, fb in [("vol_x_taker", "vol_5", "taker_buy_ratio"),
                          ("rsi_x_vol", "rsi_14", "vol_ratio"),
                          ("ret5_x_volume", "ret_5", "volume_ratio")]:
        if fa in feat_df.columns and fb in feat_df.columns:
            feat_df[name] = feat_df[fa].astype(float) * feat_df[fb].astype(float)

    blacklist = {"close", "open_time", "timestamp", "open", "high", "low",
                 "volume", "quote_volume", "ignore", "close_time"}
    feature_names = [c for c in feat_df.columns if c not in blacklist]
    X = feat_df[feature_names].values.astype(np.float64)
    logger.info("Features: %d in %.1fs", len(feature_names), time.time() - t0)

    # Split
    oos_bars = int(cfg["oos_days"] * bars_per_day)
    val_bars = int(cfg["val_days"] * bars_per_day)
    train_end = n - oos_bars
    val_start = train_end - val_bars

    if val_start < WARMUP + 100:
        logger.error("Not enough training data")
        return {"status": "error", "reason": "insufficient_train"}

    closes = df["close"].values.astype(np.float64)
    logger.info("Split: train=%d, val=%d, test=%d", val_start - WARMUP, val_bars, oos_bars)

    # Train horizons
    horizons = cfg["default_horizons"].get(symbol, [6, 12])
    models = {}
    preds_test = {}
    all_infos = {}

    # 4h: Ridge-only (OOS IC=0.035 vs LGBM IC=-0.022; LGBM overfits on 4h)
    use_ridge_only = False  # Ridge-only underperforms LGBM in full backtest
    if use_ridge_only:
        logger.info("Using Ridge-only for 4h (LGBM overfits on low-frequency data)")

    for h in horizons:
        real_time = f"{h * 24 // bars_per_day}h" if bars_per_day > 1 else f"{h}d"
        logger.info("Training h=%d (%s)...", h, real_time)
        t0 = time.time()

        if use_ridge_only:
            result = _train_ridge_horizon(
                h, X, closes, feature_names, val_start, train_end, n,
            )
        else:
            result = train_single_horizon(h, X, closes, feature_names, val_start, train_end, n)

        if result is None:
            logger.warning("h=%d FAILED", h)
            continue
        lgbm_m, xgb_m, feats, info, pred = result
        models[h] = (lgbm_m, xgb_m, feats, info)
        preds_test[h] = pred
        all_infos[h] = info
        ic_key = "ic_ridge" if use_ridge_only else "ic_ensemble"
        logger.info("h=%d: IC=%.4f, %d features (%.1fs)",
                     h, info.get(ic_key, info.get("ic_ensemble", 0)),
                     len(feats), time.time() - t0)

    if not models:
        logger.error("No horizons succeeded")
        return {"status": "error", "reason": "all_horizons_failed"}

    # Config sweep
    logger.info("Signal parameter sweep (%d horizons)...", len(models))
    best_sharpe = -999
    best_config = {}
    best_result = {}
    closes_test = closes[train_end:]
    z_window = cfg["zscore_window"]
    z_warmup = cfg["zscore_warmup"]

    for dz in cfg["dz_sweep"]:
        for mh in cfg["mh_sweep"]:
            for maxh_m in cfg["maxh_mult"]:
                maxh = mh * maxh_m
                for lo in [True, False]:
                    ic_weights = {h: max(0.01, abs(info.get("ic_ensemble", 0.01)))
                                  for h, info in all_infos.items()}
                    total_ic = sum(ic_weights.values())

                    pred_ens = np.zeros(len(closes_test))
                    for h, pred in preds_test.items():
                        pred_ens += pred * (ic_weights[h] / total_ic)

                    z = rolling_zscore(pred_ens, window=z_window, warmup=z_warmup)
                    sig = np.zeros(len(z))
                    sig[z > dz] = 1
                    sig[z < -dz] = -1

                    if lo:
                        sig[sig < 0] = 0

                    # Monthly gate (BTC only)
                    if symbol == "BTCUSDT":
                        sma_w = cfg["sma_window"]
                        sma = pd.Series(closes_test).rolling(sma_w, min_periods=sma_w // 2).mean().values
                        for i in range(len(sig)):
                            if not np.isnan(sma[i]) and closes_test[i] < sma[i]:
                                sig[i] = min(sig[i], 0)

                    # Min hold
                    cur, hold = 0, 0
                    for i in range(len(sig)):
                        s = sig[i]
                        if s != cur and s != 0:
                            cur, hold = s, 1
                        elif cur != 0 and hold < mh:
                            sig[i] = cur
                            hold += 1
                        elif s != cur:
                            cur = s
                            hold = 1 if s != 0 else 0

                    # Max hold
                    cur, hold = 0, 0
                    for i in range(len(sig)):
                        if sig[i] != 0:
                            if sig[i] == cur:
                                hold += 1
                                if hold > maxh:
                                    sig[i], cur, hold = 0, 0, 0
                            else:
                                cur, hold = sig[i], 1
                        else:
                            cur, hold = 0, 0

                    # Backtest
                    pnl = np.array([
                        sig[i - 1] * (closes_test[i] / closes_test[i - 1] - 1)
                        if sig[i - 1] != 0 else 0.0
                        for i in range(1, len(sig))
                    ])
                    trades = int(np.sum(np.abs(np.diff(sig)) > 0))
                    cost = trades * COST_BPS_RT / 10000  # COST_BPS_RT=4 is in bps

                    if np.std(pnl) > 0 and len(pnl) > 10:
                        ann = np.sqrt(bars_per_day * 365)
                        sharpe = (np.mean(pnl) - cost / len(pnl)) / np.std(pnl) * ann
                    else:
                        sharpe = 0

                    wr = np.mean(pnl[pnl != 0] > 0) * 100 if np.any(pnl != 0) else 0

                    if sharpe > best_sharpe:
                        best_sharpe = sharpe
                        best_config = {"deadzone": dz, "min_hold": mh, "max_hold": maxh,
                                       "long_only": lo, "monthly_gate": (symbol == "BTCUSDT")}
                        best_result = {"sharpe": round(sharpe, 2),
                                       "total_ret": round((pnl.sum() - cost) * 100, 1),
                                       "trades": trades, "win_rate": round(wr, 1)}

    logger.info("Best: dz=%.1f mh=%d maxh=%d lo=%s → Sharpe=%.2f trades=%d",
                best_config.get("deadzone", 0), best_config.get("min_hold", 0),
                best_config.get("max_hold", 0), best_config.get("long_only"),
                best_sharpe, best_result.get("trades", 0))

    if dry_run:
        logger.info("DRY RUN — not saving")
        return {"status": "dry_run", "config": best_config, "result": best_result}

    # Save
    model_dir.mkdir(parents=True, exist_ok=True)
    horizon_configs = []
    for h, (lgbm_m, xgb_m, feats, info) in models.items():
        if use_ridge_only:
            # 4h Ridge-only: save Ridge as both ridge and lgb (backward compat)
            ridge_path = model_dir / f"ridge_h{h}.pkl"
            lgbm_path = model_dir / f"lgb_h{h}.pkl"
            with open(ridge_path, "wb") as f:
                pickle.dump(xgb_m, f)  # xgb_m slot holds Ridge model
            with open(lgbm_path, "wb") as f:
                pickle.dump(xgb_m, f)  # save Ridge as lgb too (backward compat)
            ic_val = info.get("ic_ridge", info.get("ic_ensemble", 0))
            horizon_configs.append({
                "horizon": h, "lgbm": lgbm_path.name,
                "ridge": ridge_path.name,
                "features": feats, "ic": round(ic_val, 4),
            })
        else:
            lgbm_path = model_dir / f"lgb_h{h}.pkl"
            xgb_path = model_dir / f"xgb_h{h}.pkl"
            with open(lgbm_path, "wb") as f:
                pickle.dump(lgbm_m, f)
            with open(xgb_path, "wb") as f:
                pickle.dump(xgb_m, f)
            horizon_configs.append({
                "horizon": h, "lgbm": lgbm_path.name, "xgb": xgb_path.name,
                "features": feats, "ic": round(info.get("ic_ensemble", 0), 4),
            })

    # Backward compat: save primary ridge/lgb
    first_h = sorted(models.keys())[0]
    lgbm_first, xgb_first, feats_first, _ = models[first_h]
    with open(model_dir / "ridge_model.pkl", "wb") as f:
        pickle.dump(xgb_first, f)
    with open(model_dir / "lgb_model.pkl", "wb") as f:
        pickle.dump(lgbm_first if not use_ridge_only else xgb_first, f)

    ensemble_method = "ridge_only" if use_ridge_only else "ic_weighted"
    config = {
        "symbol": symbol, "interval": interval,
        "version": f"v11-{interval}",
        "multi_horizon": True,
        "horizons": sorted(models.keys()),
        "horizon_models": horizon_configs,
        "ensemble_method": ensemble_method,
        "features": feats_first,
        "zscore_window": z_window, "zscore_warmup": z_warmup,
        **best_config,
        "metrics": best_result,
        "train_date": time.strftime("%Y-%m-%d %H:%M"),
        "train_bars": n, "train_days": round(n_days),
    }
    with open(model_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    logger.info("Saved to %s", model_dir)
    return {"status": "ok", "model_dir": str(model_dir),
            "config": best_config, "result": best_result}


def main():
    parser = argparse.ArgumentParser(description="Train 4h/daily alpha models")
    parser.add_argument("--symbol", type=str, default=None)
    parser.add_argument("--interval", type=str, choices=["4h", "1d"], default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    symbols = [args.symbol] if args.symbol else ["BTCUSDT", "ETHUSDT"]
    intervals = [args.interval] if args.interval else ["4h", "1d"]

    results = {}
    for interval in intervals:
        for symbol in symbols:
            logger.info("\n%s Training %s %s %s", "=" * 20, symbol, interval, "=" * 20)
            r = train_symbol(symbol, interval, dry_run=args.dry_run)
            results[f"{symbol}_{interval}"] = r

    print(f"\n{'='*60}")
    print(f"  Training Summary")
    print(f"{'='*60}")
    for key, r in results.items():
        status = r.get("status", "?")
        if status in ("ok", "dry_run"):
            cfg = r.get("config", {})
            res = r.get("result", {})
            print(f"  {key:20s} {status:8s} Sharpe={res.get('sharpe', 0):6.2f} "
                  f"dz={cfg.get('deadzone', 0):.1f} mh={cfg.get('min_hold', 0)} "
                  f"trades={res.get('trades', 0)}")
        else:
            print(f"  {key:20s} {status:8s} reason={r.get('reason', '?')}")


if __name__ == "__main__":
    main()
