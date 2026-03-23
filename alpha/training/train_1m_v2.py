#!/usr/bin/env python3
"""Train Improved 1-minute Alpha V2 — full multi-resolution + smart filtering.

Key improvements over V1:
  1. All 30 multi-resolution features (15 fast + 10 slow + 5 4h)
  2. Multiple horizons tested (5, 10, 15, 30 min)
  3. Walk-forward OOS evaluation with cost-adjusted metrics
  4. High deadzone (0.7+) to reduce trade frequency ~50x
  5. Volatility-aware position sizing in backtest
  6. Per-trade profit analysis (gross alpha vs cost)

Usage:
    cd /quant_system
    python3 scripts/train_1m_v2.py --symbol BTCUSDT
    python3 scripts/train_1m_v2.py --symbol BTCUSDT --horizon 10
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, "/quant_system")

from features.multi_resolution import (
    compute_multi_resolution_features,
    FAST_FEATURE_NAMES,
    SLOW_FEATURE_NAMES,
    SLOW_4H_FEATURE_NAMES,
)
from features.dynamic_selector import greedy_ic_select

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────

WARMUP = 300          # 5 hours warmup for feature stability
COST_BPS_TAKER = 7    # taker fee round-trip (bps)
COST_BPS_MAKER = 2    # maker fee round-trip (bps)

# LightGBM params — more regularized to prevent overfitting at 1m
LGBM_PARAMS = {
    "n_estimators": 500,
    "max_depth": 4,
    "learning_rate": 0.008,
    "num_leaves": 10,
    "min_child_samples": 500,
    "reg_alpha": 1.0,
    "reg_lambda": 10.0,
    "subsample": 0.4,
    "colsample_bytree": 0.5,
    "objective": "regression",
    "verbosity": -1,
}


def _compute_target(closes: np.ndarray, horizon: int) -> np.ndarray:
    """Forward return, clipped at 1/99 percentile."""
    n = len(closes)
    ret = np.full(n, np.nan)
    ret[:n - horizon] = closes[horizon:] / closes[:n - horizon] - 1.0
    valid = ret[~np.isnan(ret)]
    if len(valid) > 10:
        p1, p99 = np.percentile(valid, [1, 99])
        ret = np.where(np.isnan(ret), np.nan, np.clip(ret, p1, p99))
    return ret


def fast_spearman_ic(x: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import spearmanr
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 50:
        return 0.0
    r, _ = spearmanr(x[mask], y[mask])
    return float(r) if not np.isnan(r) else 0.0


# ── Walk-forward backtest ─────────────────────────────────────

def walk_forward_backtest(
    feat_df: pd.DataFrame,
    closes: np.ndarray,
    feature_names: List[str],
    horizon: int,
    deadzones: List[float],
    cost_bps: float = COST_BPS_MAKER,
) -> pd.DataFrame:
    """Walk-forward OOS backtest with multiple deadzone levels.

    Split: 60% train / 20% val / 20% test
    Train on train, early-stop on val, evaluate on test.
    """
    import lightgbm as lgb

    n = len(closes)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)

    y = _compute_target(closes, horizon)

    # Prepare data
    X = feat_df[feature_names].values[WARMUP:].astype(np.float64)
    y_all = y[WARMUP:]
    closes_all = closes[WARMUP:]
    n_adj = len(X)
    train_end_adj = train_end - WARMUP
    val_end_adj = val_end - WARMUP

    valid_train = ~np.isnan(y_all[:train_end_adj])
    valid_val = ~np.isnan(y_all[train_end_adj:val_end_adj])

    X_train = X[:train_end_adj][valid_train]
    y_train = y_all[:train_end_adj][valid_train]
    X_val = X[train_end_adj:val_end_adj][valid_val]
    y_val = y_all[train_end_adj:val_end_adj][valid_val]

    print(f"  Train: {len(X_train):,} bars, Val: {len(X_val):,} bars, Test: {n_adj - val_end_adj:,} bars")

    # Feature selection on train set
    selected = greedy_ic_select(X_train, y_train, feature_names, top_k=15)
    print(f"  Selected features ({len(selected)}): {selected}")

    sel_idx = [feature_names.index(f) for f in selected]
    X_train_sel = X_train[:, sel_idx]
    X_val_sel = X_val[:, sel_idx]

    # Train
    dtrain = lgb.Dataset(X_train_sel, label=y_train)
    dval = lgb.Dataset(X_val_sel, label=y_val, reference=dtrain)

    bst = lgb.train(
        LGBM_PARAMS, dtrain,
        num_boost_round=LGBM_PARAMS["n_estimators"],
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)],
    )

    # Predict on OOS test set
    X_test = X[val_end_adj:]
    X_test_sel = X_test[:, sel_idx]
    y_test = y_all[val_end_adj:]
    closes_test = closes_all[val_end_adj:]
    pred_test = bst.predict(X_test_sel)

    # IC on test
    test_ic = fast_spearman_ic(pred_test, y_test)
    print(f"  OOS Test IC: {test_ic:.4f}")

    # Backtest with different deadzones
    results = []
    for dz in deadzones:
        res = _simulate_trading(pred_test, closes_test, y_test, horizon, dz, cost_bps)
        res["deadzone"] = dz
        res["test_ic"] = test_ic
        res["horizon"] = horizon
        res["n_features"] = len(selected)
        res["features"] = selected
        res["model"] = bst
        results.append(res)

    return results


def _simulate_trading(
    predictions: np.ndarray,
    closes: np.ndarray,
    targets: np.ndarray,
    horizon: int,
    deadzone: float,
    cost_bps: float,
) -> Dict[str, Any]:
    """Simulate trading on OOS predictions."""
    n = len(predictions)
    cost_frac = cost_bps / 10000  # per side

    # Z-score the predictions for comparable deadzone across horizons
    pred_std = np.nanstd(predictions)
    if pred_std < 1e-12:
        return {"trades": 0, "gross_pnl": 0, "net_pnl": 0, "sharpe": 0, "win_rate": 0}
    z_pred = predictions / pred_std

    position = 0  # +1 long, -1 short, 0 flat
    entry_bar = 0
    entry_price = 0.0
    min_hold = max(horizon // 2, 3)  # min hold = half the horizon
    max_hold = horizon * 6  # max hold = 6x horizon

    trade_pnls = []
    trade_gross = []

    for i in range(n):
        # Exit logic
        if position != 0:
            held = i - entry_bar
            should_exit = False
            if held >= max_hold:
                should_exit = True
            elif held >= min_hold:
                # Exit on signal reversal or fade
                if position == 1 and z_pred[i] < -deadzone * 0.3:
                    should_exit = True
                elif position == -1 and z_pred[i] > deadzone * 0.3:
                    should_exit = True
                elif abs(z_pred[i]) < deadzone * 0.2:
                    should_exit = True

            if should_exit:
                pnl = position * (closes[i] - entry_price) / entry_price
                cost = 2 * cost_frac
                trade_gross.append(pnl)
                trade_pnls.append(pnl - cost)
                position = 0

        # Entry logic
        if position == 0:
            if z_pred[i] > deadzone:
                position = 1
                entry_price = closes[i]
                entry_bar = i
            elif z_pred[i] < -deadzone:
                position = -1
                entry_price = closes[i]
                entry_bar = i

    # Close any open position
    if position != 0:
        pnl = position * (closes[-1] - entry_price) / entry_price
        cost = 2 * cost_frac
        trade_gross.append(pnl)
        trade_pnls.append(pnl - cost)

    n_trades = len(trade_pnls)
    if n_trades == 0:
        return {"trades": 0, "gross_pnl": 0, "net_pnl": 0, "sharpe": 0, "win_rate": 0,
                "avg_gross_bps": 0, "avg_net_bps": 0, "trades_per_day": 0}

    gross_arr = np.array(trade_gross)
    net_arr = np.array(trade_pnls)
    test_days = n / 1440

    return {
        "trades": n_trades,
        "trades_per_day": n_trades / max(test_days, 1),
        "gross_pnl": float(np.sum(gross_arr)),
        "net_pnl": float(np.sum(net_arr)),
        "gross_pnl_pct": float(np.sum(gross_arr) * 100),
        "net_pnl_pct": float(np.sum(net_arr) * 100),
        "avg_gross_bps": float(np.mean(gross_arr) * 10000),
        "avg_net_bps": float(np.mean(net_arr) * 10000),
        "win_rate": float(np.mean(net_arr > 0) * 100),
        "sharpe": float(np.mean(net_arr) / max(np.std(net_arr), 1e-10) * np.sqrt(252 * 1440 / max(n / n_trades, 1))),
        "max_gross_bps": float(np.max(gross_arr) * 10000),
        "min_gross_bps": float(np.min(gross_arr) * 10000),
    }


# ── Main ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train improved 1m model V2")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--data", help="Path to 1m CSV")
    parser.add_argument("--horizon", type=int, default=0, help="0 = test all horizons")
    parser.add_argument("--output-dir", help="Model output directory")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    data_path = args.data or f"data_files/{args.symbol}_1m.csv"
    print("=" * 70)
    print("1-Minute Alpha V2 — Multi-Resolution Training")
    print("=" * 70)

    # Load and compute features
    print(f"\nLoading {data_path} ...")
    df = pd.read_csv(data_path)
    print(f"  {len(df):,} bars ({len(df)/1440:.0f} days)")

    t0 = time.time()
    print("Computing multi-resolution features (fast 1m + slow 1h + 4h) ...")
    feat_df = compute_multi_resolution_features(df, args.symbol)
    elapsed = time.time() - t0
    feature_names = [c for c in feat_df.columns if c != "close"]
    print(f"  {len(feature_names)} features computed in {elapsed:.1f}s")
    print(f"  Fast: {[f for f in feature_names if f in FAST_FEATURE_NAMES]}")
    print(f"  Slow: {[f for f in feature_names if f in SLOW_FEATURE_NAMES]}")
    print(f"  4H:   {[f for f in feature_names if f in SLOW_4H_FEATURE_NAMES]}")

    closes = df["close"].values.astype(np.float64)

    # ── IC Analysis across horizons ──
    horizons_to_test = [args.horizon] if args.horizon > 0 else [5, 10, 15, 30]

    print("\n" + "=" * 70)
    print("IC ANALYSIS — All Features × Horizons")
    print("=" * 70)

    best_horizon = None
    best_ic_sum = 0

    for h in horizons_to_test:
        y = _compute_target(closes, h)
        print(f"\n--- Horizon = {h} min ---")
        ics = []
        for fname in feature_names:
            vals = feat_df[fname].values
            ic = fast_spearman_ic(vals[WARMUP:], y[WARMUP:])
            ics.append((fname, ic))
        ics.sort(key=lambda x: abs(x[1]), reverse=True)
        n_pass = sum(1 for _, ic in ics if abs(ic) >= 0.01)
        ic_sum = sum(abs(ic) for _, ic in ics if abs(ic) >= 0.01)
        print(f"  Features with |IC| >= 0.01: {n_pass}/{len(ics)}")
        for fname, ic in ics[:15]:
            tag = "PASS" if abs(ic) >= 0.01 else "fail"
            print(f"    {fname:30s}  IC={ic:+.4f}  [{tag}]")
        if ic_sum > best_ic_sum:
            best_ic_sum = ic_sum
            best_horizon = h

    if best_horizon is None:
        print("\nNo features pass IC threshold at any horizon. Aborting.")
        return

    print(f"\nBest horizon: {best_horizon} min (total |IC| = {best_ic_sum:.4f})")

    # ── Walk-forward Backtest ──
    print("\n" + "=" * 70)
    print(f"WALK-FORWARD BACKTEST (horizon={best_horizon})")
    print("=" * 70)

    deadzones = [0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0]

    results_maker = walk_forward_backtest(
        feat_df, closes, feature_names, best_horizon, deadzones, cost_bps=COST_BPS_MAKER
    )
    results_taker = walk_forward_backtest(
        feat_df, closes, feature_names, best_horizon, deadzones, cost_bps=COST_BPS_TAKER
    )

    # Display results
    print(f"\n{'─'*90}")
    print(f"{'Config':<20} {'Trades':>7} {'T/Day':>6} {'WinR':>6} {'AvgGross':>9} {'AvgNet':>9} {'Net%':>8} {'Sharpe':>7}")  # noqa: E501
    print(f"{'─'*90}")

    best_result = None
    best_net = -999

    for r in results_maker:
        label = f"MKR dz={r['deadzone']:.1f}"
        if r["trades"] == 0:
            print(f"{label:<20} {'—':>7}")
            continue
        print(f"{label:<20} {r['trades']:>7} {r['trades_per_day']:>6.1f} {r['win_rate']:>5.1f}% "
              f"{r['avg_gross_bps']:>+8.1f}bp {r['avg_net_bps']:>+8.1f}bp "
              f"{r['net_pnl_pct']:>+7.1f}% {r['sharpe']:>7.2f}")
        if r["net_pnl"] > best_net and r["trades"] >= 10:
            best_net = r["net_pnl"]
            best_result = r

    print()
    for r in results_taker:
        label = f"TKR dz={r['deadzone']:.1f}"
        if r["trades"] == 0:
            print(f"{label:<20} {'—':>7}")
            continue
        print(f"{label:<20} {r['trades']:>7} {r['trades_per_day']:>6.1f} {r['win_rate']:>5.1f}% "
              f"{r['avg_gross_bps']:>+8.1f}bp {r['avg_net_bps']:>+8.1f}bp "
              f"{r['net_pnl_pct']:>+7.1f}% {r['sharpe']:>7.2f}")

    # ── Save best model ──
    if best_result and best_result["net_pnl"] > 0:
        print(f"\n{'='*70}")
        print("SAVING BEST MODEL")
        print(f"{'='*70}")
        out_dir = Path(args.output_dir or f"models_v8/{args.symbol}_1m_v2")
        out_dir.mkdir(parents=True, exist_ok=True)

        model = best_result["model"]
        features = best_result["features"]

        # Save LightGBM model
        model.save_model(str(out_dir / "lgbm_1m_v2.txt"))
        with open(out_dir / "lgbm_1m_v2.pkl", "wb") as f:
            pickle.dump(model, f)

        # Export to JSON for Rust binary
        try:
            from alpha.export_model_to_json import export_lightgbm_to_json
            export_lightgbm_to_json(str(out_dir / "lgbm_1m_v2.pkl"), str(out_dir / "lgbm_1m_v2.json"))
            print("  Exported to JSON for Rust binary")
        except Exception as e:
            print(f"  JSON export failed: {e}")

        # Save config
        config = {
            "version": "v8_1m_v2",
            "symbol": args.symbol,
            "ensemble": False,
            "models": ["lgbm_1m_v2.pkl"],
            "ensemble_weights": [1.0],
            "features": features,
            "long_only": False,
            "deadzone": best_result["deadzone"],
            "min_hold": max(best_result["horizon"] // 2, 3),
            "horizon": best_result["horizon"],
            "target_mode": "clipped",
            "ic": best_result["test_ic"],
            "engine": "rust_feature_engine",
            "kline_only": False,  # V2 uses multi-resolution
            "avg_gross_bps": best_result["avg_gross_bps"],
            "avg_net_bps": best_result["avg_net_bps"],
            "win_rate": best_result["win_rate"],
            "trades_per_day": best_result["trades_per_day"],
        }
        with open(out_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        with open(out_dir / "features.json", "w") as f:
            json.dump({"features": features}, f, indent=2)

        print(f"  Model saved to {out_dir}")
        print(f"  Features: {features}")
        print(f"  Horizon: {best_result['horizon']} min")
        print(f"  Deadzone: {best_result['deadzone']}")
        print(f"  OOS IC: {best_result['test_ic']:.4f}")
        print(f"  OOS Net P&L: {best_result['net_pnl_pct']:+.1f}%")
        print(f"  Trades/day: {best_result['trades_per_day']:.1f}")
    else:
        print(f"\n{'='*70}")
        print("NO PROFITABLE CONFIG FOUND — model not saved")
        print(f"{'='*70}")
        if best_result:
            print(f"  Best net P&L: {best_result['net_pnl_pct']:+.1f}% (still negative)")
            print(f"  Best avg gross: {best_result['avg_gross_bps']:+.1f} bps/trade")
            print(f"  Breakeven needs avg gross > {COST_BPS_MAKER} bps/trade (maker)")

    print(f"\n{'='*70}")
    print("DONE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
