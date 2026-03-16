#!/usr/bin/env python3
"""Backtest: 1h signal + 15m execution hybrid strategy.

Compares three approaches:
1. Baseline: 1h bars, 1h model, 1h execution (current system)
2. Pure 15m: 15m bars, 15m model (if available)
3. Hybrid: 15m bars, 1h model signal, 15m execution timing

Usage:
    python3 -m scripts.backtest_hybrid_15m --symbol ETHUSDT
    python3 -m scripts.backtest_hybrid_15m --symbol BTCUSDT,ETHUSDT
"""
from __future__ import annotations

import sys
import time
import json
import logging
import pickle  # used for ML model loading
import argparse
from pathlib import Path

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, "/quant_system")

from features.batch_feature_engine import compute_features_batch
from scripts.signal_postprocess import rolling_zscore, should_exit_position
from decision.hybrid_15m_executor import (
    Hybrid15mExecutor, _Bar15m,
)

COST_BPS_RT = 4


def fast_ic(x, y):
    m = ~(np.isnan(x) | np.isnan(y))
    if m.sum() < 50:
        return 0.0
    r, _ = spearmanr(x[m], y[m])
    return float(r) if not np.isnan(r) else 0.0


def load_1h_model(symbol: str):
    """Load existing 1h model and config."""
    model_dir = Path(f"models_v8/{symbol}_gate_v2")
    with open(model_dir / "config.json") as f:
        config = json.load(f)

    horizon_models = []
    for hcfg in config.get("horizon_models", []):
        lgbm_path = model_dir / hcfg["lgbm"]
        xgb_path = model_dir / hcfg["xgb"]
        with open(lgbm_path, "rb") as f:
            lgbm_data = pickle.load(f)
        xgb_model = None
        if xgb_path.exists():
            with open(xgb_path, "rb") as f:
                xgb_model = pickle.load(f)["model"]
        horizon_models.append({
            "horizon": hcfg["horizon"],
            "lgbm": lgbm_data["model"],
            "xgb": xgb_model,
            "features": lgbm_data["features"],
        })

    return config, horizon_models


def predict_1h(horizon_models, features_dict, lgbm_xgb_w=0.5):
    """Run 1h model prediction on a feature dict."""
    preds = []
    for hm in horizon_models:
        x = np.zeros((1, len(hm["features"])))
        for j, fname in enumerate(hm["features"]):
            x[0, j] = features_dict.get(fname, 0.0)
        pred = float(hm["lgbm"].predict(x)[0])
        if hm["xgb"] is not None:
            try:
                import xgboost as xgb
                xgb_pred = float(hm["xgb"].predict(xgb.DMatrix(x))[0])
                pred = lgbm_xgb_w * pred + (1 - lgbm_xgb_w) * xgb_pred
            except Exception as e:
                logger.debug("XGBoost prediction failed in hybrid 15m backtest: %s", e)
        preds.append(pred)
    return float(np.mean(preds))


def aggregate_15m_to_1h(df_15m: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 15m bars to 1h for feature computation."""
    df = df_15m.copy()
    # Group by 1-hour windows
    df["group"] = df["open_time"] // (3600 * 1000)
    agg = df.groupby("group").agg({
        "open_time": "first",
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "close_time": "last",
        "quote_volume": "sum",
        "trades": "sum",
        "taker_buy_volume": "sum",
        "taker_buy_quote_volume": "sum",
        "ignore": "last",
    }).reset_index(drop=True)
    return agg


def backtest_baseline_1h(
    config, horizon_models, df_1h, feat_df_1h,
    train_end_1h, deadzone, min_hold, max_hold, long_only,
):
    """Baseline: 1h model on 1h bars (current system)."""
    len(df_1h) - train_end_1h
    closes = df_1h["close"].values[train_end_1h:]
    feature_names_all = list(feat_df_1h.columns)

    # Predict on each 1h bar
    preds = []
    for i in range(train_end_1h, len(df_1h)):
        feat_dict = {c: float(feat_df_1h.iloc[i][c]) for c in feature_names_all
                     if isinstance(feat_df_1h.iloc[i][c], (int, float, np.floating))}
        pred = predict_1h(horizon_models, feat_dict)
        preds.append(pred)

    preds = np.array(preds)
    z = rolling_zscore(preds, window=720, warmup=180)

    # Simple backtest
    cost_frac = COST_BPS_RT / 10000
    pos = 0.0
    entry_bar = 0
    trades = []

    for i in range(len(z)):
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
                pnl = pos * (closes[i] - closes[entry_bar]) / closes[entry_bar]
                trades.append(pnl - cost_frac)
                pos = 0.0

        if pos == 0:
            if z[i] > deadzone:
                pos = 1.0
                entry_bar = i
            elif not long_only and z[i] < -deadzone:
                pos = -1.0
                entry_bar = i

    return _compute_stats(trades, len(z), bars_per_day=24, label="1h baseline")


def backtest_hybrid(
    config, horizon_models, df_15m, feat_df_1h, df_1h,
    train_end_1h, deadzone, min_hold, max_hold, long_only,
    trailing_stop_pct=0.015,
):
    """Hybrid: 1h model signal + 15m execution timing."""
    # Map 1h bar timestamps to OOS period
    oos_start_ts = df_1h["open_time"].iloc[train_end_1h]
    df_15m_oos = df_15m[df_15m["open_time"] >= oos_start_ts].reset_index(drop=True)
    n_15m = len(df_15m_oos)

    if n_15m < 100:
        return _compute_stats([], 0, 96, "hybrid")

    # Pre-compute 1h predictions and z-scores
    feature_names_all = list(feat_df_1h.columns)
    preds_1h = []
    for i in range(train_end_1h, len(df_1h)):
        feat_dict = {c: float(feat_df_1h.iloc[i][c]) for c in feature_names_all
                     if isinstance(feat_df_1h.iloc[i][c], (int, float, np.floating))}
        pred = predict_1h(horizon_models, feat_dict)
        preds_1h.append(pred)

    z_1h = rolling_zscore(np.array(preds_1h), window=720, warmup=180)

    # Map each 15m bar to its 1h z-score
    z_1h_timestamps = df_1h["open_time"].values[train_end_1h:]
    ts_15m = df_15m_oos["open_time"].values

    # For each 15m bar, find the most recent 1h z-score
    z_for_15m = np.zeros(n_15m)
    j = 0
    for i in range(n_15m):
        while j + 1 < len(z_1h_timestamps) and z_1h_timestamps[j + 1] <= ts_15m[i]:
            j += 1
        z_for_15m[i] = z_1h[j]

    # Run hybrid executor
    executor = Hybrid15mExecutor(
        deadzone=deadzone,
        min_hold_15m=min_hold * 4,    # convert 1h holds to 15m bars
        max_hold_15m=max_hold * 4,
        trailing_stop_pct=trailing_stop_pct,
        long_only=long_only,
    )

    cost_frac = COST_BPS_RT / 10000
    trades = []
    entry_reasons = {}
    exit_reasons = {}

    for i in range(n_15m):
        bar = _Bar15m(
            open=float(df_15m_oos.iloc[i]["open"]),
            high=float(df_15m_oos.iloc[i]["high"]),
            low=float(df_15m_oos.iloc[i]["low"]),
            close=float(df_15m_oos.iloc[i]["close"]),
            volume=float(df_15m_oos.iloc[i]["volume"]),
            timestamp=int(df_15m_oos.iloc[i]["open_time"]),
        )

        event = executor.on_15m_bar(bar, z_for_15m[i])
        if event is None:
            continue

        if event["action"] == "entry":
            entry_price = event["price"]
            entry_side = 1 if event["side"] == "long" else -1
            reason = event["reason"]
            entry_reasons[reason] = entry_reasons.get(reason, 0) + 1

        elif event["action"] == "exit":
            event["price"]
            reason = event["reason"]
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1

            # Find the entry price from executor state
            # The exit event contains the exit price; we need to track entry ourselves
            # Actually let's track it properly
            pass

    # Re-run with tracking
    executor.reset()
    trades = []
    entry_price = 0.0
    entry_side = 0
    entry_reasons = {}
    exit_reasons = {}

    for i in range(n_15m):
        bar = _Bar15m(
            open=float(df_15m_oos.iloc[i]["open"]),
            high=float(df_15m_oos.iloc[i]["high"]),
            low=float(df_15m_oos.iloc[i]["low"]),
            close=float(df_15m_oos.iloc[i]["close"]),
            volume=float(df_15m_oos.iloc[i]["volume"]),
            timestamp=int(df_15m_oos.iloc[i]["open_time"]),
        )

        event = executor.on_15m_bar(bar, z_for_15m[i])
        if event is None:
            continue

        if event["action"] == "entry":
            entry_price = event["price"]
            entry_side = 1 if event["side"] == "long" else -1
            entry_reasons[event["reason"]] = entry_reasons.get(event["reason"], 0) + 1

        elif event["action"] == "exit":
            pnl = entry_side * (event["price"] - entry_price) / entry_price
            trades.append(pnl - cost_frac)
            exit_reasons[event["reason"]] = exit_reasons.get(event["reason"], 0) + 1
            entry_side = 0

    stats = _compute_stats(trades, n_15m, bars_per_day=96, label="hybrid")
    stats["entry_reasons"] = entry_reasons
    stats["exit_reasons"] = exit_reasons
    return stats


def _compute_stats(trades, n_bars, bars_per_day, label):
    if not trades:
        return {"label": label, "sharpe": 0, "trades": 0, "return": 0, "win_rate": 0}

    net = np.array(trades)
    avg_hold = n_bars / max(len(trades), 1)
    tpy = 365 * bars_per_day / max(avg_hold, 1)
    sharpe = float(np.mean(net) / max(np.std(net, ddof=1), 1e-10) * np.sqrt(tpy))

    return {
        "label": label,
        "sharpe": round(sharpe, 2),
        "trades": len(trades),
        "return": round(float(np.sum(net)) * 100, 2),
        "win_rate": round(float(np.mean(net > 0)) * 100, 1),
        "avg_net_bps": round(float(np.mean(net)) * 10000, 1),
        "max_dd": round(float(_max_drawdown(net)) * 100, 2),
        "avg_hold_hours": round(avg_hold / (bars_per_day / 24), 1),
    }


def _max_drawdown(returns):
    cumret = np.cumsum(returns)
    peak = np.maximum.accumulate(cumret)
    dd = peak - cumret
    return float(np.max(dd)) if len(dd) > 0 else 0.0


def run_comparison(symbol: str):
    """Run full comparison for one symbol."""
    print(f"\n{'='*70}")
    print(f"  {symbol} — 1h Signal + 15m Execution Hybrid")
    print(f"{'='*70}")

    # Load data
    path_15m = Path(f"data_files/{symbol}_15m.csv")
    path_1h = Path(f"data_files/{symbol}_1h.csv")
    if not path_15m.exists():
        print(f"  ERROR: {path_15m} not found")
        return
    if not path_1h.exists():
        print(f"  ERROR: {path_1h} not found")
        return

    df_15m = pd.read_csv(path_15m)
    df_1h = pd.read_csv(path_1h)

    # Also aggregate 15m → 1h for alignment check
    agg_1h = aggregate_15m_to_1h(df_15m)
    print(f"  15m data: {len(df_15m):,} bars")
    print(f"  1h data:  {len(df_1h):,} bars")
    print(f"  15m→1h:   {len(agg_1h):,} bars")

    # Align: use only the overlapping period
    overlap_start = max(df_1h["open_time"].iloc[0], df_15m["open_time"].iloc[0])
    overlap_end = min(df_1h["open_time"].iloc[-1], df_15m["open_time"].iloc[-1])
    df_1h = df_1h[(df_1h["open_time"] >= overlap_start) & (df_1h["open_time"] <= overlap_end)].reset_index(drop=True)
    df_15m = df_15m[(df_15m["open_time"] >= overlap_start) & (df_15m["open_time"] <= overlap_end)].reset_index(drop=True)
    print(f"  Overlap:  {len(df_1h):,} 1h bars, {len(df_15m):,} 15m bars")

    # Load model
    config, horizon_models = load_1h_model(symbol)
    deadzone = config.get("deadzone", 0.3)
    min_hold = config.get("min_hold", 12)
    max_hold = config.get("max_hold", 96)
    long_only = config.get("long_only", False)
    print(f"  Config: dz={deadzone}, hold=[{min_hold},{max_hold}], long_only={long_only}")

    # Compute 1h features
    print("  Computing 1h features...", end=" ", flush=True)
    t0 = time.time()
    feat_df_1h = compute_features_batch(symbol, df_1h, include_v11=False)
    print(f"done ({time.time()-t0:.1f}s)")

    # OOS split: last 6 months
    bars_per_month_1h = 24 * 30
    oos_bars = bars_per_month_1h * 6
    if oos_bars > len(df_1h) * 0.6:
        oos_bars = int(len(df_1h) * 0.4)
    train_end_1h = len(df_1h) - oos_bars
    oos_start = pd.Timestamp(df_1h["open_time"].iloc[train_end_1h], unit="ms")
    oos_end = pd.Timestamp(df_1h["open_time"].iloc[-1], unit="ms")
    print(f"  OOS: {oos_bars} 1h bars ({oos_start.date()} → {oos_end.date()})")

    # ── Strategy 1: Baseline 1h ──
    print("\n  Running baseline 1h...", end=" ", flush=True)
    t0 = time.time()
    baseline = backtest_baseline_1h(
        config, horizon_models, df_1h, feat_df_1h,
        train_end_1h, deadzone, min_hold, max_hold, long_only,
    )
    print(f"done ({time.time()-t0:.1f}s)")

    # ── Strategy 2: Hybrid ──
    results_hybrid = []
    for ts_pct in [0.01, 0.015, 0.02, 0.03]:
        print(f"\n  Running hybrid (trailing={ts_pct:.1%})...", end=" ", flush=True)
        t0 = time.time()
        hybrid = backtest_hybrid(
            config, horizon_models, df_15m, feat_df_1h, df_1h,
            train_end_1h, deadzone, min_hold, max_hold, long_only,
            trailing_stop_pct=ts_pct,
        )
        print(f"done ({time.time()-t0:.1f}s)")
        hybrid["trailing_stop"] = ts_pct
        results_hybrid.append(hybrid)

    # ── Results ──
    print(f"\n  {'='*65}")
    print(f"  COMPARISON RESULTS — {symbol}")
    print(f"  {'='*65}")

    print(f"\n  {'Strategy':>25} {'Sharpe':>8} {'Trades':>8} {'WR%':>6} {'Ret%':>8} {'MaxDD%':>8} {'AvgHold':>8} {'NetBps':>8}")
    print(f"  {'-'*25:>25} {'-'*8:>8} {'-'*8:>8} {'-'*6:>6} {'-'*8:>8} {'-'*8:>8} {'-'*8:>8} {'-'*8:>8}")

    all_results = [baseline] + results_hybrid
    for r in all_results:
        label = r.get("label", "?")
        if "trailing_stop" in r:
            label = f"hybrid ts={r['trailing_stop']:.1%}"
        print(
            f"  {label:>25} "
            f"{r['sharpe']:>8.2f} "
            f"{r['trades']:>8} "
            f"{r['win_rate']:>6.1f} "
            f"{r['return']:>+8.2f} "
            f"{r.get('max_dd', 0):>8.2f} "
            f"{r.get('avg_hold_hours', 0):>7.1f}h "
            f"{r.get('avg_net_bps', 0):>8.1f}"
        )

    # Print entry/exit reason breakdown for best hybrid
    best_hybrid = max(results_hybrid, key=lambda r: r["sharpe"])
    if best_hybrid.get("entry_reasons"):
        print("\n  Best hybrid entry reasons:")
        for reason, count in sorted(best_hybrid["entry_reasons"].items(), key=lambda x: -x[1]):
            print(f"    {reason}: {count}")
    if best_hybrid.get("exit_reasons"):
        print("  Best hybrid exit reasons:")
        for reason, count in sorted(best_hybrid["exit_reasons"].items(), key=lambda x: -x[1]):
            print(f"    {reason}: {count}")


def main():
    parser = argparse.ArgumentParser(description="Hybrid 15m Backtest")
    parser.add_argument("--symbol", default="ETHUSDT",
                        help="Comma-separated symbols")
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbol.split(",")]

    print("=" * 70)
    print("  1H SIGNAL + 15M EXECUTION — HYBRID BACKTEST")
    print("=" * 70)

    for symbol in symbols:
        run_comparison(symbol)

    print(f"\n{'='*70}")
    print("  DONE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
