#!/usr/bin/env python3
"""Full-history 4h backtest — validate the production model on all available data.

Method 1: Rolling walk-forward (retrain every 3 months, test next 3)
Method 2: Production model on OOS holdout (last 18 months)

Uses ensemble LGBM+XGB, same as production training.
"""
from __future__ import annotations
import sys
import time
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass
import numpy as np
import pandas as pd

sys.path.insert(0, "/quant_system")

from features.batch_feature_engine import compute_features_batch
from features.dynamic_selector import greedy_ic_select
from alpha.training.train_v7_alpha import INTERACTION_FEATURES, BLACKLIST
from scipy.stats import spearmanr

# ── Config ──
SYMBOL = "BTCUSDT"
WARMUP = 30
BARS_PER_DAY = 6
BARS_PER_MONTH = BARS_PER_DAY * 30  # 180
COST_MAKER_RT = 4   # bps round-trip
COST_TAKER_RT = 14


def fast_ic(x, y):
    m = ~(np.isnan(x) | np.isnan(y))
    if m.sum() < 50:
        return 0.0
    r, _ = spearmanr(x[m], y[m])
    return float(r) if not np.isnan(r) else 0.0


def resample_1m_to_4h(df_1m):
    ts_col = "open_time" if "open_time" in df_1m.columns else "timestamp"
    ts = df_1m[ts_col].values.astype(np.int64)
    groups = ts // (4 * 60 * 60_000)
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


def compute_features(df_4h):
    _has_v11 = Path("data_files/macro_daily.csv").exists()
    feat_df = compute_features_batch(SYMBOL, df_4h, include_v11=_has_v11)
    # Add interaction features
    for int_name, feat_a, feat_b in INTERACTION_FEATURES:
        if feat_a in feat_df.columns and feat_b in feat_df.columns:
            feat_df[int_name] = feat_df[feat_a].astype(float) * feat_df[feat_b].astype(float)
    feature_names = [c for c in feat_df.columns
                     if c not in ("close", "open_time", "timestamp")
                     and c not in BLACKLIST]
    return feat_df, feature_names


def compute_target(closes, horizon):
    n = len(closes)
    y = np.full(n, np.nan)
    y[:n-horizon] = closes[horizon:] / closes[:n-horizon] - 1
    v = y[~np.isnan(y)]
    if len(v) > 10:
        p1, p99 = np.percentile(v, [1, 99])
        y = np.where(np.isnan(y), np.nan, np.clip(y, p1, p99))
    return y


# ── Detailed Backtest Engine ──

@dataclass
class Trade:
    entry_bar: int
    exit_bar: int
    direction: int
    entry_price: float
    exit_price: float
    gross_pnl: float
    net_pnl: float
    score: float
    hold_bars: int


def run_backtest(
    predictions: np.ndarray,
    closes: np.ndarray,
    timestamps: np.ndarray,
    horizon: int,
    deadzone: float,
    cost_bps: float,
    long_only: bool = True,
    min_hold: int = 3,
    max_hold: int = 36,
) -> Dict[str, Any]:
    """Detailed backtest with trade-by-trade tracking."""
    n = len(predictions)
    pred_std = np.nanstd(predictions)
    if pred_std < 1e-12:
        return {"trades": [], "equity_curve": [10000.0] * n}

    z = predictions / pred_std
    cost_frac = cost_bps / 10000
    notional = 500.0  # $500 per trade (5% of $10k)

    trades = []
    pos = 0
    ep = 0.0
    eb = 0
    score = 0.0
    equity = 10000.0
    equity_curve = [equity]

    for i in range(n):
        if pos != 0:
            held = i - eb
            should_exit = False
            if held >= max_hold:
                should_exit = True
            elif held >= min_hold:
                if pos * z[i] < -0.3:
                    should_exit = True
                elif abs(z[i]) < 0.2:
                    should_exit = True

            if should_exit:
                pnl_pct = pos * (closes[i] - ep) / ep
                gross = pnl_pct * notional
                cost = cost_frac * notional
                net = gross - cost
                equity += net

                trades.append(Trade(
                    entry_bar=eb, exit_bar=i, direction=pos,
                    entry_price=ep, exit_price=closes[i],
                    gross_pnl=gross, net_pnl=net,
                    score=score, hold_bars=held,
                ))
                pos = 0

        if pos == 0:
            if z[i] > deadzone:
                pos = 1
                ep = closes[i]
                eb = i
                score = z[i]
            elif not long_only and z[i] < -deadzone:
                pos = -1
                ep = closes[i]
                eb = i
                score = z[i]

        equity_curve.append(equity)

    # Close open position
    if pos != 0:
        pnl_pct = pos * (closes[-1] - ep) / ep
        gross = pnl_pct * notional
        cost = cost_frac * notional
        net = gross - cost
        equity += net
        trades.append(Trade(
            entry_bar=eb, exit_bar=n-1, direction=pos,
            entry_price=ep, exit_price=closes[-1],
            gross_pnl=gross, net_pnl=net,
            score=score, hold_bars=n-1-eb,
        ))
        equity_curve.append(equity)

    return {"trades": trades, "equity_curve": equity_curve}


def analyze_trades(trades: List[Trade], timestamps: np.ndarray, n_bars: int, label: str):
    """Print detailed trade analysis."""
    if not trades:
        print(f"\n  [{label}] No trades.")
        return {}

    gross = np.array([t.gross_pnl for t in trades])
    net = np.array([t.net_pnl for t in trades])
    holds = np.array([t.hold_bars for t in trades])
    days = n_bars / BARS_PER_DAY

    wins = sum(1 for t in trades if t.net_pnl > 0)
    longs = sum(1 for t in trades if t.direction > 0)
    shorts = len(trades) - longs

    # Monthly breakdown
    monthly = {}
    for t in trades:
        ts_ms = int(timestamps[min(t.exit_bar, len(timestamps)-1)])
        month = pd.Timestamp(ts_ms, unit="ms").strftime("%Y-%m")
        if month not in monthly:
            monthly[month] = {"gross": 0, "net": 0, "trades": 0, "wins": 0}
        monthly[month]["gross"] += t.gross_pnl
        monthly[month]["net"] += t.net_pnl
        monthly[month]["trades"] += 1
        if t.net_pnl > 0:
            monthly[month]["wins"] += 1

    # Drawdown
    eq = 10000.0
    peak = eq
    max_dd = 0.0
    for t in trades:
        eq += t.net_pnl
        peak = max(peak, eq)
        dd = (peak - eq) / peak
        max_dd = max(max_dd, dd)

    # Sharpe (annualized from per-trade)
    if len(net) > 1:
        avg_hold_days = np.mean(holds) * 4 / 24  # bars * 4h / 24h
        trades_per_year = 365 / max(avg_hold_days, 0.5)
        sharpe = float(np.mean(net) / max(np.std(net, ddof=1), 1e-10) * np.sqrt(trades_per_year))
    else:
        sharpe = 0.0

    print(f"\n  [{label}]")
    print(f"  {'─'*65}")
    print(f"  Trades: {len(trades)} ({len(trades)/max(days,1):.2f}/day, "
          f"{len(trades)/max(days/7,1):.1f}/week)  Long: {longs}  Short: {shorts}")
    print(f"  Win Rate: {wins/len(trades)*100:.1f}%")
    print(f"  Avg Hold: {np.mean(holds):.1f} bars ({np.mean(holds)*4:.0f}h)")
    print(f"  Avg Gross: {np.mean(gross)/500*10000:+.1f} bps   Avg Net: {np.mean(net)/500*10000:+.1f} bps")
    print(f"  Total Gross: ${np.sum(gross):+.0f}   Total Net: ${np.sum(net):+.0f}")
    print(f"  Net Return: {np.sum(net)/10000*100:+.2f}%   (on $10k capital)")
    print(f"  Max Drawdown: {max_dd*100:.2f}%")
    print(f"  Sharpe (annual): {sharpe:.2f}")

    # Monthly table
    print(f"\n  {'Month':>8} {'Trades':>7} {'WinR':>6} {'Gross$':>8} {'Net$':>8} {'CumNet$':>9}")
    print(f"  {'─'*50}")
    cum = 0
    pos_months = 0
    for month in sorted(monthly.keys()):
        m = monthly[month]
        cum += m["net"]
        wr = m["wins"] / m["trades"] * 100 if m["trades"] > 0 else 0
        if m["net"] > 0:
            pos_months += 1
        print(f"  {month:>8} {m['trades']:>7} {wr:>5.0f}% {m['gross']:>+8.1f} {m['net']:>+8.1f} {cum:>+9.1f}")
    print(f"  Positive months: {pos_months}/{len(monthly)}")

    # Best/worst trades
    sorted_trades = sorted(trades, key=lambda t: t.net_pnl)
    print("\n  Worst 3 trades:")
    for t in sorted_trades[:3]:
        ts = pd.Timestamp(int(timestamps[t.entry_bar]), unit="ms").strftime("%Y-%m-%d %H:%M")
        print(f"    {ts} {'L' if t.direction>0 else 'S'} @ ${t.entry_price:.0f} "
              f"hold={t.hold_bars}bars ({t.hold_bars*4}h) "
              f"gross=${t.gross_pnl:+.1f} net=${t.net_pnl:+.1f}")
    print("  Best 3 trades:")
    for t in sorted_trades[-3:]:
        ts = pd.Timestamp(int(timestamps[t.entry_bar]), unit="ms").strftime("%Y-%m-%d %H:%M")
        print(f"    {ts} {'L' if t.direction>0 else 'S'} @ ${t.entry_price:.0f} "
              f"hold={t.hold_bars}bars ({t.hold_bars*4}h) "
              f"gross=${t.gross_pnl:+.1f} net=${t.net_pnl:+.1f}")

    return {
        "trades": len(trades), "win_rate": wins/len(trades)*100,
        "avg_gross_bps": float(np.mean(gross)/500*10000),
        "avg_net_bps": float(np.mean(net)/500*10000),
        "total_net": float(np.sum(net)),
        "net_return_pct": float(np.sum(net)/10000*100),
        "max_dd": max_dd, "sharpe": sharpe,
        "pos_months": pos_months, "total_months": len(monthly),
        "longs": longs, "shorts": shorts,
    }


# ── Main ──

def main():
    print("=" * 70)
    print("4H FULL HISTORY BACKTEST — BTCUSDT")
    print("=" * 70)

    # Load data
    df_1m = pd.read_csv("/quant_system/data_files/BTCUSDT_1m.csv")
    print(f"Loaded {len(df_1m):,} 1m bars")
    df_4h = resample_1m_to_4h(df_1m)
    n = len(df_4h)
    print(f"Resampled to {n:,} 4h bars ({n/BARS_PER_DAY:.0f} days)")

    timestamps = df_4h["open_time"].values.astype(np.int64)
    start_date = pd.Timestamp(timestamps[0], unit="ms").strftime("%Y-%m-%d")
    end_date = pd.Timestamp(timestamps[-1], unit="ms").strftime("%Y-%m-%d")
    print(f"Date range: {start_date} → {end_date}")

    # Features
    t0 = time.time()
    feat_df, feature_names = compute_features(df_4h)
    print(f"Features: {len(feature_names)} in {time.time()-t0:.1f}s")

    closes = df_4h["close"].values.astype(np.float64)

    # Load production config
    config_path = Path("models_v8/BTCUSDT_4h_v1/config.json")
    with open(config_path) as f:
        prod_config = json.load(f)

    horizon = prod_config["horizon"]       # 12
    deadzone = prod_config["deadzone"]     # 2.0
    min_hold = prod_config["min_hold"]     # 3
    max_hold = prod_config["max_hold"]     # 36
    long_only = prod_config["long_only"]   # True
    prod_features = prod_config["features"]
    prod_params = {**prod_config["params"], "objective": "regression", "verbosity": -1}
    xgb_params_cfg = prod_config["xgb_params"]

    print(f"\nProduction config: horizon={horizon} ({horizon*4}h), dz={deadzone}, "
          f"hold=[{min_hold},{max_hold}], long_only={long_only}")
    print(f"Features: {len(prod_features)}")

    # ═══════════════════════════════════════════════════════════════
    # METHOD 1: Rolling Walk-Forward (retrain every 3 months)
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("METHOD 1: Rolling Walk-Forward (retrain every 3 months)")
    print("  Ensemble LGBM+XGB, feature reselection per fold")
    print("=" * 70)

    import lightgbm as lgb
    import xgboost as xgb

    FOLD_SIZE = BARS_PER_MONTH * 3   # 540 bars = 3 months
    MIN_TRAIN = BARS_PER_MONTH * 6   # 1080 bars = 6 months

    y = compute_target(closes, horizon)
    X = feat_df[feature_names].values.astype(np.float64)

    all_predictions = np.full(n, np.nan)
    fold_idx = 0

    fold_start = MIN_TRAIN
    while fold_start + FOLD_SIZE <= n:
        fold_end = min(fold_start + FOLD_SIZE, n)
        # Use last 25% of training data as validation for early stopping
        val_start = fold_start - FOLD_SIZE // 2

        # Train data
        valid_tr = ~np.isnan(y[WARMUP:val_start])
        valid_val = ~np.isnan(y[val_start:fold_start])

        X_tr = X[WARMUP:val_start][valid_tr]
        y_tr = y[WARMUP:val_start][valid_tr]
        X_val = X[val_start:fold_start][valid_val]
        y_val = y[val_start:fold_start][valid_val]

        if len(X_tr) < 200 or len(X_val) < 50:
            fold_start += FOLD_SIZE
            continue

        # Feature selection per fold
        selected = greedy_ic_select(X_tr, y_tr, feature_names, top_k=14)
        sel_idx = [feature_names.index(f) for f in selected]

        # LGBM
        dtrain = lgb.Dataset(X_tr[:, sel_idx], label=y_tr)
        dval = lgb.Dataset(X_val[:, sel_idx], label=y_val, reference=dtrain)
        lgbm_bst = lgb.train(prod_params, dtrain, num_boost_round=500,
                             valid_sets=[dval],
                             callbacks=[lgb.early_stopping(50, verbose=False),
                                        lgb.log_evaluation(0)])

        # XGB
        xgb_p = {**xgb_params_cfg}
        dtrain_x = xgb.DMatrix(X_tr[:, sel_idx], label=y_tr)
        dval_x = xgb.DMatrix(X_val[:, sel_idx], label=y_val)
        xgb_bst = xgb.train(xgb_p, dtrain_x, num_boost_round=500,
                             evals=[(dval_x, "val")],
                             early_stopping_rounds=50, verbose_eval=False)

        # Predict on test fold
        X_test = X[fold_start:fold_end][:, sel_idx]
        lgbm_pred = lgbm_bst.predict(X_test)
        xgb_pred = xgb_bst.predict(xgb.DMatrix(X_test))
        ensemble_pred = 0.5 * lgbm_pred + 0.5 * xgb_pred

        all_predictions[fold_start:fold_end] = ensemble_pred

        # Fold IC
        y_test = y[fold_start:fold_end]
        ic = fast_ic(ensemble_pred, y_test)
        fold_idx += 1
        f_start_date = pd.Timestamp(timestamps[fold_start], unit="ms").strftime("%Y-%m-%d")
        f_end_date = pd.Timestamp(timestamps[fold_end-1], unit="ms").strftime("%Y-%m-%d")
        print(f"  Fold {fold_idx}: {f_start_date}→{f_end_date}  "
              f"train={len(X_tr)}, IC={ic:.4f}, features={selected[:5]}...")

        fold_start += FOLD_SIZE

    # Backtest on walk-forward predictions
    valid_mask = ~np.isnan(all_predictions)
    first_valid = np.argmax(valid_mask)
    pred_slice = all_predictions[first_valid:]
    close_slice = closes[first_valid:]
    ts_slice = timestamps[first_valid:]
    n_test = len(pred_slice)

    print(f"\n  Walk-forward coverage: {n_test} bars ({n_test/BARS_PER_DAY:.0f} days)")

    for cost_label, cost_bps in [("Maker (4bp)", COST_MAKER_RT), ("Taker (14bp)", COST_TAKER_RT)]:
        bt = run_backtest(pred_slice, close_slice, ts_slice,
                          horizon, deadzone, cost_bps, long_only, min_hold, max_hold)
        analyze_trades(bt["trades"], ts_slice, n_test,
                               f"Walk-Forward {cost_label}")

    # ═══════════════════════════════════════════════════════════════
    # METHOD 2: Production Model on OOS (last 18 months)
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("METHOD 2: Production Model on OOS (last 18 months)")
    print("  Uses the trained production LGBM+XGB ensemble")
    print("=" * 70)

    # Load production models
    with open("models_v8/BTCUSDT_4h_v1/lgbm_v8.pkl", "rb") as f:
        lgbm_data = pickle.load(f)
    with open("models_v8/BTCUSDT_4h_v1/xgb_v8.pkl", "rb") as f:
        xgb_data = pickle.load(f)

    lgbm_model = lgbm_data["model"]
    xgb_model = xgb_data["model"]
    model_features = lgbm_data["features"]

    # Align features
    oos_start = n - BARS_PER_MONTH * 18
    oos_feat = feat_df[feature_names].values[oos_start:].astype(np.float64)
    oos_closes = closes[oos_start:]
    oos_ts = timestamps[oos_start:]

    # Map model features to feature_names indices
    model_feat_idx = []
    for mf in model_features:
        if mf in feature_names:
            model_feat_idx.append(feature_names.index(mf))
        else:
            print(f"  Warning: feature '{mf}' not found, using zeros")
            model_feat_idx.append(-1)

    X_oos = np.zeros((len(oos_feat), len(model_features)))
    for j, idx in enumerate(model_feat_idx):
        if idx >= 0:
            X_oos[:, j] = oos_feat[:, idx]

    lgbm_pred = lgbm_model.predict(X_oos)
    xgb_pred = xgb_model.predict(xgb.DMatrix(X_oos))
    oos_pred = 0.5 * lgbm_pred + 0.5 * xgb_pred

    oos_ic = fast_ic(oos_pred, compute_target(oos_closes, horizon))
    print(f"  OOS bars: {len(oos_closes)} ({len(oos_closes)/BARS_PER_DAY:.0f} days)")
    print(f"  OOS IC: {oos_ic:.4f}")

    for cost_label, cost_bps in [("Maker (4bp)", COST_MAKER_RT), ("Taker (14bp)", COST_TAKER_RT)]:
        bt = run_backtest(oos_pred, oos_closes, oos_ts,
                          horizon, deadzone, cost_bps, long_only, min_hold, max_hold)
        analyze_trades(bt["trades"], oos_ts, len(oos_closes),
                               f"Production Model {cost_label}")

    # ═══════════════════════════════════════════════════════════════
    # COMPARISON SUMMARY
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("SUMMARY: 4h Model vs 1h V8 vs 30m")
    print("=" * 70)

    # Load 1h V8 metrics for comparison
    v8_1h = {}
    v8_path = Path("models_v8/BTCUSDT_gate_v2/config.json")
    if v8_path.exists():
        with open(v8_path) as f:
            v8_1h = json.load(f).get("metrics", {})

    v8_30m = {}
    v30_path = Path("models_v8/BTCUSDT_30m_v1/config.json")
    if v30_path.exists():
        with open(v30_path) as f:
            v8_30m = json.load(f).get("metrics", {})

    print(f"\n  {'Metric':<25s} {'4h (new)':>12s} {'1h V8':>12s} {'30m':>12s}")
    print(f"  {'─'*60}")
    print(f"  {'OOS Sharpe':<25s} {prod_config['metrics']['sharpe']:>+12.2f} "
          f"{v8_1h.get('sharpe', 0):>+12.2f} {v8_30m.get('wf_avg_sharpe', 0):>+12.2f}")
    print(f"  {'OOS IC':<25s} {prod_config['metrics']['ic']:>12.4f} "
          f"{v8_1h.get('ic', 0):>12.4f} {v8_30m.get('oos_ic', 0):>12.4f}")
    print(f"  {'Total Return':<25s} {prod_config['metrics']['total_return']*100:>+12.2f}% "
          f"{v8_1h.get('total_return', 0)*100:>+12.2f}% {'':>12s}")
    print(f"  {'Avg Net bp/trade':<25s} {prod_config['metrics']['avg_net_bps']:>+12.1f} "
          f"{'N/A':>12s} {v8_30m.get('final_avg_net_bps', 0):>+12.1f}")
    print(f"  {'Trades (OOS)':<25s} {prod_config['metrics']['trades']:>12d} "
          f"{'N/A':>12s} {v8_30m.get('final_trades', 0):>12d}")

    print("\n  Done.")


if __name__ == "__main__":
    main()
