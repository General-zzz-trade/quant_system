#!/usr/bin/env python3
"""Full-history 30m backtest — validate the production model on all available data.

Trains on first 60%, validates on next 20%, tests on last 20%.
Also runs a pure OOS rolling backtest: retrain every 3 months, test on next 3 months.
"""
from __future__ import annotations
import sys
import time
import json
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass
import numpy as np
import pandas as pd

sys.path.insert(0, "/quant_system")

from features.batch_feature_engine import compute_features_batch
from features.dynamic_selector import greedy_ic_select
from scipy.stats import spearmanr

# ── Config ──
SYMBOL = "BTCUSDT"
WARMUP = 100
COST_MAKER_RT = 4   # bps round-trip
COST_TAKER_RT = 14
LAG_BARS = 2
LAGGED_FEATURES = {
    "basis", "basis_zscore_24", "basis_momentum", "basis_extreme", "basis_carry_adj",
    "funding_rate", "funding_ma8", "funding_zscore_24", "funding_momentum",
    "funding_extreme", "funding_cumulative_8", "funding_sign_persist",
    "funding_annualized", "funding_vs_vol", "funding_term_slope",
    "fgi_normalized", "fgi_zscore_7", "fgi_extreme",
}


def fast_ic(x, y):
    m = ~(np.isnan(x) | np.isnan(y))
    if m.sum() < 50:
        return 0.0
    r, _ = spearmanr(x[m], y[m])
    return float(r) if not np.isnan(r) else 0.0


def resample_1m_to_30m(df_1m):
    ts_col = "open_time" if "open_time" in df_1m.columns else "timestamp"
    ts = df_1m[ts_col].values.astype(np.int64)
    groups = ts // (30 * 60_000)
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


def compute_features(df_30m):
    _has_v11 = Path("data_files/macro_daily.csv").exists()
    feat_df = compute_features_batch(SYMBOL, df_30m, include_v11=_has_v11)
    for col in LAGGED_FEATURES:
        if col in feat_df.columns:
            feat_df[col] = feat_df[col].shift(LAG_BARS)
    feature_names = [c for c in feat_df.columns if c not in ("close", "open_time", "timestamp")]
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
    direction: int  # +1 long, -1 short
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
) -> Dict[str, Any]:
    """Detailed backtest with trade-by-trade tracking."""
    n = len(predictions)
    pred_std = np.nanstd(predictions)
    if pred_std < 1e-12:
        return {"trades": [], "equity_curve": [10000.0] * n}

    z = predictions / pred_std
    min_hold = max(horizon // 2, 1)
    max_hold = horizon * 6
    cost_frac = cost_bps / 10000

    trades = []
    pos = 0
    ep = 0.0
    eb = 0
    score = 0.0
    equity = 10000.0
    equity_curve = [equity]
    notional = 300.0  # fixed notional per trade

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
            elif z[i] < -deadzone:
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
    days = n_bars / 48

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

    # Sharpe (annualized from daily)
    daily_pnl = {}
    for t in trades:
        ts_ms = int(timestamps[min(t.exit_bar, len(timestamps)-1)])
        day = pd.Timestamp(ts_ms, unit="ms").strftime("%Y-%m-%d")
        daily_pnl[day] = daily_pnl.get(day, 0) + t.net_pnl
    if daily_pnl:
        daily_arr = np.array(list(daily_pnl.values()))
        sharpe = float(np.mean(daily_arr) / max(np.std(daily_arr), 1e-10) * np.sqrt(365))
    else:
        sharpe = 0.0

    print(f"\n  [{label}]")
    print(f"  {'─'*65}")
    print(f"  Trades: {len(trades)} ({len(trades)/days:.1f}/day)  Long: {longs}  Short: {shorts}")
    print(f"  Win Rate: {wins/len(trades)*100:.1f}%")
    print(f"  Avg Hold: {np.mean(holds):.1f} bars ({np.mean(holds)*30:.0f} min)")
    print(f"  Avg Gross: {np.mean(gross)/300*10000:+.1f} bps   Avg Net: {np.mean(net)/300*10000:+.1f} bps")
    print(f"  Total Gross: ${np.sum(gross):+.0f}   Total Net: ${np.sum(net):+.0f}")
    print(f"  Net Return: {np.sum(net)/10000*100:+.2f}%")
    print(f"  Max Drawdown: {max_dd*100:.2f}%")
    print(f"  Sharpe (annual): {sharpe:.2f}")

    # Monthly table
    print(f"\n  {'Month':>8} {'Trades':>7} {'WinR':>6} {'Gross$':>8} {'Net$':>8} {'CumNet$':>9}")
    print(f"  {'─'*50}")
    cum = 0
    for month in sorted(monthly.keys()):
        m = monthly[month]
        cum += m["net"]
        wr = m["wins"] / m["trades"] * 100 if m["trades"] > 0 else 0
        print(f"  {month:>8} {m['trades']:>7} {wr:>5.0f}% {m['gross']:>+8.1f} {m['net']:>+8.1f} {cum:>+9.1f}")

    # Best/worst trades
    sorted_trades = sorted(trades, key=lambda t: t.net_pnl)
    print("\n  Worst 3 trades:")
    for t in sorted_trades[:3]:
        ts = pd.Timestamp(int(timestamps[t.entry_bar]), unit="ms").strftime("%m-%d %H:%M")
        print(f"    {ts} {'L' if t.direction>0 else 'S'} hold={t.hold_bars} "
              f"gross=${t.gross_pnl:+.1f} net=${t.net_pnl:+.1f}")
    print("  Best 3 trades:")
    for t in sorted_trades[-3:]:
        ts = pd.Timestamp(int(timestamps[t.entry_bar]), unit="ms").strftime("%m-%d %H:%M")
        print(f"    {ts} {'L' if t.direction>0 else 'S'} hold={t.hold_bars} "
              f"gross=${t.gross_pnl:+.1f} net=${t.net_pnl:+.1f}")

    return {
        "trades": len(trades), "win_rate": wins/len(trades)*100,
        "avg_net_bps": float(np.mean(net)/300*10000),
        "total_net": float(np.sum(net)),
        "max_dd": max_dd, "sharpe": sharpe,
    }


# ── Main ──

def main():
    print("=" * 70)
    print("30m FULL HISTORY BACKTEST — BTCUSDT")
    print("=" * 70)

    # Load data
    df_1m = pd.read_csv("/quant_system/data_files/BTCUSDT_1m.csv")
    print(f"Loaded {len(df_1m):,} 1m bars")
    df_30m = resample_1m_to_30m(df_1m)
    n = len(df_30m)
    print(f"Resampled to {n:,} 30m bars ({n/48:.0f} days)")

    ts_col = "open_time"
    timestamps = df_30m[ts_col].values.astype(np.int64)
    start_date = pd.Timestamp(timestamps[0], unit="ms").strftime("%Y-%m-%d")
    end_date = pd.Timestamp(timestamps[-1], unit="ms").strftime("%Y-%m-%d")
    print(f"Date range: {start_date} → {end_date}")

    # Features
    t0 = time.time()
    feat_df, feature_names = compute_features(df_30m)
    print(f"Features: {len(feature_names)} in {time.time()-t0:.1f}s")

    closes = df_30m["close"].values.astype(np.float64)
    horizon = 2  # 2 bars = 60 min

    # ═══════════════════════════════════════════════════════════
    # METHOD 1: Rolling Walk-Forward (retrain every 3 months)
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("METHOD 1: Rolling Walk-Forward (retrain every 3 months, test next 3)")
    print("=" * 70)

    import lightgbm as lgb

    FOLD_SIZE = 4320  # ~3 months
    MIN_TRAIN = 8640  # ~6 months minimum

    y = compute_target(closes, horizon)
    X = feat_df[feature_names].values.astype(np.float64)

    # Load production model params
    with open("models_v8/BTCUSDT_30m_v1/config.json") as f:
        prod_config = json.load(f)
    deadzone = prod_config["deadzone"]
    params = {**prod_config["params"], "objective": "regression", "verbosity": -1}

    all_predictions = np.full(n, np.nan)
    all_features_used = {}
    fold_idx = 0

    fold_start = MIN_TRAIN
    while fold_start + FOLD_SIZE <= n:
        fold_end = min(fold_start + FOLD_SIZE, n)
        val_start = max(fold_start - FOLD_SIZE // 2, WARMUP + 1000)

        # Train on [WARMUP, val_start)
        valid_tr = ~np.isnan(y[WARMUP:val_start])
        valid_val = ~np.isnan(y[val_start:fold_start])

        X_tr = X[WARMUP:val_start][valid_tr]
        y_tr = y[WARMUP:val_start][valid_tr]
        X_val = X[val_start:fold_start][valid_val]
        y_val = y[val_start:fold_start][valid_val]

        if len(X_tr) < 1000 or len(X_val) < 200:
            fold_start += FOLD_SIZE
            continue

        selected = greedy_ic_select(X_tr, y_tr, feature_names, top_k=15)
        sel_idx = [feature_names.index(f) for f in selected]

        dtrain = lgb.Dataset(X_tr[:, sel_idx], label=y_tr)
        dval = lgb.Dataset(X_val[:, sel_idx], label=y_val, reference=dtrain)
        bst = lgb.train(params, dtrain, num_boost_round=500,
                        valid_sets=[dval],
                        callbacks=[lgb.early_stopping(50, verbose=False)])

        # Predict OOS fold
        pred = bst.predict(X[fold_start:fold_end][:, sel_idx])
        all_predictions[fold_start:fold_end] = pred

        ic = fast_ic(pred, y[fold_start:fold_end])
        t_start = pd.Timestamp(timestamps[fold_start], unit="ms").strftime("%Y-%m")
        t_end = pd.Timestamp(timestamps[min(fold_end-1, n-1)], unit="ms").strftime("%Y-%m")
        print(f"  Fold {fold_idx+1}: {t_start}→{t_end}  IC={ic:.4f}  features={selected[:5]}...")

        all_features_used[fold_idx] = selected
        fold_idx += 1
        fold_start += FOLD_SIZE

    # Backtest on all OOS predictions
    oos_mask = ~np.isnan(all_predictions)
    oos_indices = np.where(oos_mask)[0]
    if len(oos_indices) > 0:
        first_oos = oos_indices[0]
        pred_oos = all_predictions[first_oos:]
        c_oos = closes[first_oos:]
        ts_oos = timestamps[first_oos:]

        # Replace NaN predictions with 0 (no signal)
        pred_oos = np.nan_to_num(pred_oos, 0.0)

        for cost_label, cost_bps in [("Maker (2bp/side)", COST_MAKER_RT), ("Taker (7bp/side)", COST_TAKER_RT)]:
            result = run_backtest(pred_oos, c_oos, ts_oos, horizon, deadzone, cost_bps)
            analyze_trades(result["trades"], ts_oos, len(c_oos), f"Walk-Forward OOS — {cost_label}")

    # ═══════════════════════════════════════════════════════════
    # METHOD 2: Use Production Model on Full OOS Period
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("METHOD 2: Production Model (trained on 80%) tested on last 20%")
    print("=" * 70)

    # Load production model
    model_path = Path("models_v8/BTCUSDT_30m_v1/lgbm_30m.txt")
    if model_path.exists():
        bst_prod = lgb.Booster(model_file=str(model_path))
        prod_features = prod_config["features"]
        prod_sel_idx = [feature_names.index(f) for f in prod_features if f in feature_names]

        if len(prod_sel_idx) == len(prod_features):
            test_start = int(n * 0.8)
            X_test = X[test_start:][:, prod_sel_idx]
            pred_prod = bst_prod.predict(X_test)
            c_test = closes[test_start:]
            ts_test = timestamps[test_start:]

            ic_prod = fast_ic(pred_prod, y[test_start:])
            print(f"  Production model OOS IC: {ic_prod:.4f}")
            print(f"  Test period: {pd.Timestamp(ts_test[0], unit='ms').strftime('%Y-%m-%d')} → "
                  f"{pd.Timestamp(ts_test[-1], unit='ms').strftime('%Y-%m-%d')}")

            for cost_label, cost_bps in [("Maker", COST_MAKER_RT), ("Taker", COST_TAKER_RT)]:
                result = run_backtest(pred_prod, c_test, ts_test, horizon, deadzone, cost_bps)
                analyze_trades(result["trades"], ts_test, len(c_test), f"Production OOS — {cost_label}")
        else:
            missing = [f for f in prod_features if f not in feature_names]
            print(f"  Missing features: {missing}")
    else:
        print(f"  Model not found: {model_path}")

    # ═══════════════════════════════════════════════════════════
    # METHOD 3: BTC Price Context
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PRICE CONTEXT")
    print("=" * 70)
    print(f"  Start: ${closes[0]:,.0f}")
    print(f"  End:   ${closes[-1]:,.0f}")
    print(f"  Change: {(closes[-1]/closes[0]-1)*100:+.1f}%")
    print(f"  Max:   ${np.max(closes):,.0f}")
    print(f"  Min:   ${np.min(closes):,.0f}")

    print(f"\n{'='*70}")
    print("DONE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
