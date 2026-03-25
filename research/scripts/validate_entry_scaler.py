#!/usr/bin/env python3
"""Validate the Entry Scaler integration with walk-forward backtest.

Compares:
  A. Baseline (entry_scaler disabled)
  B. Entry Scaler enabled (BB-based position scaling)

Uses the same signal pipeline as production (z-score → deadzone → min_hold)
with proper WF expanding-window folds.
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import json
import numpy as np
import pandas as pd
from pathlib import Path

MODEL_DIR = Path("models_v8")
DATA_DIR = Path("data_files")

FEE = 0.0004
SLIPPAGE = 0.0003
COST = FEE + SLIPPAGE


def load_1h(symbol: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / f"{symbol}_1h.csv")
    df["datetime"] = pd.to_datetime(df["open_time"], unit="ms")
    df = df.sort_values("open_time").reset_index(drop=True)
    return df


def generate_signals(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Generate 1h signals matching production pipeline."""
    model_dir = "BTCUSDT_gate_v2" if "BTC" in symbol else "ETHUSDT_gate_v2"
    cfg = json.load(open(MODEL_DIR / model_dir / "config.json"))
    dz = cfg.get("deadzone", 1.0)
    min_hold = cfg.get("min_hold", 24)
    max_hold = cfg.get("max_hold", 144)
    long_only = cfg.get("long_only", False)
    monthly_gate = cfg.get("monthly_gate", False)

    close = df["close"]

    # Momentum composite prediction proxy
    feats = pd.DataFrame(index=df.index)
    for w in [5, 10, 20, 50]:
        feats[f"ret_{w}"] = close.pct_change(w)

    pred = (feats["ret_5"] * 0.3 + feats["ret_10"] * 0.3 +
            feats["ret_20"] * 0.2 + feats["ret_50"] * 0.2)

    # Z-score
    z = (pred - pred.rolling(720, min_periods=180).mean()) / \
        pred.rolling(720, min_periods=180).std()

    # Discretize
    signal = pd.Series(0, index=df.index)
    signal[z > dz] = 1
    signal[z < -dz] = -1

    if long_only:
        signal[signal < 0] = 0

    if monthly_gate:
        sma480 = close.rolling(480).mean()
        signal[close < sma480] = signal[close < sma480].clip(upper=0)

    # Min hold
    current_sig = 0
    hold_count = 0
    for i in range(len(signal)):
        new_sig = signal.iloc[i]
        if new_sig != current_sig and new_sig != 0:
            current_sig = new_sig
            hold_count = 1
        elif current_sig != 0 and hold_count < min_hold:
            signal.iloc[i] = current_sig
            hold_count += 1
        elif new_sig != current_sig:
            current_sig = new_sig
            hold_count = 1 if new_sig != 0 else 0

    # Max hold
    current_sig = 0
    hold_count = 0
    for i in range(len(signal)):
        if signal.iloc[i] != 0:
            if signal.iloc[i] == current_sig:
                hold_count += 1
                if hold_count > max_hold:
                    signal.iloc[i] = 0
                    current_sig = 0
                    hold_count = 0
            else:
                current_sig = signal.iloc[i]
                hold_count = 1
        else:
            current_sig = 0
            hold_count = 0

    # Compute BB position for entry scaling
    w = 12
    ma = close.rolling(w).mean()
    std = close.rolling(w).std()
    bb_pos = (close - ma) / std.replace(0, np.nan)

    out = df[["datetime", "close", "high", "low"]].copy()
    out["signal"] = signal
    out["z"] = z
    out["bb_pos"] = bb_pos

    return out


def compute_entry_scale(bb_pos: float, direction: int) -> float:
    """Mirror of AlphaRunner._compute_entry_scale."""
    if direction == 1:  # long
        if bb_pos < -1.0:
            return 1.2
        elif bb_pos < -0.5:
            return 1.0
        elif bb_pos < 0:
            return 0.7
        elif bb_pos < 0.5:
            return 0.5
        else:
            return 0.3
    elif direction == -1:  # short
        if bb_pos > 1.0:
            return 1.2
        elif bb_pos > 0.5:
            return 1.0
        elif bb_pos > 0:
            return 0.7
        elif bb_pos > -0.5:
            return 0.5
        else:
            return 0.3
    return 1.0


def backtest_with_scaler(data: pd.DataFrame, leverage: float, use_scaler: bool) -> dict:
    """Run backtest with or without entry scaler."""
    equity = 500.0  # start small
    peak_equity = equity
    max_dd = 0.0
    trades = []
    pos = 0
    entry_price = 0.0
    entry_scale = 1.0

    equity_curve = []

    for i in range(1, len(data)):
        new_sig = data["signal"].iloc[i]
        close = data["close"].iloc[i]
        bb = data["bb_pos"].iloc[i] if pd.notna(data["bb_pos"].iloc[i]) else 0

        if new_sig != pos:
            # Close existing
            if pos != 0 and entry_price > 0:
                raw_ret = pos * (close / entry_price - 1)
                net_ret = raw_ret - 2 * COST  # round trip cost
                pnl = equity * leverage * entry_scale * abs(net_ret) * np.sign(net_ret)
                equity += pnl
                trades.append({
                    "ret": net_ret,
                    "pnl": pnl,
                    "scale": entry_scale,
                    "direction": pos,
                })

            # Open new
            if new_sig != 0:
                pos = new_sig
                entry_price = close
                entry_scale = compute_entry_scale(bb, new_sig) if use_scaler else 1.0
            else:
                pos = 0
                entry_price = 0.0
                entry_scale = 1.0

        peak_equity = max(peak_equity, equity)
        dd = (equity - peak_equity) / peak_equity * 100 if peak_equity > 0 else 0
        max_dd = min(max_dd, dd)
        equity_curve.append(equity)

    if not trades:
        return {"sharpe": 0, "total_ret": 0, "max_dd": 0, "n_trades": 0,
                "win_rate": 0, "final_equity": equity, "avg_scale": 1.0}

    # Daily PnL for Sharpe
    n_days = (data["datetime"].iloc[-1] - data["datetime"].iloc[0]).days
    daily_ret = (equity / 500.0) ** (365 / max(n_days, 1)) - 1  # annualized

    rets = [t["ret"] * leverage for t in trades]
    sharpe_trade = np.mean(rets) / np.std(rets) * np.sqrt(len(trades) / max(n_days / 365, 0.1)) if np.std(rets) > 0 else 0

    # Compute daily equity returns
    eq_series = pd.Series(equity_curve)
    daily_eq = eq_series.iloc[::24]  # sample every 24 bars (daily)
    daily_rets = daily_eq.pct_change().dropna()
    sharpe = daily_rets.mean() / daily_rets.std() * np.sqrt(365) if daily_rets.std() > 0 else 0

    win_rate = sum(1 for t in trades if t["ret"] > 0) / len(trades) * 100
    avg_scale = np.mean([t["scale"] for t in trades])

    return {
        "sharpe": sharpe,
        "total_ret": (equity / 500.0 - 1) * 100,
        "max_dd": max_dd,
        "n_trades": len(trades),
        "win_rate": win_rate,
        "final_equity": equity,
        "avg_scale": avg_scale,
    }


def walkforward_test(data: pd.DataFrame, symbol: str, leverage: float = 10.0):
    """Walk-forward with expanding window."""
    print(f"\n{'='*70}")
    print(f"  {symbol} — Walk-Forward Entry Scaler Validation ({leverage}x leverage)")
    print(f"{'='*70}")

    total_bars = len(data)
    min_train = 4000  # ~167 days warmup
    fold_size = 720   # 30 days per fold
    n_folds = (total_bars - min_train) // fold_size

    print(f"  Total bars: {total_bars}, Folds: {n_folds}")

    results_base = []
    results_scaler = []

    for fold in range(n_folds):
        oos_start = min_train + fold * fold_size
        oos_end = min(oos_start + fold_size, total_bars)
        fold_data = data.iloc[oos_start:oos_end].reset_index(drop=True)

        if len(fold_data) < 100:
            continue

        r_base = backtest_with_scaler(fold_data, leverage, use_scaler=False)
        r_scale = backtest_with_scaler(fold_data, leverage, use_scaler=True)

        results_base.append(r_base)
        results_scaler.append(r_scale)

    # Aggregate
    def agg(results, label):
        sharpes = [r["sharpe"] for r in results if r["n_trades"] > 0]
        rets = [r["total_ret"] for r in results if r["n_trades"] > 0]
        dds = [r["max_dd"] for r in results if r["n_trades"] > 0]
        wrs = [r["win_rate"] for r in results if r["n_trades"] > 0]
        scales = [r["avg_scale"] for r in results if r["n_trades"] > 0]
        trades = [r["n_trades"] for r in results]

        n_pass = sum(1 for s in sharpes if s > 0)

        print(f"\n  {label}:")
        print(f"    Folds with trades: {len(sharpes)}/{len(results)}")
        print(f"    PASS (Sharpe>0): {n_pass}/{len(sharpes)}")
        print(f"    Mean Sharpe: {np.mean(sharpes):.2f} ± {np.std(sharpes):.2f}")
        print(f"    Median Sharpe: {np.median(sharpes):.2f}")
        print(f"    Mean Return: {np.mean(rets):.1f}%")
        print(f"    Mean MaxDD: {np.mean(dds):.1f}%")
        print(f"    Worst MaxDD: {min(dds):.1f}%")
        print(f"    Mean WR: {np.mean(wrs):.1f}%")
        print(f"    Avg Scale: {np.mean(scales):.2f}")
        print(f"    Total Trades: {sum(trades)}")
        return {
            "mean_sharpe": np.mean(sharpes),
            "mean_dd": np.mean(dds),
            "worst_dd": min(dds),
            "pass_rate": n_pass / len(sharpes) if sharpes else 0,
            "mean_ret": np.mean(rets),
        }

    r1 = agg(results_base, "BASELINE (no scaler)")
    r2 = agg(results_scaler, "ENTRY SCALER (BB-based)")

    # Improvement
    print("\n  --- Improvement ---")
    sharpe_chg = (r2["mean_sharpe"] / r1["mean_sharpe"] - 1) * 100 if r1["mean_sharpe"] != 0 else 0
    dd_chg = r2["mean_dd"] - r1["mean_dd"]
    worst_dd_chg = r2["worst_dd"] - r1["worst_dd"]
    print(f"    Sharpe: {sharpe_chg:+.1f}%")
    print(f"    Mean MaxDD: {dd_chg:+.1f}% (positive = less drawdown)")
    print(f"    Worst MaxDD: {worst_dd_chg:+.1f}%")

    # Full period backtest
    print("\n  --- Full Period Backtest ---")
    full_base = backtest_with_scaler(data, leverage, use_scaler=False)
    full_scale = backtest_with_scaler(data, leverage, use_scaler=True)

    print(f"    {'Metric':<20} {'Baseline':>12} {'Scaler':>12} {'Change':>12}")
    print(f"    {'-'*56}")
    metrics = [
        ("Sharpe", "sharpe", ".2f"),
        ("Total Return %", "total_ret", ".1f"),
        ("Max Drawdown %", "max_dd", ".1f"),
        ("Win Rate %", "win_rate", ".1f"),
        ("Trades", "n_trades", ".0f"),
        ("Avg Scale", "avg_scale", ".2f"),
        ("Final Equity $", "final_equity", ".0f"),
    ]
    for name, key, fmt in metrics:
        v1 = full_base[key]
        v2 = full_scale[key]
        diff = v2 - v1
        print(f"    {name:<20} {v1:>12{fmt}} {v2:>12{fmt}} {diff:>+12{fmt}}")


def main():
    leverage = 10.0

    for symbol in ["BTCUSDT", "ETHUSDT"]:
        df = load_1h(symbol)
        data = generate_signals(df, symbol)

        # Skip warmup
        data = data.iloc[800:].reset_index(drop=True)
        print(f"\n  {symbol}: {len(data)} bars after warmup")

        walkforward_test(data, symbol, leverage)


if __name__ == "__main__":
    main()
