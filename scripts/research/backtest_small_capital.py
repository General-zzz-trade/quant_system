#!/usr/bin/env python3
"""Small Capital High Leverage Backtest — simulate real growth from $100-$500.

Tests the production alpha (BTC + ETH) with realistic conditions:
  - Initial equity: $100, $200, $500
  - Leverage: 5x, 10x, 20x
  - Realistic costs: fees, slippage, spread, funding
  - ATR adaptive stop-loss
  - Monthly gate (BTC only)
  - Intra-bar stop (checks high/low, not just close)
  - Liquidation simulation
  - Min order $5 (Bybit)

Usage:
    python3 -m scripts.research.backtest_small_capital
    python3 -m scripts.research.backtest_small_capital --equity 100 --leverage 10
    python3 -m scripts.research.backtest_small_capital --sweep
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from execution.sim.realistic_backtest import (
    BacktestConfig,
    BacktestResult,
    run_realistic_backtest,
)
from scripts.shared.signal_postprocess import pred_to_signal

log = logging.getLogger("small_capital")


def _load_funding(symbol: str) -> Dict[int, float]:
    """Load funding rate schedule."""
    path = Path(f"data_files/{symbol}_funding.csv")
    if not path.exists():
        return {}
    schedule = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                schedule[int(row["timestamp"])] = float(row["funding_rate"])
            except (ValueError, KeyError):
                pass
    return schedule


def _align_funding_to_bars(
    timestamps: np.ndarray, funding_schedule: Dict[int, float]
) -> np.ndarray:
    """Align funding rates to bar timestamps (forward-fill)."""
    n = len(timestamps)
    rates = np.zeros(n)
    sorted_ts = sorted(funding_schedule.keys())
    idx = 0
    last_rate = 0.0
    for i in range(n):
        ts = int(timestamps[i])
        while idx < len(sorted_ts) and sorted_ts[idx] <= ts:
            last_rate = funding_schedule[sorted_ts[idx]]
            idx += 1
        rates[i] = last_rate
    return rates


def generate_signal(
    symbol: str,
    df: pd.DataFrame,
    feat_df: pd.DataFrame,
    deadzone: float,
    min_hold: int,
    max_hold: int,
    long_only: bool,
    monthly_gate: bool = False,
    monthly_gate_window: int = 480,
) -> np.ndarray:
    """Generate signal using multi-feature prediction proxy."""
    closes = df["close"].values.astype(float)
    n = len(feat_df)
    pred = np.zeros(n, dtype=float)

    # Feature ensemble (proxy for Ridge+LGBM model)
    weights = {
        "ret_24": 0.3,
        "rsi_14": -0.2,
        "macd_hist": 0.15,
        "close_vs_ma20": 0.15,
        "funding_zscore_24": -0.1,
        "cvd_20": 0.1,
    }
    for feat, w in weights.items():
        if feat in feat_df.columns:
            vals = feat_df[feat].values.astype(float)
            vals = np.nan_to_num(vals, 0.0)
            mean = np.nanmean(vals)
            std = np.nanstd(vals)
            if std > 1e-8:
                pred += w * (vals - mean) / std

    pred = np.nan_to_num(pred, 0.0)

    signal = np.asarray(
        pred_to_signal(
            pred, deadzone=deadzone, min_hold=min_hold,
            zscore_window=720, zscore_warmup=180, long_only=long_only,
            max_hold=max_hold,
        ),
        dtype=float,
    )

    # Monthly gate: zero signal when close < SMA(window)
    if monthly_gate and len(closes) == len(signal):
        sma = pd.Series(closes).rolling(monthly_gate_window).mean().values
        for i in range(len(signal)):
            if not np.isnan(sma[i]) and closes[i] < sma[i]:
                signal[i] = 0.0

    return signal


def run_single_backtest(
    symbol: str,
    initial_equity: float,
    leverage: float,
    adaptive_stop: bool = True,
) -> Dict:
    """Run realistic backtest for a single symbol+config."""
    from scripts.train_v7_alpha import _load_and_compute_features

    df = pd.read_csv(f"data_files/{symbol}_1h.csv")
    feat_df = _load_and_compute_features(symbol, df)

    closes = df["close"].values.astype(float)
    highs = df["high"].values.astype(float)
    lows = df["low"].values.astype(float)
    volumes = df["volume"].values.astype(float)
    timestamps = df["open_time"].values.astype(float)

    # Production params
    if symbol == "BTCUSDT":
        deadzone, min_hold, max_hold = 1.0, 24, 144
        monthly_gate = True
    else:
        deadzone, min_hold, max_hold = 0.4, 18, 60
        monthly_gate = False

    signal = generate_signal(
        symbol, df, feat_df, deadzone, min_hold, max_hold,
        long_only=True, monthly_gate=monthly_gate,
    )

    # Load funding rates
    funding_sched = _load_funding(symbol)
    funding_rates = _align_funding_to_bars(timestamps, funding_sched)

    cfg = BacktestConfig(
        initial_equity=initial_equity,
        leverage=leverage,
        fee_bps=4.0,              # Bybit taker fee
        base_slippage_bps=1.0,
        spread_bps=1.0,
        min_order_notional=5.0,
        adaptive_stop=adaptive_stop,
        atr_stop_mult=1.2,
        atr_trail_trigger=0.5,
        atr_trail_step=0.2,
        atr_breakeven_trigger=0.5,
    )

    result = run_realistic_backtest(
        closes, highs, lows, volumes, signal,
        cfg=cfg, funding_rates=funding_rates,
    )

    return {
        "symbol": symbol,
        "initial_equity": initial_equity,
        "leverage": leverage,
        "final_equity": round(result.equity_curve[-1], 2),
        "total_return_pct": round(result.total_return_pct, 1),
        "sharpe": round(result.sharpe, 2),
        "max_dd_pct": round(result.max_drawdown_pct, 1),
        "n_trades": result.n_trades,
        "win_rate": round(result.win_rate, 1),
        "n_liquidations": result.n_liquidations,
        "fees": round(result.total_fees, 2),
        "slippage": round(result.total_slippage, 2),
        "funding": round(result.total_funding, 2),
        "equity_curve": result.equity_curve,
    }


def print_results(results: List[Dict]) -> None:
    """Print formatted results table."""
    print()
    print("=" * 120)
    print("  Small Capital Realistic Backtest Results")
    print("=" * 120)
    print(f"  {'Symbol':>8} {'Equity':>7} {'Lev':>4} {'Final$':>9} {'Return':>8} "
          f"{'Sharpe':>7} {'MaxDD':>7} {'Trades':>7} {'WR%':>5} {'Liqs':>5} "
          f"{'Fees$':>7} {'Slip$':>7} {'Fund$':>7}")
    print(f"  {'-'*8} {'-'*7} {'-'*4} {'-'*9} {'-'*8} "
          f"{'-'*7} {'-'*7} {'-'*7} {'-'*5} {'-'*5} "
          f"{'-'*7} {'-'*7} {'-'*7}")

    for r in results:
        print(f"  {r['symbol']:>8} ${r['initial_equity']:>5.0f} {r['leverage']:>3.0f}x "
              f"${r['final_equity']:>7.0f} {r['total_return_pct']:>+7.1f}% "
              f"{r['sharpe']:>7.2f} {r['max_dd_pct']:>6.1f}% {r['n_trades']:>7} "
              f"{r['win_rate']:>4.1f}% {r['n_liquidations']:>5} "
              f"${r['fees']:>6.0f} ${r['slippage']:>6.0f} ${r['funding']:>6.0f}")

    print("=" * 120)

    # Growth milestones
    print()
    print("  Growth Milestones:")
    for r in results:
        eq = r["equity_curve"]
        n = len(eq)
        milestones = []
        for target in [200, 500, 1000, 5000, 10000]:
            idx = np.searchsorted(eq, target)
            if idx < n:
                days = idx / 24
                milestones.append(f"${target:,} in {days:.0f}d")
        if milestones:
            print(f"    {r['symbol']} ${r['initial_equity']:.0f}@{r['leverage']:.0f}x: "
                  f"{', '.join(milestones)}")
        else:
            print(f"    {r['symbol']} ${r['initial_equity']:.0f}@{r['leverage']:.0f}x: "
                  f"did not reach $200")

    # Bust analysis
    print()
    print("  Risk Analysis:")
    for r in results:
        eq = r["equity_curve"]
        min_eq = np.min(eq)
        bust_risk = min_eq < r["initial_equity"] * 0.1  # <10% remaining
        print(f"    {r['symbol']} ${r['initial_equity']:.0f}@{r['leverage']:.0f}x: "
              f"min equity ${min_eq:.0f} ({min_eq/r['initial_equity']*100:.0f}% of start), "
              f"bust risk: {'HIGH' if bust_risk else 'LOW'}, "
              f"liquidations: {r['n_liquidations']}")


def main():
    parser = argparse.ArgumentParser(description="Small Capital Backtest")
    parser.add_argument("--equity", type=float, default=None)
    parser.add_argument("--leverage", type=float, default=None)
    parser.add_argument("--sweep", action="store_true", help="Sweep equity × leverage grid")
    parser.add_argument("--symbol", default=None, help="Single symbol (default: BTC+ETH)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    symbols = [args.symbol] if args.symbol else ["BTCUSDT", "ETHUSDT"]

    if args.sweep:
        equities = [100, 200, 500]
        leverages = [5, 10, 20]
    elif args.equity and args.leverage:
        equities = [args.equity]
        leverages = [args.leverage]
    else:
        equities = [100, 500]
        leverages = [10, 20]

    results = []
    for symbol in symbols:
        log.info("Loading %s...", symbol)
        for eq in equities:
            for lev in leverages:
                log.info("  %s $%d @ %dx...", symbol, eq, lev)
                r = run_single_backtest(symbol, eq, lev)
                results.append(r)

    print_results(results)


if __name__ == "__main__":
    main()
