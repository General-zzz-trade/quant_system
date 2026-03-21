#!/usr/bin/env python3
"""BTC/ETH Ratio Mean-Reversion Pair Trading — Parameter Sweep & Backtest.

Strategy: track BTC/ETH price ratio, trade deviations from rolling mean.
  - z > z_entry: ratio high -> short BTC, long ETH
  - z < -z_entry: ratio low -> long BTC, short ETH
  - exit when |z| < z_exit (mean reversion complete)
  - equal dollar notional on both legs

Usage:
    python3 -m scripts.research.pair_trading_research
    python3 -m scripts.research.pair_trading_research --fee-bps 4.0
"""
from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


# ── Configuration ──────────────────────────────────────────────

DATA_DIR = Path("data_files")
WF_RESULTS_DIR = Path("results/walkforward")

# Parameter sweep ranges
Z_ENTRY_GRID = [1.5, 2.0, 2.5]
Z_EXIT_GRID = [0.3, 0.5, 0.8]
MA_WINDOW_GRID = [480, 720, 960]

# Annualization factor: 8760 hourly bars per year
HOURS_PER_YEAR = 8760


@dataclass
class PairConfig:
    z_entry: float = 2.0
    z_exit: float = 0.5
    ma_window: int = 720
    fee_bps: float = 4.0       # per leg, taker
    slippage_bps: float = 1.0  # per leg
    notional: float = 10_000.0  # dollar notional per leg


@dataclass
class PairResult:
    z_entry: float
    z_exit: float
    ma_window: int
    sharpe: float
    total_return: float
    annual_return: float
    max_drawdown: float
    n_trades: int
    avg_hold_bars: float
    win_rate: float
    corr_btc_alpha: float


# ── Data Loading ───────────────────────────────────────────────

def load_aligned_prices() -> pd.DataFrame:
    """Load BTC and ETH 1h data, inner-join on open_time."""
    btc_path = DATA_DIR / "BTCUSDT_1h.csv"
    eth_path = DATA_DIR / "ETHUSDT_1h.csv"

    if not btc_path.exists():
        raise FileNotFoundError(f"BTC data not found: {btc_path}")
    if not eth_path.exists():
        raise FileNotFoundError(f"ETH data not found: {eth_path}")

    btc = pd.read_csv(btc_path, usecols=["open_time", "close"])
    eth = pd.read_csv(eth_path, usecols=["open_time", "close"])

    btc = btc.rename(columns={"close": "btc_close"})
    eth = eth.rename(columns={"close": "eth_close"})

    # Inner join on timestamp
    df = pd.merge(btc, eth, on="open_time", how="inner").sort_values("open_time").reset_index(drop=True)

    # Filter out placeholder rows (price == 0 or very low volume periods)
    df = df[(df["btc_close"] > 100) & (df["eth_close"] > 1)].reset_index(drop=True)

    print(f"Loaded {len(df)} aligned hourly bars "
          f"({df['open_time'].iloc[0]} -> {df['open_time'].iloc[-1]})")
    return df


# ── Backtest Engine ────────────────────────────────────────────

def compute_ratio_zscore(df: pd.DataFrame, ma_window: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute BTC/ETH ratio and its rolling z-score."""
    ratio = (df["btc_close"] / df["eth_close"]).values
    ratio_ma = pd.Series(ratio).rolling(ma_window, min_periods=ma_window).mean().values
    ratio_std = pd.Series(ratio).rolling(ma_window, min_periods=ma_window).std().values

    # Avoid division by zero
    ratio_std = np.where(ratio_std < 1e-10, np.nan, ratio_std)
    zscore = (ratio - ratio_ma) / ratio_std

    return ratio, zscore


def run_backtest(df: pd.DataFrame, cfg: PairConfig) -> tuple[PairResult, np.ndarray]:
    """Run pair trading backtest, return result + per-bar returns array."""
    ratio, zscore = compute_ratio_zscore(df, cfg.ma_window)
    n = len(df)

    btc_close = df["btc_close"].values
    eth_close = df["eth_close"].values

    # Cost per round-trip per leg (entry + exit)
    cost_per_leg = (cfg.fee_bps + cfg.slippage_bps) / 10_000.0

    # State tracking
    position = 0  # +1 = long ratio (long BTC, short ETH), -1 = short ratio
    entry_btc = 0.0
    entry_eth = 0.0
    entry_bar = 0

    # Per-bar P&L (as fraction of total notional = 2 * notional)
    bar_returns = np.zeros(n)
    trades: list[dict] = []

    for i in range(1, n):
        z = zscore[i]
        if np.isnan(z):
            continue

        # Mark-to-market existing position
        if position != 0:
            # BTC leg return
            btc_ret = (btc_close[i] / btc_close[i - 1] - 1.0)
            # ETH leg return
            eth_ret = (eth_close[i] / eth_close[i - 1] - 1.0)

            if position == 1:
                # Long BTC, short ETH
                bar_returns[i] = 0.5 * btc_ret - 0.5 * eth_ret
            else:
                # Short BTC, long ETH
                bar_returns[i] = -0.5 * btc_ret + 0.5 * eth_ret

        # Check exit
        if position != 0 and abs(z) < cfg.z_exit:
            # Exit: pay costs on both legs
            trade_return = sum(bar_returns[entry_bar + 1: i + 1])
            net_return = trade_return - 2.0 * cost_per_leg  # 2 legs, round-trip each
            trades.append({
                "entry_bar": entry_bar,
                "exit_bar": i,
                "side": position,
                "hold_bars": i - entry_bar,
                "gross_return": trade_return,
                "net_return": net_return,
            })
            # Deduct exit costs from this bar
            bar_returns[i] -= 2.0 * cost_per_leg
            position = 0

        # Check entry (only if flat)
        if position == 0:
            if z > cfg.z_entry:
                # Ratio high -> short BTC, long ETH (short the ratio)
                position = -1
                entry_btc = btc_close[i]
                entry_eth = eth_close[i]
                entry_bar = i
                # Deduct entry costs
                bar_returns[i] -= 2.0 * cost_per_leg
            elif z < -cfg.z_entry:
                # Ratio low -> long BTC, short ETH (long the ratio)
                position = 1
                entry_btc = btc_close[i]
                entry_eth = eth_close[i]
                entry_bar = i
                bar_returns[i] -= 2.0 * cost_per_leg

    # Close any open position at end
    if position != 0:
        trade_return = sum(bar_returns[entry_bar + 1: n])
        net_return = trade_return - 2.0 * cost_per_leg
        trades.append({
            "entry_bar": entry_bar,
            "exit_bar": n - 1,
            "side": position,
            "hold_bars": n - 1 - entry_bar,
            "gross_return": trade_return,
            "net_return": net_return,
        })
        bar_returns[n - 1] -= 2.0 * cost_per_leg

    # Compute stats
    equity = np.cumprod(1.0 + bar_returns)
    total_return = equity[-1] / equity[0] - 1.0

    n_years = n / HOURS_PER_YEAR
    annual_return = (1.0 + total_return) ** (1.0 / max(n_years, 0.01)) - 1.0

    # Sharpe (annualized)
    if bar_returns.std() > 0:
        sharpe = bar_returns.mean() / bar_returns.std() * np.sqrt(HOURS_PER_YEAR)
    else:
        sharpe = 0.0

    # Max drawdown
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak
    max_dd = drawdown.min()

    # Trade stats
    n_trades = len(trades)
    if n_trades > 0:
        avg_hold = np.mean([t["hold_bars"] for t in trades])
        win_rate = np.mean([1.0 if t["net_return"] > 0 else 0.0 for t in trades])
    else:
        avg_hold = 0.0
        win_rate = 0.0

    # Correlation with BTC directional alpha
    corr = _compute_btc_alpha_correlation(bar_returns, n)

    result = PairResult(
        z_entry=cfg.z_entry,
        z_exit=cfg.z_exit,
        ma_window=cfg.ma_window,
        sharpe=sharpe,
        total_return=total_return,
        annual_return=annual_return,
        max_drawdown=max_dd,
        n_trades=n_trades,
        avg_hold_bars=avg_hold,
        win_rate=win_rate,
        corr_btc_alpha=corr,
    )

    return result, bar_returns


def _compute_btc_alpha_correlation(pair_returns: np.ndarray, n_bars: int) -> float:
    """Compute correlation between pair strategy returns and BTC directional alpha.

    Uses per-fold returns from walk-forward results to build a proxy
    BTC alpha return series, then correlates with pair trading returns.
    """
    wf_path = WF_RESULTS_DIR / "wf_BTCUSDT.json"
    if not wf_path.exists():
        return np.nan

    try:
        with open(wf_path) as f:
            wf = json.load(f)

        fold_returns = wf.get("summary", {}).get("fold_returns", [])
        if not fold_returns:
            return np.nan

        # Build a coarse BTC alpha return series: spread each fold's return
        # evenly across its portion of the total bar count
        n_folds = len(fold_returns)
        bars_per_fold = n_bars // max(n_folds, 1)

        btc_alpha_returns = np.zeros(n_bars)
        for i, fold_ret in enumerate(fold_returns):
            start = i * bars_per_fold
            end = min((i + 1) * bars_per_fold, n_bars)
            if start < n_bars and end > start:
                # Spread fold return evenly across bars
                per_bar = fold_ret / max(end - start, 1)
                btc_alpha_returns[start:end] = per_bar

        # Compute correlation on rolling windows (monthly = 720 bars)
        # to get a meaningful signal correlation
        window = 720
        if n_bars < window * 2:
            # Not enough data, compute raw correlation
            valid = ~(np.isnan(pair_returns) | np.isnan(btc_alpha_returns))
            if valid.sum() > 100:
                return float(np.corrcoef(pair_returns[valid], btc_alpha_returns[valid])[0, 1])
            return np.nan

        # Rolling cumulative returns correlation
        pair_cum = pd.Series(pair_returns).rolling(window).sum().dropna().values
        alpha_cum = pd.Series(btc_alpha_returns).rolling(window).sum().dropna().values
        min_len = min(len(pair_cum), len(alpha_cum))
        if min_len > 10:
            return float(np.corrcoef(pair_cum[:min_len], alpha_cum[:min_len])[0, 1])

    except Exception:
        pass

    return np.nan


# ── Parameter Sweep ────────────────────────────────────────────

def run_parameter_sweep(df: pd.DataFrame, fee_bps: float, slippage_bps: float) -> list[PairResult]:
    """Run backtest across full parameter grid."""
    results: list[PairResult] = []
    combos = list(itertools.product(Z_ENTRY_GRID, Z_EXIT_GRID, MA_WINDOW_GRID))

    print(f"\nRunning {len(combos)} parameter combinations...\n")

    for z_entry, z_exit, ma_window in combos:
        # Skip invalid combos where exit >= entry
        if z_exit >= z_entry:
            continue

        cfg = PairConfig(
            z_entry=z_entry,
            z_exit=z_exit,
            ma_window=ma_window,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
        )
        result, _ = run_backtest(df, cfg)
        results.append(result)

    return results


def print_sweep_table(results: list[PairResult]) -> None:
    """Print formatted parameter sweep results table."""
    # Sort by Sharpe descending
    results = sorted(results, key=lambda r: r.sharpe, reverse=True)

    header = (
        f"{'z_entry':>8} {'z_exit':>7} {'MA_win':>7} | "
        f"{'Sharpe':>8} {'TotRet':>8} {'AnnRet':>8} {'MaxDD':>8} | "
        f"{'Trades':>7} {'AvgHold':>8} {'WinR':>6} {'Corr':>6}"
    )
    sep = "-" * len(header)

    print("\n" + sep)
    print("  BTC/ETH Pair Trading — Parameter Sweep Results")
    print(sep)
    print(header)
    print(sep)

    for r in results:
        corr_str = f"{r.corr_btc_alpha:6.3f}" if not np.isnan(r.corr_btc_alpha) else "   N/A"
        print(
            f"{r.z_entry:8.1f} {r.z_exit:7.1f} {r.ma_window:7d} | "
            f"{r.sharpe:8.2f} {r.total_return:7.1%} {r.annual_return:7.1%} {r.max_drawdown:7.1%} | "
            f"{r.n_trades:7d} {r.avg_hold_bars:8.1f} {r.win_rate:5.1%} {corr_str}"
        )

    print(sep)


def print_best_config(df: pd.DataFrame, results: list[PairResult], fee_bps: float, slippage_bps: float) -> None:
    """Print detailed stats for the best configuration."""
    best = max(results, key=lambda r: r.sharpe)

    print("\n" + "=" * 70)
    print("  BEST CONFIGURATION")
    print("=" * 70)
    print(f"  z_entry:    {best.z_entry}")
    print(f"  z_exit:     {best.z_exit}")
    print(f"  ma_window:  {best.ma_window}")
    print(f"  fee_bps:    {fee_bps}")
    print(f"  slippage:   {slippage_bps} bps")
    print("=" * 70)

    # Re-run to get full equity curve
    cfg = PairConfig(
        z_entry=best.z_entry,
        z_exit=best.z_exit,
        ma_window=best.ma_window,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
    )
    result, bar_returns = run_backtest(df, cfg)

    equity = np.cumprod(1.0 + bar_returns)
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak

    # Monthly returns
    n = len(bar_returns)
    bars_per_month = 720  # ~30 days
    n_months = n // bars_per_month
    monthly_returns = []
    for m in range(n_months):
        start = m * bars_per_month
        end = (m + 1) * bars_per_month
        month_ret = np.prod(1.0 + bar_returns[start:end]) - 1.0
        monthly_returns.append(month_ret)

    monthly_returns = np.array(monthly_returns)
    positive_months = (monthly_returns > 0).sum()

    print(f"\n  Performance Summary:")
    print(f"  {'Sharpe Ratio:':<25} {result.sharpe:.2f}")
    print(f"  {'Total Return:':<25} {result.total_return:.1%}")
    print(f"  {'Annual Return:':<25} {result.annual_return:.1%}")
    print(f"  {'Max Drawdown:':<25} {result.max_drawdown:.1%}")
    print(f"  {'Number of Trades:':<25} {result.n_trades}")
    print(f"  {'Avg Hold (bars):':<25} {result.avg_hold_bars:.1f}")
    print(f"  {'Win Rate:':<25} {result.win_rate:.1%}")
    print(f"  {'Positive Months:':<25} {positive_months}/{n_months} "
          f"({positive_months / max(n_months, 1):.0%})")

    if len(monthly_returns) > 0:
        print(f"  {'Monthly Return (median):':<25} {np.median(monthly_returns):.2%}")
        print(f"  {'Monthly Return (worst):':<25} {monthly_returns.min():.2%}")
        print(f"  {'Monthly Return (best):':<25} {monthly_returns.max():.2%}")

    corr_str = f"{result.corr_btc_alpha:.3f}" if not np.isnan(result.corr_btc_alpha) else "N/A"
    print(f"\n  {'Corr with BTC Alpha:':<25} {corr_str}")

    if not np.isnan(result.corr_btc_alpha):
        if abs(result.corr_btc_alpha) < 0.3:
            print("  -> LOW correlation: good diversifier for BTC directional alpha")
        elif abs(result.corr_btc_alpha) < 0.6:
            print("  -> MODERATE correlation: partial diversification benefit")
        else:
            print("  -> HIGH correlation: limited diversification benefit")

    # Pass/Fail assessment
    print(f"\n  Assessment:")
    if result.sharpe >= 1.0 and result.max_drawdown > -0.20:
        print("  PASS — viable strategy (Sharpe >= 1.0, MaxDD > -20%)")
    elif result.sharpe >= 0.5:
        print("  WEAK — marginal (Sharpe 0.5-1.0), needs further optimization")
    else:
        print("  FAIL — insufficient risk-adjusted returns (Sharpe < 0.5)")

    print("=" * 70)


# ── Main ───────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="BTC/ETH pair trading ratio mean-reversion research"
    )
    parser.add_argument("--fee-bps", type=float, default=4.0,
                        help="Taker fee per leg in bps (default: 4.0)")
    parser.add_argument("--slippage-bps", type=float, default=1.0,
                        help="Slippage per leg in bps (default: 1.0)")
    args = parser.parse_args()

    print("=" * 70)
    print("  BTC/ETH Ratio Mean-Reversion Pair Trading Research")
    print("=" * 70)
    print(f"  Fee: {args.fee_bps} bps/leg  |  Slippage: {args.slippage_bps} bps/leg")
    print(f"  Round-trip cost per trade: {2 * (args.fee_bps + args.slippage_bps):.1f} bps (2 legs)")

    # Load data
    df = load_aligned_prices()

    # Show ratio stats
    ratio = (df["btc_close"] / df["eth_close"]).values
    valid_ratio = ratio[~np.isnan(ratio)]
    print(f"\n  BTC/ETH Ratio Stats:")
    print(f"    Mean:   {np.mean(valid_ratio):.2f}")
    print(f"    Std:    {np.std(valid_ratio):.2f}")
    print(f"    Min:    {np.min(valid_ratio):.2f}")
    print(f"    Max:    {np.max(valid_ratio):.2f}")
    print(f"    Latest: {valid_ratio[-1]:.2f}")

    # Run parameter sweep
    results = run_parameter_sweep(df, args.fee_bps, args.slippage_bps)

    if not results:
        print("No valid parameter combinations to test.")
        return

    # Print results
    print_sweep_table(results)
    print_best_config(df, results, args.fee_bps, args.slippage_bps)


if __name__ == "__main__":
    main()
