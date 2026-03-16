#!/usr/bin/env python3
"""Leverage Survival Simulation — Monte Carlo for small accounts.

Uses actual walk-forward fold returns to simulate compounding
under different leverage/stop-loss configurations.

Answers: at what leverage does the alpha stop being viable?
How fast can $500 grow? What's the bust probability?

Usage:
    python3 -m scripts.research.leverage_survival_sim
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def load_fold_returns(path: str = "results/walkforward/wf_ETHUSDT.json") -> list[float]:
    """Load per-fold returns from walk-forward results."""
    with open(path) as f:
        data = json.load(f)
    return [fold["total_return"] for fold in data["folds"]]


def simulate_paths(
    fold_returns: list[float],
    leverage: float,
    initial_equity: float,
    stop_loss_pct: float,
    n_sims: int = 10000,
    bars_per_fold: int = 2190,  # 3 months of 1h bars
) -> dict:
    """Monte Carlo: sample fold returns with replacement, compound with leverage."""
    rng = np.random.RandomState(42)
    n_folds = len(fold_returns)

    # Convert fold returns to per-bar returns (approximate)
    # Each fold covers ~2190 bars, but trades are intermittent
    # Use fold return directly as the 3-month block return
    fold_arr = np.array(fold_returns)

    # For leverage: scale returns (but also scale losses)
    # Effective return = leverage × base_return - leverage × (leverage-1) × funding_drag
    # Approximate funding drag: ~0.01% per 8h = ~0.03% per day = ~2.7% per quarter
    quarterly_funding_drag = 0.027 * (leverage - 1)  # only on leveraged portion

    results = {
        "leverage": leverage,
        "initial_equity": initial_equity,
        "stop_loss_pct": stop_loss_pct,
    }

    final_equities = []
    max_drawdowns = []
    bust_count = 0
    time_to_10x = []
    time_to_100x = []

    # Simulate 20 folds forward (5 years) per path
    sim_folds = 20

    for sim in range(n_sims):
        equity = initial_equity
        peak = equity
        max_dd = 0
        reached_10x = False
        reached_100x = False

        for fold_i in range(sim_folds):
            # Random fold return (bootstrap)
            base_return = fold_arr[rng.randint(0, n_folds)]

            # Leverage scales return
            leveraged_return = base_return * leverage

            # Funding drag (approximate)
            leveraged_return -= quarterly_funding_drag

            # Stop-loss cap: can't lose more than stop_loss × leverage per trade
            # In a fold, multiple trades happen. Approximate: cap fold loss
            # at -equity × position_fraction × leverage × stop_loss_pct × n_trades_per_fold
            # Conservative: cap fold loss at -90% (can't go below 10% of equity)
            fold_pnl = equity * leveraged_return
            fold_pnl = max(fold_pnl, -equity * 0.90)  # floor: keep 10%

            equity += fold_pnl
            equity = max(equity, 0.01)  # can't go negative

            if equity < initial_equity * 0.05:  # bust = below 5% of initial
                bust_count += 1
                break

            peak = max(peak, equity)
            dd = (peak - equity) / peak
            max_dd = max(max_dd, dd)

            if not reached_10x and equity >= initial_equity * 10:
                reached_10x = True
                time_to_10x.append((fold_i + 1) * 3)  # months

            if not reached_100x and equity >= initial_equity * 100:
                reached_100x = True
                time_to_100x.append((fold_i + 1) * 3)

        final_equities.append(equity)
        max_drawdowns.append(max_dd)

    final_arr = np.array(final_equities)
    dd_arr = np.array(max_drawdowns)

    results["median_final"] = float(np.median(final_arr))
    results["p10_final"] = float(np.percentile(final_arr, 10))
    results["p90_final"] = float(np.percentile(final_arr, 90))
    results["mean_final"] = float(np.mean(final_arr))
    results["bust_rate"] = bust_count / n_sims * 100
    results["median_max_dd"] = float(np.median(dd_arr)) * 100
    results["p95_max_dd"] = float(np.percentile(dd_arr, 95)) * 100
    results["pct_profitable"] = float(np.mean(final_arr > initial_equity)) * 100
    results["pct_10x"] = len(time_to_10x) / n_sims * 100
    results["median_months_to_10x"] = float(np.median(time_to_10x)) if time_to_10x else 999
    results["pct_100x"] = len(time_to_100x) / n_sims * 100
    results["median_months_to_100x"] = float(np.median(time_to_100x)) if time_to_100x else 999

    return results


def main() -> None:
    print(f"\n{'='*85}")
    print("  LEVERAGE SURVIVAL SIMULATION — ETH Alpha (Sharpe 1.52, mh=18)")
    print(f"{'='*85}")

    fold_returns = load_fold_returns()
    print(f"  Fold returns: {len(fold_returns)} folds")
    print(f"  Mean fold return: {np.mean(fold_returns)*100:+.1f}%")
    print(f"  Std fold return: {np.std(fold_returns)*100:.1f}%")
    print(f"  Positive folds: {sum(1 for r in fold_returns if r > 0)}/{len(fold_returns)}")

    initial = 500.0  # $500 starting capital
    leverages = [1, 2, 3, 5, 7, 10, 15, 20]

    print(f"\n  Starting capital: ${initial:.0f}")
    print(f"  Simulation: 20 folds forward (5 years), 10K paths per leverage")

    print(f"\n  {'Lev':>4} {'Median$':>10} {'P10$':>10} {'P90$':>10} "
          f"{'Bust%':>6} {'MaxDD':>6} {'Prof%':>6} "
          f"{'→10x%':>6} {'→10x_mo':>8} {'→100x%':>7} {'→100x_mo':>9}")
    print(f"  {'-'*100}")

    for lev in leverages:
        res = simulate_paths(fold_returns, leverage=lev, initial_equity=initial,
                             stop_loss_pct=0.02)
        print(f"  {lev:>4}x {res['median_final']:>9,.0f} {res['p10_final']:>9,.0f} "
              f"{res['p90_final']:>9,.0f} {res['bust_rate']:>5.1f}% "
              f"{res['median_max_dd']:>5.0f}% {res['pct_profitable']:>5.1f}% "
              f"{res['pct_10x']:>5.1f}% {res['median_months_to_10x']:>7.0f}mo "
              f"{res['pct_100x']:>6.1f}% {res['median_months_to_100x']:>8.0f}mo")

    # Risk-adjusted optimal
    print(f"\n  {'='*85}")
    print("  LEVERAGE vs RISK ANALYSIS")
    print(f"  {'='*85}")

    print("\n  Key insights:")
    for lev in [3, 5, 10]:
        res = simulate_paths(fold_returns, leverage=lev, initial_equity=initial,
                             stop_loss_pct=0.02)
        kelly_approx = np.mean(fold_returns) / np.var(fold_returns) if np.var(fold_returns) > 0 else 1
        print(f"\n  {lev}x leverage:")
        print(f"    5-year median: ${res['median_final']:,.0f} ({res['median_final']/initial:.0f}x)")
        print(f"    Bust probability: {res['bust_rate']:.1f}%")
        print(f"    Worst-case DD (95th pct): {res['p95_max_dd']:.0f}%")
        print(f"    Probability of 10x: {res['pct_10x']:.1f}%")
        if res['pct_10x'] > 0:
            print(f"    Median time to 10x: {res['median_months_to_10x']:.0f} months")

    print(f"\n  Kelly optimal leverage (full Kelly): {kelly_approx:.1f}x")
    print(f"  Recommended (half-Kelly): {kelly_approx/2:.1f}x")


if __name__ == "__main__":
    main()
