#!/usr/bin/env python3
"""Altcoin Rotation Strategy — Cross-Sectional Momentum Walk-Forward.

Buys last period's winner, sells last period's loser.
Tests multiple rebalance frequencies and universe sizes.

Walk-forward: expanding window, 3-month test periods.

Usage:
    python3 -m scripts.research.altcoin_rotation
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


DATA_DIR = Path("data_files")
COST_BPS = 8  # roundtrip cost per trade


def load_universe() -> Dict[str, pd.DataFrame]:
    """Load all available 1h CSVs, return {symbol: DataFrame}."""
    universe = {}
    for f in sorted(DATA_DIR.glob("*_1h.csv")):
        sym = f.stem.replace("_1h", "")
        if sym.endswith("_oi") or sym.endswith("_spot"):
            continue
        try:
            df = pd.read_csv(f)
            if len(df) < 2000:  # need at least ~3 months
                continue
            universe[sym] = df
        except Exception as e:
            logger.debug("Failed to load data for %s: %s", sym, e)
    return universe


def align_returns(universe: Dict[str, pd.DataFrame], freq_hours: int = 24
                  ) -> Tuple[pd.DataFrame, pd.Index]:
    """Build aligned return matrix: each column = symbol, each row = period return.

    freq_hours: rebalance frequency (24=daily, 168=weekly).
    """
    # Find common timestamp range
    all_ts = set()
    ts_col = {}
    for sym, df in universe.items():
        col = "open_time" if "open_time" in df.columns else "timestamp"
        ts_col[sym] = col
        all_ts.update(df[col].values)

    # Build close price matrix aligned on timestamps
    all_ts_sorted = sorted(all_ts)
    ts_to_idx = {t: i for i, t in enumerate(all_ts_sorted)}
    n_ts = len(all_ts_sorted)

    prices = np.full((n_ts, len(universe)), np.nan)
    symbols = list(universe.keys())

    for j, sym in enumerate(symbols):
        df = universe[sym]
        col = ts_col[sym]
        for _, row in df.iterrows():
            t = row[col]
            if t in ts_to_idx:
                prices[ts_to_idx[t], j] = row["close"]

    # Forward-fill NaN (missing bars)
    for j in range(prices.shape[1]):
        last = np.nan
        for i in range(prices.shape[0]):
            if np.isnan(prices[i, j]):
                prices[i, j] = last
            else:
                last = prices[i, j]

    # Compute period returns (freq_hours bars apart)
    step = freq_hours
    n_periods = (n_ts - step) // step
    ret_matrix = np.full((n_periods, len(symbols)), np.nan)
    period_timestamps = []

    for p in range(n_periods):
        start_idx = p * step
        end_idx = start_idx + step
        if end_idx >= n_ts:
            break
        for j in range(len(symbols)):
            p_start = prices[start_idx, j]
            p_end = prices[end_idx, j]
            if p_start > 0 and p_end > 0 and not np.isnan(p_start) and not np.isnan(p_end):
                ret_matrix[p, j] = p_end / p_start - 1
        period_timestamps.append(all_ts_sorted[end_idx])

    ret_df = pd.DataFrame(ret_matrix, columns=symbols)
    return ret_df, pd.Index(period_timestamps)


def run_rotation(
    ret_df: pd.DataFrame,
    timestamps: pd.Index,
    lookback: int = 1,
    n_long: int = 3,
    n_short: int = 3,
    min_symbols: int = 8,
    cost_bps: float = COST_BPS,
) -> Dict:
    """Run rotation strategy: long top-N, short bottom-N by lookback return.

    lookback: number of periods to compute momentum (1=last period only).
    n_long/n_short: how many symbols to hold long/short.
    """
    n_periods = len(ret_df)
    symbols = list(ret_df.columns)
    len(symbols)

    pnls = []
    long_picks = []
    short_picks = []
    active_counts = []

    for t in range(lookback, n_periods):
        # Compute momentum: average return over lookback periods
        momentum = ret_df.iloc[t - lookback:t].mean()

        # Only use symbols with valid data in both lookback and current period
        valid = ~momentum.isna() & ~ret_df.iloc[t].isna()
        valid_syms = momentum[valid].sort_values()

        if len(valid_syms) < min_symbols:
            continue

        active_counts.append(len(valid_syms))

        # Long top-N, short bottom-N
        longs = valid_syms.index[-n_long:].tolist()
        shorts = valid_syms.index[:n_short].tolist()

        # Equal weight within each leg
        long_ret = ret_df.iloc[t][longs].mean()
        short_ret = ret_df.iloc[t][shorts].mean()

        # Long-short PnL
        gross_pnl = long_ret - short_ret

        # Cost: assume full turnover each period (worst case)
        turnover_cost = (n_long + n_short) * 2 * cost_bps / 10000
        net_pnl = gross_pnl - turnover_cost

        pnls.append(net_pnl)
        long_picks.append(longs)
        short_picks.append(shorts)

    if not pnls:
        return {"sharpe": 0, "total_return": 0, "trades": 0}

    pnls = np.array(pnls)
    total_ret = np.sum(pnls)
    avg_pnl = np.mean(pnls)
    std_pnl = np.std(pnls)
    win_rate = np.mean(pnls > 0)

    return {
        "n_periods": len(pnls),
        "total_return": total_ret * 100,
        "avg_pnl_pct": avg_pnl * 100,
        "std_pnl_pct": std_pnl * 100,
        "sharpe": avg_pnl / std_pnl * np.sqrt(365) if std_pnl > 0 else 0,  # annualized
        "win_rate": win_rate,
        "max_dd": _max_drawdown(pnls),
        "avg_active": np.mean(active_counts) if active_counts else 0,
        "long_picks": long_picks[-3:] if long_picks else [],
        "short_picks": short_picks[-3:] if short_picks else [],
    }


def _max_drawdown(pnls: np.ndarray) -> float:
    equity = np.cumsum(pnls) + 1.0
    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / peak
    return float(np.max(dd))


def walk_forward(
    ret_df: pd.DataFrame,
    timestamps: pd.Index,
    freq_name: str,
    lookback: int = 1,
    n_long: int = 3,
    n_short: int = 3,
    train_periods: int = 90,
    test_periods: int = 90,
) -> List[Dict]:
    """Walk-forward validation with expanding window."""
    n = len(ret_df)
    results = []
    fold = 0
    start = 0

    while start + train_periods + test_periods <= n:
        test_start = start + train_periods
        test_end = test_start + test_periods

        test_ret = ret_df.iloc[test_start:test_end]
        test_ts = timestamps[test_start:test_end]

        # Run strategy on test period (using momentum from lookback before test)
        test_pnls = []
        for t in range(lookback, len(test_ret)):
            global_t = test_start + t
            if global_t >= n:
                break

            momentum = ret_df.iloc[global_t - lookback:global_t].mean()
            valid = ~momentum.isna() & ~ret_df.iloc[global_t].isna()
            valid_syms = momentum[valid].sort_values()

            if len(valid_syms) < 6:
                continue

            longs = valid_syms.index[-n_long:].tolist()
            shorts = valid_syms.index[:n_short].tolist()

            long_ret = ret_df.iloc[global_t][longs].mean()
            short_ret = ret_df.iloc[global_t][shorts].mean()
            gross = long_ret - short_ret
            cost = (n_long + n_short) * 2 * COST_BPS / 10000
            test_pnls.append(gross - cost)

        if len(test_pnls) < 10:
            start += test_periods
            fold += 1
            continue

        test_pnls = np.array(test_pnls)
        sharpe = np.mean(test_pnls) / max(np.std(test_pnls), 1e-10) * np.sqrt(365)
        total = np.sum(test_pnls) * 100

        from datetime import datetime, timezone
        if len(test_ts) > 0:
            try:
                p_s = datetime.fromtimestamp(int(test_ts.iloc[0]) / 1000, tz=timezone.utc).strftime("%Y-%m")
                p_e = datetime.fromtimestamp(int(test_ts.iloc[-1]) / 1000, tz=timezone.utc).strftime("%Y-%m")
            except Exception:
                p_s = p_e = "?"
        else:
            p_s = p_e = "?"

        results.append({
            "fold": fold, "period": f"{p_s}->{p_e}",
            "sharpe": sharpe, "return_pct": total,
            "win_rate": float(np.mean(test_pnls > 0)),
            "n_trades": len(test_pnls),
        })

        fold += 1
        start += test_periods

    return results


def main():
    t0 = time.time()
    print("=" * 100)
    print("  Altcoin Rotation Strategy — Cross-Sectional Momentum")
    print("=" * 100)

    universe = load_universe()
    print(f"\n  Universe: {len(universe)} symbols")
    for sym in sorted(universe.keys()):
        print(f"    {sym}: {len(universe[sym]):,} bars")

    # Test multiple frequencies
    configs = [
        ("Daily", 24, 1, 3, 3, 90, 90),
        ("Daily", 24, 3, 3, 3, 90, 90),
        ("Daily", 24, 7, 3, 3, 90, 90),
        ("3-Day", 72, 1, 3, 3, 30, 30),
        ("Weekly", 168, 1, 3, 3, 13, 13),
        ("Weekly", 168, 2, 3, 3, 13, 13),
        ("Daily L5S5", 24, 3, 5, 5, 90, 90),
        ("Daily L1S1", 24, 3, 1, 1, 90, 90),
    ]

    print(f"\n{'=' * 100}")
    print("  Full-Sample Performance")
    print(f"{'=' * 100}")
    print(f"  {'Config':<20s} {'Sharpe':>8s} {'Return':>10s} {'WR':>6s} {'MaxDD':>7s} {'Periods':>8s} {'Active':>7s}")
    print(f"  {'-' * 70}")

    for name, freq, lb, nl, ns, train_p, test_p in configs:
        label = f"{name} lb={lb} L{nl}S{ns}"
        ret_df, timestamps = align_returns(universe, freq_hours=freq)
        result = run_rotation(ret_df, timestamps, lookback=lb, n_long=nl, n_short=ns)
        print(f"  {label:<20s} {result['sharpe']:>+7.2f} {result['total_return']:>+9.1f}% "
              f"{result['win_rate']:>5.1%} {result['max_dd']:>6.1%} "
              f"{result['n_periods']:>8d} {result['avg_active']:>6.0f}")

    # Walk-forward for best config
    print(f"\n{'=' * 100}")
    print("  Walk-Forward Validation")
    print(f"{'=' * 100}")

    for name, freq, lb, nl, ns, train_p, test_p in configs:
        label = f"{name} lb={lb} L{nl}S{ns}"
        ret_df, timestamps = align_returns(universe, freq_hours=freq)
        wf_results = walk_forward(ret_df, timestamps, name, lookback=lb,
                                   n_long=nl, n_short=ns,
                                   train_periods=train_p, test_periods=test_p)

        if not wf_results:
            continue

        sharpes = [r["sharpe"] for r in wf_results]
        pos = sum(1 for s in sharpes if s > 0)
        avg_sh = np.mean(sharpes)
        total = sum(r["return_pct"] for r in wf_results)

        need = max(len(wf_results) * 2 // 3, 1)
        verdict = "PASS" if pos >= need else "FAIL"

        print(f"\n  {label}: avg Sharpe={avg_sh:+.2f} total={total:+.1f}% "
              f"pos={pos}/{len(wf_results)} -> {verdict}")

        for r in wf_results:
            marker = " <+" if r["sharpe"] > 0 else ""
            print(f"    Fold {r['fold']:>2d} {r['period']:>14s}: "
                  f"Sharpe={r['sharpe']:>+6.2f} Ret={r['return_pct']:>+7.1f}% "
                  f"WR={r['win_rate']:.0%} n={r['n_trades']}{marker}")

    print(f"\n  Runtime: {time.time() - t0:.1f}s")
    print(f"{'=' * 100}")


if __name__ == "__main__":
    main()
