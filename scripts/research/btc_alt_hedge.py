#!/usr/bin/env python3
"""BTC Long + ALT Short Hedge Strategy — walk-forward validation.

Strategy: 50% long BTC + 50% short equal-weight ALT basket
Filter: Only short ALTs when ALT/BTC ratio < MA (BTC outperforming)
Rebalance: Monthly

Backtest result: +709%, Sharpe 1.80, MaxDD -15% (6.1 years)

Usage:
    python3 -m scripts.research.btc_alt_hedge
    python3 -m scripts.research.btc_alt_hedge --alt-weight 0.3
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
from pathlib import Path


ALT_BASKET = [
    "ADAUSDT", "DOGEUSDT", "XRPUSDT", "LINKUSDT", "DOTUSDT",
    "AVAXUSDT", "NEOUSDT", "FILUSDT", "ZECUSDT", "BCHUSDT",
]

DEFAULT_ALT_WEIGHT = 0.5
DEFAULT_MA_WINDOW = 480  # 20 days in 1h bars
DEFAULT_REBAL_BARS = 720  # 30 days
DEFAULT_FUNDING_BPS = 1.0  # per 8h


def load_prices(symbols: list[str], btc_symbol: str = "BTCUSDT") -> tuple[np.ndarray, np.ndarray, np.ndarray,
    list[str]]:
    """Load and align prices for BTC + ALTs by timestamp."""
    data_dir = Path("data_files")

    btc_df = pd.read_csv(data_dir / f"{btc_symbol}_1h.csv")
    btc_ts = btc_df["open_time"].values
    btc_close = btc_df["close"].values.astype(float)

    all_data = {}
    for sym in symbols:
        p = data_dir / f"{sym}_1h.csv"
        if p.exists():
            df = pd.read_csv(p, usecols=["open_time", "close"])
            all_data[sym] = dict(zip(df["open_time"].values, df["close"].values.astype(float)))

    valid_syms = [s for s in symbols if s in all_data]
    n = len(btc_ts)
    alt_matrix = np.full((n, len(valid_syms)), np.nan)
    for j, sym in enumerate(valid_syms):
        for i, ts in enumerate(btc_ts):
            p = all_data[sym].get(ts)
            if p is not None:
                alt_matrix[i, j] = p

    return btc_ts, btc_close, alt_matrix, valid_syms


def run_hedge_backtest(
    btc_close: np.ndarray,
    alt_matrix: np.ndarray,
    *,
    alt_weight: float = DEFAULT_ALT_WEIGHT,
    ma_window: int = DEFAULT_MA_WINDOW,
    rebal_bars: int = DEFAULT_REBAL_BARS,
    funding_bps_per_8h: float = DEFAULT_FUNDING_BPS,
    fee_bps: float = 8.0,
    use_ma_filter: bool = True,
    min_alts: int = 3,
    start_bar: int = 0,
    end_bar: int = -1,
) -> dict:
    """Run BTC long + ALT short hedge backtest."""
    n = len(btc_close)
    if end_bar == -1:
        end_bar = n

    fee = fee_bps / 10000
    funding_per_bar = funding_bps_per_8h / 10000 / 8

    start = max(start_bar, ma_window + 1)
    equity = 1.0
    equity_curve = []
    funding_paid = 0.0
    active_bars = 0

    prev_btc = btc_close[start]
    prev_alts = alt_matrix[start].copy()

    for i in range(start, end_bar):
        if np.isnan(btc_close[i]):
            equity_curve.append(equity)
            continue

        valid = ~np.isnan(alt_matrix[i]) & ~np.isnan(prev_alts)
        n_valid = valid.sum()
        if n_valid < min_alts:
            equity_curve.append(equity)
            prev_btc = btc_close[i]
            prev_alts = alt_matrix[i].copy()
            continue

        # BTC.D filter
        is_short_active = True
        if use_ma_filter and i > ma_window:
            ratios = []
            for j in range(max(start, i - ma_window), i):
                if not np.isnan(btc_close[j]):
                    aa = np.nanmean(alt_matrix[j][valid])
                    if aa > 0:
                        ratios.append(aa / btc_close[j])
            if ratios:
                ratio_ma = np.mean(ratios)
                current_ratio = np.nanmean(alt_matrix[i][valid]) / btc_close[i]
                if current_ratio > ratio_ma:
                    is_short_active = False

        btc_ret = btc_close[i] / prev_btc - 1 if prev_btc > 0 else 0

        if is_short_active:
            alt_rets = []
            for j in range(alt_matrix.shape[1]):
                if valid[j] and prev_alts[j] > 0:
                    alt_rets.append(alt_matrix[i, j] / prev_alts[j] - 1)

            if alt_rets:
                avg_alt_ret = np.mean(alt_rets)
                port_ret = (1 - alt_weight) * btc_ret + alt_weight * (-avg_alt_ret)
                port_ret -= alt_weight * funding_per_bar
                funding_paid += alt_weight * funding_per_bar
                active_bars += 1
            else:
                port_ret = (1 - alt_weight) * btc_ret
        else:
            port_ret = (1 - alt_weight) * btc_ret

        if i % rebal_bars == 0:
            port_ret -= fee

        equity *= (1 + port_ret)
        equity_curve.append(equity)
        prev_btc = btc_close[i]
        prev_alts = alt_matrix[i].copy()

    eq = np.array(equity_curve) if equity_curve else np.array([1.0])
    total_ret = eq[-1] / eq[0] - 1
    years = (end_bar - start) / 8760

    daily = eq[::24]
    if len(daily) > 30:
        dr = np.diff(daily) / daily[:-1]
        sharpe = np.mean(dr) / np.std(dr) * np.sqrt(365) if np.std(dr) > 0 else 0
    else:
        sharpe = 0

    peak = np.maximum.accumulate(eq)
    max_dd = float(np.min((eq - peak) / peak)) * 100
    annual_ret = (eq[-1] / eq[0]) ** (1 / max(years, 0.1)) - 1

    return {
        "total_ret": total_ret * 100,
        "annual_ret": annual_ret * 100,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "funding_pct": funding_paid * 100,
        "years": years,
        "active_pct": active_bars / max(len(equity_curve), 1) * 100,
        "equity_curve": eq,
    }


def walk_forward(btc_close, alt_matrix, **kwargs):
    """Walk-forward validation with 3-month test windows."""
    n = len(btc_close)
    min_train = 8760
    test_bars = 2190

    folds = []
    ts = min_train
    while ts + test_bars <= n:
        folds.append((ts, min(ts + test_bars, n)))
        ts += test_bars

    fold_results = []
    for te_s, te_e in folds:
        r = run_hedge_backtest(btc_close, alt_matrix,
                               start_bar=te_s, end_bar=te_e, **kwargs)
        fold_results.append(r)

    sharpes = [f["sharpe"] for f in fold_results]
    rets = [f["total_ret"] for f in fold_results]
    pos_s = sum(1 for s in sharpes if s > 0)

    return {
        "n_folds": len(folds),
        "pos_folds": pos_s,
        "avg_sharpe": float(np.mean(sharpes)),
        "total_ret": sum(rets),
        "fold_sharpes": sharpes,
        "fold_rets": rets,
    }


def main():
    parser = argparse.ArgumentParser(description="BTC+ALT Hedge Strategy")
    parser.add_argument("--alt-weight", type=float, default=DEFAULT_ALT_WEIGHT)
    parser.add_argument("--no-filter", action="store_true")
    parser.add_argument("--walkforward", action="store_true")
    args = parser.parse_args()

    print("=" * 70)
    print("  BTC Long + ALT Short Hedge Strategy")
    print("=" * 70)

    btc_ts, btc_close, alt_matrix, valid_syms = load_prices(ALT_BASKET)
    print(f"  BTC bars: {len(btc_close):,}, ALTs: {len(valid_syms)}")

    if args.walkforward:
        print("\n  Walk-forward validation:")
        r = walk_forward(btc_close, alt_matrix,
                         alt_weight=args.alt_weight,
                         use_ma_filter=not args.no_filter)
        print(f"  {r['pos_folds']}/{r['n_folds']} positive Sharpe")
        print(f"  Avg Sharpe: {r['avg_sharpe']:.2f}")
        print(f"  Total return: {r['total_ret']:+.0f}%")
        for i, (s, ret) in enumerate(zip(r["fold_sharpes"], r["fold_rets"])):
            print(f"    Fold {i}: Sharpe={s:.2f} Ret={ret:+.1f}%")
    else:
        r = run_hedge_backtest(btc_close, alt_matrix,
                               alt_weight=args.alt_weight,
                               use_ma_filter=not args.no_filter)
        print(f"\n  Total return: {r['total_ret']:+.0f}%")
        print(f"  Annual return: {r['annual_ret']:+.1f}%")
        print(f"  Sharpe: {r['sharpe']:.2f}")
        print(f"  Max DD: {r['max_dd']:.0f}%")
        print(f"  Active %: {r['active_pct']:.0f}%")
        print(f"  Funding cost: {r['funding_pct']:.1f}%")


if __name__ == "__main__":
    main()
