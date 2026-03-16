#!/usr/bin/env python3
"""Polymarket BTC Up/Down 5m -- Market Making Backtest.

Simulates market making on 5-minute binary options using historical data.
Tests multiple strategies:
  1. Pure Overround: sell both Up+Down asks, earn overround
  2. RSI-informed MM: bias direction based on RSI(5) signal
  3. RSI taker baseline (for comparison)

Key assumptions (from live CLOB data analysis):
  - Average spread: 1.34% (1 cent on $0.50 contract)
  - Overround: 1.36% (sell both sides for $1.0136, pay $1.00)
  - Maker fee: 0% (Polymarket maker incentive)
  - Taker fee: ~1% (charged to counterparty)
  - Average depth: bid $1,474 / ask $2,217

Fill model: independent of outcome, based on window index hash.

Usage:
    python3 -m scripts.research.polymarket_mm_backtest
"""
from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# --- CLOB parameters (from live data analysis) ------------------------------
DEFAULT_SPREAD = 0.01       # 1 cent typical spread
MAKER_FEE = 0.0             # Polymarket maker: 0%
TAKER_FEE = 0.01            # ~1% taker fee (our counterparty pays this)


# --- RSI computation --------------------------------------------------------

def compute_rsi_series(prices: np.ndarray, period: int = 5) -> np.ndarray:
    """Wilder's RSI over price series."""
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    n = len(deltas)
    if period > n:
        return np.full(len(prices), 50.0)
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    avg_gain[period - 1] = np.mean(gains[:period])
    avg_loss[period - 1] = np.mean(losses[:period])
    for i in range(period, n):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gains[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + losses[i]) / period
    rs = avg_gain / np.where(avg_loss > 1e-10, avg_loss, 1e-10)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    return np.concatenate([[50.0] * period, rsi[period - 1:]])


# --- Deterministic fill randomness (independent of outcome) -----------------

def fill_random(window_idx: int, salt: int = 0) -> float:
    """Deterministic pseudo-random [0,1) based on window index.

    Uses hash of index, NOT btc_close or result, so fill decision
    is independent of outcome.
    """
    h = hashlib.md5(f"{window_idx}:{salt}".encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


# --- Market Making Engine ----------------------------------------------------

@dataclass
class MMTrade:
    window_idx: int
    side: str           # "up", "down", "both"
    action: str         # "buy" or "sell"
    price: float
    size: float
    result: str         # actual outcome "Up" or "Down"
    pnl: float
    strategy: str


@dataclass
class MMResult:
    label: str
    trades: List[MMTrade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)

    @property
    def n_trades(self) -> int:
        return len(self.trades)

    @property
    def total_pnl(self) -> float:
        return sum(t.pnl for t in self.trades)


def _sell_pnl(ask_price: float, size: float, actual_up: bool, sold_up: bool) -> float:
    """PnL from selling a binary contract.

    Sell Up at ask: collect ask*size. If Up wins, pay 1*size. If Down, keep all.
    Sell Down at ask: collect ask*size. If Down wins, pay 1*size. If Up, keep all.
    """
    if sold_up:
        if actual_up:
            return (ask_price - 1.0) * size  # lose: owe $1, collected ask
        else:
            return ask_price * size  # win: keep premium
    else:
        if not actual_up:
            return (ask_price - 1.0) * size  # lose
        else:
            return ask_price * size  # win


def _buy_pnl(bid_price: float, size: float, actual_up: bool, bought_up: bool) -> float:
    """PnL from buying a binary contract.

    Buy Up at bid: pay bid*size. If Up wins, receive 1*size. If Down, lose bid.
    """
    if bought_up:
        if actual_up:
            return (1.0 - bid_price) * size  # win
        else:
            return -bid_price * size  # lose
    else:
        if not actual_up:
            return (1.0 - bid_price) * size  # win
        else:
            return -bid_price * size  # lose


# --- Strategy 1: Pure Overround ----------------------------------------------

def run_overround(
    df: pd.DataFrame,
    size: float = 10.0,
    initial_equity: float = 100.0,
    fill_rate_both: float = 0.08,
    fill_rate_single: float = 0.15,
) -> MMResult:
    """Sell both Up ask + Down ask every window.

    Both fill: earn overround (risk-free).
    One side fills: 50/50 directional exposure.
    """
    results_col = df["result"].values
    n = len(df)
    result = MMResult(label="Overround")
    equity = initial_equity
    result.equity_curve.append(equity)

    up_ask = 0.51  # slightly above 0.50 mid
    dn_ask = 0.51

    for i in range(n):
        if equity < size * 0.5:
            result.equity_curve.append(equity)
            continue

        actual_up = results_col[i] == "Up"
        bet_size = min(size, equity * 0.10)  # max 10% equity per side

        r1 = fill_random(i, salt=1)
        r2 = fill_random(i, salt=2)

        both_filled = r1 < fill_rate_both
        up_only = (not both_filled) and r1 < fill_rate_single
        dn_only = (not both_filled) and (not up_only) and r2 < fill_rate_single

        if both_filled:
            collected = (up_ask + dn_ask) * bet_size
            payout = 1.0 * bet_size
            pnl = collected - payout
            result.trades.append(MMTrade(i, "both", "sell", up_ask + dn_ask,
                                        bet_size, results_col[i], pnl, "both_fill"))
            equity += pnl
        elif up_only:
            pnl = _sell_pnl(up_ask, bet_size, actual_up, sold_up=True)
            result.trades.append(MMTrade(i, "up", "sell", up_ask, bet_size,
                                        results_col[i], pnl, "up_only"))
            equity += pnl
        elif dn_only:
            pnl = _sell_pnl(dn_ask, bet_size, actual_up, sold_up=False)
            result.trades.append(MMTrade(i, "down", "sell", dn_ask, bet_size,
                                        results_col[i], pnl, "dn_only"))
            equity += pnl

        equity = max(equity, 0.0)
        result.equity_curve.append(equity)

    return result


# --- Strategy 2: RSI-Informed MM ---------------------------------------------

def run_rsi_mm(
    df: pd.DataFrame,
    size: float = 10.0,
    initial_equity: float = 100.0,
    fill_rate_both: float = 0.08,
    fill_rate_single: float = 0.15,
    rsi_os: float = 25.0,
    rsi_ob: float = 75.0,
) -> MMResult:
    """RSI-informed market making.

    RSI neutral: overround (sell both)
    RSI oversold: lean Up (buy Up bid + sell Down ask)
    RSI overbought: lean Down (buy Down bid + sell Up ask)
    """
    closes = df["close"].values
    results_col = df["result"].values
    rsi = compute_rsi_series(closes, 5)
    n = len(df)

    result = MMResult(label="RSI-Informed MM")
    equity = initial_equity
    result.equity_curve.append(equity)

    for i in range(10, n):
        if equity < size * 0.5:
            result.equity_curve.append(equity)
            continue

        actual_up = results_col[i] == "Up"
        bet_size = min(size, equity * 0.10)
        cur_rsi = rsi[i]

        r1 = fill_random(i, salt=10)
        r2 = fill_random(i, salt=20)

        if cur_rsi < rsi_os:
            # Lean bullish: buy Up at bid, sell Down at ask
            up_bid = 0.49
            dn_ask = 0.51
            filled_buy = r1 < fill_rate_single
            filled_sell = r2 < fill_rate_single

            if filled_buy and filled_sell:
                # Bought Up + Sold Down = net long Up position
                # If Up: win on buy (+0.51) + win on sell (+0.51) = +1.02
                # If Down: lose on buy (-0.49) + lose on sell (-0.49) = -0.98
                pnl_buy = _buy_pnl(up_bid, bet_size, actual_up, bought_up=True)
                pnl_sell = _sell_pnl(dn_ask, bet_size, actual_up, sold_up=False)
                pnl = pnl_buy + pnl_sell
                result.trades.append(MMTrade(i, "both", "buy+sell", 0, bet_size,
                                            results_col[i], pnl, "rsi_bull_both"))
                equity += pnl
            elif filled_buy:
                pnl = _buy_pnl(up_bid, bet_size, actual_up, bought_up=True)
                result.trades.append(MMTrade(i, "up", "buy", up_bid, bet_size,
                                            results_col[i], pnl, "rsi_bull_buy"))
                equity += pnl
            elif filled_sell:
                pnl = _sell_pnl(dn_ask, bet_size, actual_up, sold_up=False)
                result.trades.append(MMTrade(i, "down", "sell", dn_ask, bet_size,
                                            results_col[i], pnl, "rsi_bull_sell"))
                equity += pnl

        elif cur_rsi > rsi_ob:
            # Lean bearish: buy Down at bid, sell Up at ask
            dn_bid = 0.49
            up_ask = 0.51
            filled_buy = r1 < fill_rate_single
            filled_sell = r2 < fill_rate_single

            if filled_buy and filled_sell:
                pnl_buy = _buy_pnl(dn_bid, bet_size, actual_up, bought_up=False)
                pnl_sell = _sell_pnl(up_ask, bet_size, actual_up, sold_up=True)
                pnl = pnl_buy + pnl_sell
                result.trades.append(MMTrade(i, "both", "buy+sell", 0, bet_size,
                                            results_col[i], pnl, "rsi_bear_both"))
                equity += pnl
            elif filled_buy:
                pnl = _buy_pnl(dn_bid, bet_size, actual_up, bought_up=False)
                result.trades.append(MMTrade(i, "down", "buy", dn_bid, bet_size,
                                            results_col[i], pnl, "rsi_bear_buy"))
                equity += pnl
            elif filled_sell:
                pnl = _sell_pnl(up_ask, bet_size, actual_up, sold_up=True)
                result.trades.append(MMTrade(i, "up", "sell", up_ask, bet_size,
                                            results_col[i], pnl, "rsi_bear_sell"))
                equity += pnl

        else:
            # Neutral: overround (sell both)
            up_ask = 0.51
            dn_ask = 0.51
            both_filled = r1 < fill_rate_both
            up_only = (not both_filled) and r1 < fill_rate_single
            dn_only = (not both_filled) and (not up_only) and r2 < fill_rate_single

            if both_filled:
                collected = (up_ask + dn_ask) * bet_size
                payout = 1.0 * bet_size
                pnl = collected - payout
                result.trades.append(MMTrade(i, "both", "sell", up_ask + dn_ask,
                                            bet_size, results_col[i], pnl, "neutral_both"))
                equity += pnl
            elif up_only:
                pnl = _sell_pnl(up_ask, bet_size, actual_up, sold_up=True)
                result.trades.append(MMTrade(i, "up", "sell", up_ask, bet_size,
                                            results_col[i], pnl, "neutral_up"))
                equity += pnl
            elif dn_only:
                pnl = _sell_pnl(dn_ask, bet_size, actual_up, sold_up=False)
                result.trades.append(MMTrade(i, "down", "sell", dn_ask, bet_size,
                                            results_col[i], pnl, "neutral_dn"))
                equity += pnl

        equity = max(equity, 0.0)
        result.equity_curve.append(equity)

    return result


# --- Strategy 3: RSI Taker Baseline ------------------------------------------

def run_rsi_taker(
    df: pd.DataFrame,
    size: float = 10.0,
    initial_equity: float = 100.0,
    rsi_os: float = 25.0,
    rsi_ob: float = 75.0,
) -> MMResult:
    """Pure RSI taker: buy at ask when RSI extreme. Always fills (taker)."""
    closes = df["close"].values
    results_col = df["result"].values
    rsi = compute_rsi_series(closes, 5)
    n = len(df)

    result = MMResult(label="RSI Taker")
    equity = initial_equity
    result.equity_curve.append(equity)

    for i in range(10, n):
        if equity < 1.0:
            result.equity_curve.append(equity)
            continue

        actual_up = results_col[i] == "Up"
        bet_size = min(size, equity * 0.10)

        if cur_rsi := rsi[i]:
            pass

        if cur_rsi < rsi_os:
            # Buy Up at ask (taker)
            buy_price = 0.505  # mid + half spread
            pnl = _buy_pnl(buy_price, bet_size, actual_up, bought_up=True)
            result.trades.append(MMTrade(i, "up", "buy", buy_price, bet_size,
                                        results_col[i], pnl, "taker_up"))
            equity += pnl
        elif cur_rsi > rsi_ob:
            buy_price = 0.505
            pnl = _buy_pnl(buy_price, bet_size, actual_up, bought_up=False)
            result.trades.append(MMTrade(i, "down", "buy", buy_price, bet_size,
                                        results_col[i], pnl, "taker_dn"))
            equity += pnl

        equity = max(equity, 0.0)
        result.equity_curve.append(equity)

    return result


# --- Analysis ----------------------------------------------------------------

def analyze(r: MMResult) -> Dict:
    """Compute metrics."""
    if not r.trades:
        return {"label": r.label, "trades": 0}

    pnls = np.array([t.pnl for t in r.trades])
    eq = np.array(r.equity_curve)
    peak = np.maximum.accumulate(eq)
    dd = (peak - eq) / np.where(peak > 0, peak, 1)
    max_dd = float(np.max(dd))
    days = len(eq) / (12 * 24)

    # Sharpe from daily PnL
    bars_per_day = 12 * 24
    n_days = int(days)
    daily_pnls = []
    for d in range(n_days):
        s = d * bars_per_day
        e = min((d + 1) * bars_per_day, len(eq) - 1)
        daily_pnls.append(eq[e] - eq[s])
    daily_pnls = np.array(daily_pnls)
    sharpe = 0.0
    if len(daily_pnls) > 10 and np.std(daily_pnls) > 0:
        sharpe = float(np.mean(daily_pnls) / np.std(daily_pnls) * np.sqrt(365))

    # Strategy breakdown
    strat_counts: Dict[str, int] = {}
    strat_pnl: Dict[str, float] = {}
    for t in r.trades:
        strat_counts[t.strategy] = strat_counts.get(t.strategy, 0) + 1
        strat_pnl[t.strategy] = strat_pnl.get(t.strategy, 0) + t.pnl

    return {
        "label": r.label,
        "trades": len(pnls),
        "trades_per_day": len(pnls) / days if days > 0 else 0,
        "total_pnl": float(np.sum(pnls)),
        "win_rate": float(np.mean(pnls > 0)),
        "avg_pnl": float(np.mean(pnls)),
        "max_dd": max_dd,
        "final_equity": eq[-1],
        "sharpe": sharpe,
        "strat_counts": strat_counts,
        "strat_pnl": strat_pnl,
        "days": days,
    }


# --- Main --------------------------------------------------------------------

def main():
    t0 = time.time()

    print("=" * 100)
    print("  Polymarket BTC Up/Down 5m -- Market Making Backtest (v2)")
    print("=" * 100)

    df = pd.read_csv("data/polymarket/btc_5m_updown_history.csv")
    print(f"\n  Data: {len(df):,} windows, "
          f"{df['datetime'].iloc[0][:10]} -> {df['datetime'].iloc[-1][:10]}")

    # --- Run all strategies with multiple fill rate scenarios ---
    print("\n  Running strategies across fill rate scenarios...\n")

    print(f"  {'Strategy':<28s} {'Fill%':>6s} {'Trades':>7s} {'T/Day':>6s} "
          f"{'WinR':>6s} {'PnL':>10s} {'Final$':>8s} {'MaxDD':>7s} {'Sharpe':>7s}")
    print(f"  {'~' * 92}")

    # Test fill rates: 5%, 10%, 15%, 20%
    for fill_rate in [0.05, 0.10, 0.15, 0.20]:
        both_rate = fill_rate * 0.5  # both sides filling is rarer

        r_over = run_overround(df, size=10, initial_equity=100,
                               fill_rate_both=both_rate, fill_rate_single=fill_rate)
        s = analyze(r_over)
        if s["trades"] > 0:
            print(f"  {'Overround':<28s} {fill_rate:>5.0%} {s['trades']:>7d} "
                  f"{s['trades_per_day']:>5.1f} {s['win_rate']:>5.1%} "
                  f"${s['total_pnl']:>+9.1f} ${s['final_equity']:>7.1f} "
                  f"{s['max_dd']:>6.1%} {s['sharpe']:>+6.2f}")

    print()
    for fill_rate in [0.05, 0.10, 0.15, 0.20]:
        both_rate = fill_rate * 0.5

        r_rsi = run_rsi_mm(df, size=10, initial_equity=100,
                           fill_rate_both=both_rate, fill_rate_single=fill_rate)
        s = analyze(r_rsi)
        if s["trades"] > 0:
            print(f"  {'RSI-MM':<28s} {fill_rate:>5.0%} {s['trades']:>7d} "
                  f"{s['trades_per_day']:>5.1f} {s['win_rate']:>5.1%} "
                  f"${s['total_pnl']:>+9.1f} ${s['final_equity']:>7.1f} "
                  f"{s['max_dd']:>6.1%} {s['sharpe']:>+6.2f}")

    print()
    # RSI taker is always 100% fill (taker)
    r_taker = run_rsi_taker(df, size=10, initial_equity=100)
    s_taker = analyze(r_taker)
    if s_taker["trades"] > 0:
        print(f"  {'RSI Taker (baseline)':<28s} {'100%':>6s} {s_taker['trades']:>7d} "
              f"{s_taker['trades_per_day']:>5.1f} {s_taker['win_rate']:>5.1%} "
              f"${s_taker['total_pnl']:>+9.1f} ${s_taker['final_equity']:>7.1f} "
              f"{s_taker['max_dd']:>6.1%} {s_taker['sharpe']:>+6.2f}")

    # --- Detailed breakdown for 10% fill rate scenario ---
    print(f"\n{'=' * 100}")
    print(f"  Detailed Breakdown (10% fill rate)")
    print(f"{'=' * 100}")

    for strat_name, runner in [
        ("Overround", lambda: run_overround(df, size=10, initial_equity=100,
                                            fill_rate_both=0.05, fill_rate_single=0.10)),
        ("RSI-MM", lambda: run_rsi_mm(df, size=10, initial_equity=100,
                                       fill_rate_both=0.05, fill_rate_single=0.10)),
    ]:
        r = runner()
        s = analyze(r)
        print(f"\n  {strat_name} (trades={s['trades']}, PnL=${s['total_pnl']:+.1f}):")
        for name, count in sorted(s["strat_counts"].items()):
            pnl = s["strat_pnl"].get(name, 0)
            avg = pnl / count if count > 0 else 0
            wr = sum(1 for t in r.trades if t.strategy == name and t.pnl > 0) / max(count, 1)
            print(f"    {name:<20s}: {count:>6d} trades  WR={wr:.1%}  "
                  f"PnL=${pnl:>+9.1f}  avg=${avg:>+.3f}")

    # --- Monthly P&L for best scenario ---
    print(f"\n{'=' * 100}")
    print(f"  Monthly P&L (Overround, 10% fill, $100 start)")
    print(f"{'=' * 100}")

    r_best = run_overround(df, size=10, initial_equity=100,
                           fill_rate_both=0.05, fill_rate_single=0.10)
    eq = np.array(r_best.equity_curve)
    bars_per_month = 12 * 24 * 30
    n_months = min(len(eq) // bars_per_month, 28)

    print(f"\n  {'Month':>6s} {'Start$':>8s} {'End$':>8s} {'PnL':>9s} {'Trades':>7s}")
    print(f"  {'~' * 44}")

    pos_months = 0
    for m in range(n_months):
        s_idx = m * bars_per_month
        e_idx = min((m + 1) * bars_per_month, len(eq) - 1)
        eq_s = eq[s_idx]
        eq_e = eq[e_idx]
        pnl_m = eq_e - eq_s
        trades_m = sum(1 for t in r_best.trades if s_idx <= t.window_idx < e_idx)
        if pnl_m > 0:
            pos_months += 1
        print(f"  {m+1:>6d} ${eq_s:>7.1f} ${eq_e:>7.1f} ${pnl_m:>+8.1f} {trades_m:>7d}")

    print(f"\n  Positive months: {pos_months}/{n_months}")
    total_ret = (eq[-1] - 100) / 100 * 100
    print(f"  Total: ${eq[-1]:.1f} ({total_ret:+.1f}%)")

    # --- RSI-MM monthly ---
    print(f"\n{'=' * 100}")
    print(f"  Monthly P&L (RSI-MM, 10% fill, $100 start)")
    print(f"{'=' * 100}")

    r_rsimm = run_rsi_mm(df, size=10, initial_equity=100,
                          fill_rate_both=0.05, fill_rate_single=0.10)
    eq2 = np.array(r_rsimm.equity_curve)
    n_months2 = min(len(eq2) // bars_per_month, 28)

    print(f"\n  {'Month':>6s} {'Start$':>8s} {'End$':>8s} {'PnL':>9s} {'Trades':>7s}")
    print(f"  {'~' * 44}")

    pos_m2 = 0
    for m in range(n_months2):
        s_idx = m * bars_per_month
        e_idx = min((m + 1) * bars_per_month, len(eq2) - 1)
        eq_s = eq2[s_idx]
        eq_e = eq2[e_idx]
        pnl_m = eq_e - eq_s
        trades_m = sum(1 for t in r_rsimm.trades if s_idx <= t.window_idx < e_idx)
        if pnl_m > 0:
            pos_m2 += 1
        print(f"  {m+1:>6d} ${eq_s:>7.1f} ${eq_e:>7.1f} ${pnl_m:>+8.1f} {trades_m:>7d}")

    print(f"\n  Positive months: {pos_m2}/{n_months2}")
    total_ret2 = (eq2[-1] - 100) / 100 * 100
    print(f"  Total: ${eq2[-1]:.1f} ({total_ret2:+.1f}%)")

    print(f"\n  Runtime: {time.time() - t0:.1f}s")
    print(f"{'=' * 100}")


if __name__ == "__main__":
    main()
