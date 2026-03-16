#!/usr/bin/env python3
"""Funding Rate Mean-Reversion Alpha — Full Walk-Forward Backtest.

Tests: extreme funding z-score triggers entry, receives funding payments,
exits on z-score normalization or time/stop.

Includes: realistic costs (fee, slippage), funding payment tracking,
per-settlement P&L, walk-forward validation.

Usage:
    python3 -m scripts.research.backtest_funding_alpha --symbol BTCUSDT
    python3 -m scripts.research.backtest_funding_alpha --symbol ETHUSDT --mode momentum
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ── Configuration ──────────────────────────────────────────────

@dataclass
class FundingAlphaConfig:
    # Signal thresholds
    z_entry: float = 2.0        # |z-score| above this → entry
    z_exit: float = 0.5         # |z-score| below this → exit
    # Risk
    stop_loss_pct: float = 0.02  # 2% stop
    max_hold_bars: int = 24      # max 24h hold (= 3 funding settlements)
    position_frac: float = 0.30  # 30% of equity per trade
    leverage: float = 3.0
    # Costs
    fee_bps: float = 4.0        # taker fee
    slippage_bps: float = 1.0
    # Filters
    vol_filter_percentile: float = 80  # skip high-vol periods
    min_persist: int = 2         # min funding_sign_persist to enter
    # Mode
    mode: str = "reversal"       # "reversal" or "momentum"


@dataclass
class Trade:
    entry_bar: int
    entry_price: float
    side: int  # 1=long, -1=short
    qty_frac: float
    z_at_entry: float
    funding_collected: float = 0.0
    exit_bar: int = 0
    exit_price: float = 0.0
    exit_reason: str = ""
    pnl_pct: float = 0.0
    gross_pnl_pct: float = 0.0
    fee_paid_pct: float = 0.0


# ── Data loading ───────────────────────────────────────────────

def load_data(symbol: str) -> tuple[pd.DataFrame, Dict[int, float]]:
    """Load 1h klines + funding schedule."""
    df = pd.read_csv(f"data_files/{symbol}_1h.csv")

    funding_path = Path(f"data_files/{symbol}_funding.csv")
    schedule: Dict[int, float] = {}
    if funding_path.exists():
        with open(funding_path, newline="") as f:
            for row in csv.DictReader(f):
                schedule[int(row["timestamp"])] = float(row["funding_rate"])

    return df, schedule


def compute_funding_features(df: pd.DataFrame, schedule: Dict[int, float]) -> pd.DataFrame:
    """Compute rolling funding z-score and related features."""
    n = len(df)
    ts_col = df["open_time"].values

    # Map funding to bars
    funding_times = sorted(schedule.keys())
    fidx = 0
    funding_at_bar = np.full(n, np.nan)
    for i in range(n):
        ts = int(ts_col[i])
        while fidx < len(funding_times) and funding_times[fidx] <= ts:
            fidx += 1
        if fidx > 0:
            funding_at_bar[i] = schedule[funding_times[fidx - 1]]

    # Rolling z-score (24-bar = ~3 funding settlements)
    window = 24
    zscore = np.full(n, np.nan)
    for i in range(window, n):
        w = funding_at_bar[i - window:i]
        valid = w[~np.isnan(w)]
        if len(valid) >= 10:
            mu, std = np.mean(valid), np.std(valid)
            if std > 1e-10 and not np.isnan(funding_at_bar[i]):
                zscore[i] = (funding_at_bar[i] - mu) / std

    # Sign persistence
    persist = np.zeros(n)
    for i in range(1, n):
        if not np.isnan(funding_at_bar[i]) and not np.isnan(funding_at_bar[i - 1]):
            if np.sign(funding_at_bar[i]) == np.sign(funding_at_bar[i - 1]) and funding_at_bar[i] != 0:
                persist[i] = persist[i - 1] + 1

    # Volatility (for filter)
    closes = df["close"].values.astype(float)
    ret = np.diff(np.log(closes), prepend=np.log(closes[0]))
    vol20 = np.full(n, np.nan)
    for i in range(20, n):
        vol20[i] = np.std(ret[i - 20:i])

    # Is this bar a funding settlement? (every 8h = bars 0,8,16 of each day)
    is_settlement = np.zeros(n, dtype=bool)
    for i in range(n):
        ts = int(ts_col[i])
        dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
        if dt.hour in (0, 8, 16):
            is_settlement[i] = True

    result = df[["open_time", "open", "high", "low", "close", "volume"]].copy()
    result["funding_rate"] = funding_at_bar
    result["funding_zscore"] = zscore
    result["funding_persist"] = persist
    result["vol20"] = vol20
    result["is_settlement"] = is_settlement
    return result


# ── Backtest engine ────────────────────────────────────────────

def run_backtest(
    data: pd.DataFrame, cfg: FundingAlphaConfig,
    start_bar: int = 0, end_bar: int = -1,
) -> tuple[List[Trade], np.ndarray]:
    """Run funding alpha backtest on a slice of data."""
    if end_bar == -1:
        end_bar = len(data)

    closes = data["close"].values.astype(float)
    highs = data["high"].values.astype(float)
    lows = data["low"].values.astype(float)
    zscore = data["funding_zscore"].values.astype(float)
    persist = data["funding_persist"].values.astype(float)
    vol20 = data["vol20"].values.astype(float)
    funding = data["funding_rate"].values.astype(float)
    is_settle = data["is_settlement"].values

    # Vol filter threshold
    valid_vol = vol20[~np.isnan(vol20)]
    vol_thresh = np.percentile(valid_vol, cfg.vol_filter_percentile) if len(valid_vol) > 100 else 999

    trades: List[Trade] = []
    equity_curve = np.ones(end_bar - start_bar)
    equity = 1.0
    position: Optional[Trade] = None
    cost_mult = (cfg.fee_bps + cfg.slippage_bps) / 10000

    for i in range(start_bar, end_bar):
        idx = i - start_bar
        z = zscore[i]
        c = closes[i]

        # Funding payment on settlement bars
        if position is not None and is_settle[i] and not np.isnan(funding[i]):
            # If short: receive positive funding, pay negative
            # If long: pay positive funding, receive negative
            fr = funding[i]
            payment = -position.side * fr * cfg.position_frac * cfg.leverage
            position.funding_collected += payment
            equity += payment

        # Exit checks
        if position is not None:
            bars_held = i - position.entry_bar
            side = position.side
            entry_p = position.entry_price

            # Stop loss (intra-bar)
            if side > 0 and lows[i] <= entry_p * (1 - cfg.stop_loss_pct):
                exit_p = entry_p * (1 - cfg.stop_loss_pct)
                _close_trade(position, i, exit_p, "stop", cfg, equity_curve, idx, cost_mult)
                equity += position.pnl_pct
                trades.append(position)
                position = None
            elif side < 0 and highs[i] >= entry_p * (1 + cfg.stop_loss_pct):
                exit_p = entry_p * (1 + cfg.stop_loss_pct)
                _close_trade(position, i, exit_p, "stop", cfg, equity_curve, idx, cost_mult)
                equity += position.pnl_pct
                trades.append(position)
                position = None
            # Z-score exit
            elif not np.isnan(z) and abs(z) < cfg.z_exit:
                _close_trade(position, i, c, "z_exit", cfg, equity_curve, idx, cost_mult)
                equity += position.pnl_pct
                trades.append(position)
                position = None
            # Max hold
            elif bars_held >= cfg.max_hold_bars:
                _close_trade(position, i, c, "max_hold", cfg, equity_curve, idx, cost_mult)
                equity += position.pnl_pct
                trades.append(position)
                position = None

        # Entry check
        if position is None and not np.isnan(z) and not np.isnan(vol20[i]):
            if vol20[i] <= vol_thresh and persist[i] >= cfg.min_persist:
                if cfg.mode == "reversal":
                    if z > cfg.z_entry:
                        # Crowded long → short
                        position = Trade(entry_bar=i, entry_price=c * (1 + cost_mult),
                                         side=-1, qty_frac=cfg.position_frac, z_at_entry=z)
                    elif z < -cfg.z_entry:
                        # Crowded short → long
                        position = Trade(entry_bar=i, entry_price=c * (1 + cost_mult),
                                         side=1, qty_frac=cfg.position_frac, z_at_entry=z)
                elif cfg.mode == "momentum":
                    if z > cfg.z_entry:
                        # High funding momentum → long
                        position = Trade(entry_bar=i, entry_price=c * (1 + cost_mult),
                                         side=1, qty_frac=cfg.position_frac, z_at_entry=z)
                    elif z < -cfg.z_entry:
                        # Low funding momentum → short
                        position = Trade(entry_bar=i, entry_price=c * (1 + cost_mult),
                                         side=-1, qty_frac=cfg.position_frac, z_at_entry=z)

        equity_curve[idx] = equity

    # Close open position at end
    if position is not None:
        _close_trade(position, end_bar - 1, closes[end_bar - 1], "end", cfg, equity_curve,
                     end_bar - 1 - start_bar, cost_mult)
        equity += position.pnl_pct
        trades.append(position)

    return trades, equity_curve


def _close_trade(trade: Trade, bar: int, exit_price: float, reason: str,
                 cfg: FundingAlphaConfig, eq_curve: np.ndarray, idx: int,
                 cost_mult: float) -> None:
    trade.exit_bar = bar
    trade.exit_price = exit_price
    trade.exit_reason = reason
    raw_return = trade.side * (exit_price / trade.entry_price - 1.0)
    trade.gross_pnl_pct = raw_return * cfg.position_frac * cfg.leverage
    trade.fee_paid_pct = cost_mult * 2 * cfg.position_frac * cfg.leverage  # entry + exit
    trade.pnl_pct = trade.gross_pnl_pct - trade.fee_paid_pct + trade.funding_collected


# ── Walk-forward validation ────────────────────────────────────

def walk_forward(data: pd.DataFrame, cfg: FundingAlphaConfig,
                 min_train: int = 8760, test_window: int = 2190) -> None:
    """Walk-forward backtest with expanding window."""
    n = len(data)
    folds = []
    test_start = min_train
    fold_idx = 0

    while test_start + test_window <= n:
        folds.append((0, test_start, test_start, min(test_start + test_window, n)))
        test_start += test_window
        fold_idx += 1

    if not folds:
        print("  Not enough data for walk-forward")
        return

    print(f"  Folds: {len(folds)}, Mode: {cfg.mode}, z_entry={cfg.z_entry}, "
          f"stop={cfg.stop_loss_pct*100:.1f}%, max_hold={cfg.max_hold_bars}h")
    print(f"\n  {'Fold':<6} {'Period':<22} {'Trades':>6} {'WinR':>6} {'PnL':>8} "
          f"{'Sharpe':>7} {'Funding':>8} {'AvgHold':>7}")
    print(f"  {'-'*75}")

    all_sharpes = []
    all_returns = []
    total_trades = 0

    ts_col = data["open_time"].values
    for fi, (tr_s, tr_e, te_s, te_e) in enumerate(folds):
        trades, eq = run_backtest(data, cfg, start_bar=te_s, end_bar=te_e)
        total_trades += len(trades)

        if not trades:
            ts0 = datetime.fromtimestamp(int(ts_col[te_s]) / 1000, tz=timezone.utc)
            ts1 = datetime.fromtimestamp(int(ts_col[min(te_e - 1, n - 1)]) / 1000, tz=timezone.utc)
            period = f"{ts0:%Y-%m}→{ts1:%Y-%m}"
            print(f"  {fi:<6} {period:<22} {'0':>6} {'---':>6} {'---':>8} {'---':>7} {'---':>8} {'---':>7}")
            all_sharpes.append(0)
            all_returns.append(0)
            continue

        pnls = [t.pnl_pct for t in trades]
        win_rate = sum(1 for p in pnls if p > 0) / len(pnls)
        total_pnl = sum(pnls)
        funding_total = sum(t.funding_collected for t in trades)
        avg_hold = np.mean([t.exit_bar - t.entry_bar for t in trades])
        sharpe = (np.mean(pnls) / np.std(pnls) * np.sqrt(len(pnls))) if np.std(pnls) > 0 and len(pnls) > 1 else 0

        ts0 = datetime.fromtimestamp(int(ts_col[te_s]) / 1000, tz=timezone.utc)
        ts1 = datetime.fromtimestamp(int(ts_col[min(te_e - 1, n - 1)]) / 1000, tz=timezone.utc)
        period = f"{ts0:%Y-%m}→{ts1:%Y-%m}"

        print(f"  {fi:<6} {period:<22} {len(trades):>6} {win_rate*100:>5.1f}% "
              f"{total_pnl*100:>+7.2f}% {sharpe:>7.2f} {funding_total*100:>+7.3f}% {avg_hold:>6.1f}h")

        all_sharpes.append(sharpe)
        all_returns.append(total_pnl)

    n_folds = len(folds)
    pos_sharpe = sum(1 for s in all_sharpes if s > 0)
    threshold = int(np.ceil(n_folds * 2 / 3))
    passed = pos_sharpe >= threshold
    total_ret = sum(all_returns)
    avg_sharpe = np.mean(all_sharpes) if all_sharpes else 0

    print(f"  {'-'*75}")
    print(f"\n  VERDICT: {pos_sharpe}/{n_folds} positive Sharpe "
          f"(need >= {threshold}) → {'PASS' if passed else 'FAIL'}")
    print(f"  Average Sharpe: {avg_sharpe:.2f}")
    print(f"  Total return: {total_ret*100:+.2f}%")
    print(f"  Total trades: {total_trades} ({total_trades/n_folds:.0f}/fold)")


# ── Main ───────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Funding Alpha Backtest")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--mode", choices=["reversal", "momentum"], default="reversal")
    parser.add_argument("--z-entry", type=float, default=2.0)
    parser.add_argument("--z-exit", type=float, default=0.5)
    parser.add_argument("--stop", type=float, default=0.02)
    parser.add_argument("--max-hold", type=int, default=24)
    parser.add_argument("--leverage", type=float, default=3.0)
    parser.add_argument("--vol-filter", type=float, default=80)
    parser.add_argument("--min-persist", type=int, default=2)
    args = parser.parse_args()

    symbol = args.symbol.upper()
    cfg = FundingAlphaConfig(
        z_entry=args.z_entry, z_exit=args.z_exit,
        stop_loss_pct=args.stop, max_hold_bars=args.max_hold,
        leverage=args.leverage, vol_filter_percentile=args.vol_filter,
        min_persist=args.min_persist, mode=args.mode,
    )

    print(f"\n{'='*75}")
    print(f"  Funding Alpha Backtest: {symbol} ({cfg.mode})")
    print(f"{'='*75}")

    df, schedule = load_data(symbol)
    print(f"  Bars: {len(df)}, Funding settlements: {len(schedule)}")

    data = compute_funding_features(df, schedule)
    print(f"  Features computed. Non-NaN z-scores: "
          f"{(~np.isnan(data['funding_zscore'].values)).sum()}")

    walk_forward(data, cfg)


if __name__ == "__main__":
    main()
