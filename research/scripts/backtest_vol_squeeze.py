#!/usr/bin/env python3
"""Volatility Squeeze Release Alpha — Walk-Forward Backtest.

Hypothesis: When BB width compresses to historical lows (squeeze),
the subsequent expansion direction can be predicted by taker flow +
funding direction. Entry during squeeze (tight spread), exit on release.

Usage:
    python3 -m scripts.research.backtest_vol_squeeze --symbol ETHUSDT
    python3 -m scripts.research.backtest_vol_squeeze --symbol BTCUSDT --squeeze-pct 15
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


@dataclass
class SqueezeConfig:
    squeeze_percentile: float = 20    # BB width below this = squeeze
    squeeze_min_bars: int = 6         # min bars in squeeze before entry
    release_percentile: float = 50    # BB width above this = release (exit)
    direction_threshold: float = 0.15 # min |directional_pressure| to enter
    stop_loss_pct: float = 0.015      # 1.5%
    max_hold_bars: int = 72           # max 3 days
    position_frac: float = 0.30
    leverage: float = 3.0
    fee_bps: float = 2.0              # maker (limit order in squeeze)
    slippage_bps: float = 0.5         # tight spread during squeeze
    lookback: int = 500               # window for percentile calc


@dataclass
class Trade:
    entry_bar: int
    entry_price: float
    side: int
    squeeze_duration: int
    directional_pressure: float
    funding_collected: float = 0.0
    exit_bar: int = 0
    exit_price: float = 0.0
    exit_reason: str = ""
    pnl_pct: float = 0.0


def load_data(symbol: str) -> tuple[pd.DataFrame, Dict[int, float]]:
    df = pd.read_csv(f"data_files/{symbol}_1h.csv")
    funding_path = Path(f"data_files/{symbol}_funding.csv")
    schedule: Dict[int, float] = {}
    if funding_path.exists():
        with open(funding_path, newline="") as f:
            for row in csv.DictReader(f):
                schedule[int(row["timestamp"])] = float(row["funding_rate"])
    return df, schedule


def compute_features(df: pd.DataFrame, schedule: Dict[int, float],
                     lookback: int = 500) -> pd.DataFrame:
    """Compute BB width, taker flow, funding, and squeeze indicators."""
    n = len(df)
    closes = df["close"].values.astype(float)
    volumes = df["volume"].values.astype(float)
    df["high"].values.astype(float)
    df["low"].values.astype(float)
    taker_buy = df["taker_buy_volume"].values.astype(float)
    ts_col = df["open_time"].values

    # BB width (20-bar)
    bb_width = np.full(n, np.nan)
    for i in range(20, n):
        window = closes[i - 20:i]
        mu = np.mean(window)
        if mu > 0:
            bb_width[i] = np.std(window) * 2 / mu  # width as fraction of price

    # BB width percentile (rolling lookback)
    bb_pctile = np.full(n, np.nan)
    for i in range(lookback, n):
        w = bb_width[i - lookback:i]
        valid = w[~np.isnan(w)]
        if len(valid) > 50:
            bb_pctile[i] = np.searchsorted(np.sort(valid), bb_width[i]) / len(valid) * 100

    # Squeeze duration (consecutive bars below 20th percentile)
    squeeze_dur = np.zeros(n)
    for i in range(1, n):
        if not np.isnan(bb_pctile[i]) and bb_pctile[i] < 20:
            squeeze_dur[i] = squeeze_dur[i - 1] + 1

    # Taker buy ratio (EMA10)
    taker_ratio = np.where(volumes > 0, taker_buy / volumes, 0.5)
    taker_ema = np.full(n, 0.5)
    alpha = 2 / 11
    for i in range(1, n):
        taker_ema[i] = alpha * taker_ratio[i] + (1 - alpha) * taker_ema[i - 1]

    # Taker imbalance: centered at 0
    taker_imbalance = 2 * taker_ema - 1  # [-1, +1]

    # Funding direction
    funding_times = sorted(schedule.keys())
    fidx = 0
    funding_at_bar = np.full(n, 0.0)
    for i in range(n):
        ts = int(ts_col[i])
        while fidx < len(funding_times) and funding_times[fidx] <= ts:
            fidx += 1
        if fidx > 0:
            funding_at_bar[i] = schedule[funding_times[fidx - 1]]

    funding_sign = np.sign(funding_at_bar)

    # Directional pressure = taker_imbalance + 0.3 * funding_sign
    # taker flow is primary, funding is confirmation
    dir_pressure = taker_imbalance + 0.3 * funding_sign

    # Volatility (for reference)
    ret = np.diff(np.log(closes), prepend=np.log(closes[0]))
    vol20 = np.full(n, np.nan)
    for i in range(20, n):
        vol20[i] = np.std(ret[i - 20:i])

    # Is settlement bar
    is_settle = np.zeros(n, dtype=bool)
    for i in range(n):
        dt = datetime.fromtimestamp(int(ts_col[i]) / 1000, tz=timezone.utc)
        if dt.hour in (0, 8, 16):
            is_settle[i] = True

    result = df[["open_time", "open", "high", "low", "close", "volume"]].copy()
    result["bb_width"] = bb_width
    result["bb_pctile"] = bb_pctile
    result["squeeze_dur"] = squeeze_dur
    result["dir_pressure"] = dir_pressure
    result["taker_imbalance"] = taker_imbalance
    result["funding_rate"] = funding_at_bar
    result["vol20"] = vol20
    result["is_settlement"] = is_settle
    return result


def run_backtest(data: pd.DataFrame, cfg: SqueezeConfig,
                 start_bar: int = 0, end_bar: int = -1) -> List[Trade]:
    if end_bar == -1:
        end_bar = len(data)

    closes = data["close"].values.astype(float)
    highs = data["high"].values.astype(float)
    lows = data["low"].values.astype(float)
    bb_pctile = data["bb_pctile"].values.astype(float)
    squeeze_dur = data["squeeze_dur"].values.astype(float)
    dir_pressure = data["dir_pressure"].values.astype(float)
    funding = data["funding_rate"].values.astype(float)
    is_settle = data["is_settlement"].values

    trades: List[Trade] = []
    position: Optional[Trade] = None
    cost_entry = (cfg.fee_bps + cfg.slippage_bps) / 10000
    cost_exit = (cfg.fee_bps + cfg.slippage_bps) / 10000

    for i in range(start_bar, end_bar):
        c = closes[i]

        # Funding on settlement
        if position is not None and is_settle[i]:
            fr = funding[i]
            payment = -position.side * fr * cfg.position_frac * cfg.leverage
            position.funding_collected += payment

        # Exit checks
        if position is not None:
            bars_held = i - position.entry_bar
            side = position.side
            ep = position.entry_price

            # Stop loss
            stopped = False
            if side > 0 and lows[i] <= ep * (1 - cfg.stop_loss_pct):
                exit_p = ep * (1 - cfg.stop_loss_pct)
                stopped = True
            elif side < 0 and highs[i] >= ep * (1 + cfg.stop_loss_pct):
                exit_p = ep * (1 + cfg.stop_loss_pct)
                stopped = True

            if stopped:
                _close(position, i, exit_p, "stop", cfg, cost_exit)
                trades.append(position)
                position = None
                continue

            # Release exit: BB width above release_percentile
            if not np.isnan(bb_pctile[i]) and bb_pctile[i] >= cfg.release_percentile:
                _close(position, i, c, "release", cfg, cost_exit)
                trades.append(position)
                position = None
                continue

            # Max hold
            if bars_held >= cfg.max_hold_bars:
                _close(position, i, c, "max_hold", cfg, cost_exit)
                trades.append(position)
                position = None
                continue

        # Entry: squeeze confirmed + direction
        if position is None:
            if (not np.isnan(bb_pctile[i])
                    and bb_pctile[i] < cfg.squeeze_percentile
                    and squeeze_dur[i] >= cfg.squeeze_min_bars
                    and not np.isnan(dir_pressure[i])
                    and abs(dir_pressure[i]) >= cfg.direction_threshold):

                side = 1 if dir_pressure[i] > 0 else -1
                entry_p = c * (1 + side * cost_entry)  # slippage direction
                position = Trade(
                    entry_bar=i, entry_price=entry_p, side=side,
                    squeeze_duration=int(squeeze_dur[i]),
                    directional_pressure=dir_pressure[i],
                )

    # Close open
    if position is not None:
        _close(position, end_bar - 1, closes[end_bar - 1], "end", cfg, cost_exit)
        trades.append(position)

    return trades


def _close(trade: Trade, bar: int, price: float, reason: str,
           cfg: SqueezeConfig, cost: float) -> None:
    trade.exit_bar = bar
    trade.exit_price = price
    trade.exit_reason = reason
    raw_ret = trade.side * (price / trade.entry_price - 1.0)
    gross = raw_ret * cfg.position_frac * cfg.leverage
    fee = cost * 2 * cfg.position_frac * cfg.leverage
    trade.pnl_pct = gross - fee + trade.funding_collected


def walk_forward(data: pd.DataFrame, cfg: SqueezeConfig) -> None:
    n = len(data)
    min_train = 8760
    test_window = 2190
    ts_col = data["open_time"].values

    folds = []
    test_start = min_train
    while test_start + test_window <= n:
        folds.append((test_start, min(test_start + test_window, n)))
        test_start += test_window

    if not folds:
        print("  Not enough data")
        return

    print(f"  Folds: {len(folds)}, squeeze<{cfg.squeeze_percentile}pct, "
          f"min_dur={cfg.squeeze_min_bars}h, dir>{cfg.direction_threshold}")
    print(f"  stop={cfg.stop_loss_pct*100:.1f}%, release>{cfg.release_percentile}pct, "
          f"max_hold={cfg.max_hold_bars}h, lev={cfg.leverage}x")
    print(f"\n  {'Fold':<5} {'Period':<22} {'Trades':>6} {'WinR':>6} {'PnL':>8} "
          f"{'Sharpe':>7} {'AvgHold':>7} {'Exits':>20}")
    print(f"  {'-'*80}")

    all_sharpes = []
    all_returns = []
    total_trades = 0

    for fi, (te_s, te_e) in enumerate(folds):
        trades = run_backtest(data, cfg, start_bar=te_s, end_bar=te_e)
        total_trades += len(trades)

        ts0 = datetime.fromtimestamp(int(ts_col[te_s]) / 1000, tz=timezone.utc)
        ts1 = datetime.fromtimestamp(int(ts_col[min(te_e - 1, n - 1)]) / 1000, tz=timezone.utc)
        period = f"{ts0:%Y-%m}→{ts1:%Y-%m}"

        if not trades:
            print(f"  {fi:<5} {period:<22} {'0':>6}")
            all_sharpes.append(0)
            all_returns.append(0)
            continue

        pnls = [t.pnl_pct for t in trades]
        win_rate = sum(1 for p in pnls if p > 0) / len(pnls)
        total_pnl = sum(pnls)
        avg_hold = np.mean([t.exit_bar - t.entry_bar for t in trades])
        sharpe = (np.mean(pnls) / np.std(pnls) * np.sqrt(len(pnls))) if np.std(pnls) > 0 and len(pnls) > 1 else 0

        exits = {}
        for t in trades:
            exits[t.exit_reason] = exits.get(t.exit_reason, 0) + 1
        exit_str = " ".join(f"{k}:{v}" for k, v in sorted(exits.items()))

        print(f"  {fi:<5} {period:<22} {len(trades):>6} {win_rate*100:>5.1f}% "
              f"{total_pnl*100:>+7.2f}% {sharpe:>7.2f} {avg_hold:>6.1f}h {exit_str:>20}")

        all_sharpes.append(sharpe)
        all_returns.append(total_pnl)

    n_folds = len(folds)
    pos_sharpe = sum(1 for s in all_sharpes if s > 0)
    threshold = int(np.ceil(n_folds * 2 / 3))
    passed = pos_sharpe >= threshold

    print(f"  {'-'*80}")
    print(f"\n  VERDICT: {pos_sharpe}/{n_folds} positive Sharpe "
          f"(need >= {threshold}) → {'PASS' if passed else 'FAIL'}")
    print(f"  Average Sharpe: {np.mean(all_sharpes):.2f}")
    print(f"  Total return: {sum(all_returns)*100:+.2f}%")
    print(f"  Total trades: {total_trades} ({total_trades/n_folds:.0f}/fold)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Volatility Squeeze Alpha")
    parser.add_argument("--symbol", default="ETHUSDT")
    parser.add_argument("--squeeze-pct", type=float, default=20)
    parser.add_argument("--squeeze-min", type=int, default=6)
    parser.add_argument("--release-pct", type=float, default=50)
    parser.add_argument("--dir-threshold", type=float, default=0.15)
    parser.add_argument("--stop", type=float, default=0.015)
    parser.add_argument("--max-hold", type=int, default=72)
    parser.add_argument("--leverage", type=float, default=3.0)
    args = parser.parse_args()

    symbol = args.symbol.upper()
    cfg = SqueezeConfig(
        squeeze_percentile=args.squeeze_pct,
        squeeze_min_bars=args.squeeze_min,
        release_percentile=args.release_pct,
        direction_threshold=args.dir_threshold,
        stop_loss_pct=args.stop,
        max_hold_bars=args.max_hold,
        leverage=args.leverage,
    )

    print(f"\n{'='*80}")
    print(f"  Volatility Squeeze Alpha: {symbol}")
    print(f"{'='*80}")

    df, schedule = load_data(symbol)
    data = compute_features(df, schedule, lookback=cfg.lookback)
    print(f"  Bars: {len(data)}")

    # Stats
    bb = data["bb_pctile"].values
    valid_bb = bb[~np.isnan(bb)]
    squeeze_bars = np.sum(valid_bb < cfg.squeeze_percentile)
    print(f"  Squeeze bars: {squeeze_bars} ({squeeze_bars/len(valid_bb)*100:.1f}%)")

    walk_forward(data, cfg)


if __name__ == "__main__":
    main()
