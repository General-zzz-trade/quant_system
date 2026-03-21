#!/usr/bin/env python3
"""ETH 15m Alpha Optimization — tighter params for short-timeframe alpha.

The 15m ETH alpha barely passes (Sharpe 1.04). This script tests:
  1. Tighter deadzone (0.3 vs current 0.5)
  2. Shorter min_hold (8 bars = 2h vs 16 bars = 4h)
  3. Microstructure feature integration (VPIN, ob_imbalance)
  4. Faster z-score window (180 vs 720)

Goal: push ETH 15m from Sharpe 1.04 → 1.5+ for stronger COMBO signal.

Usage:
    python3 -m scripts.research.eth_15m_optimizer
    python3 -m scripts.research.eth_15m_optimizer --sweep  # param sweep
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from features.enriched_computer import EnrichedFeatureComputer
from scripts.shared.signal_postprocess import apply_signal_constraints

log = logging.getLogger("eth_15m_opt")


@dataclass
class ParamSet:
    """Parameter set for 15m alpha optimization."""
    deadzone: float = 0.3
    min_hold: int = 8           # 8 bars × 15m = 2 hours
    max_hold: int = 40          # 40 bars × 15m = 10 hours
    zscore_window: int = 480    # 480 × 15m = 5 days
    zscore_warmup: int = 120    # 120 × 15m = 30 hours
    long_only: bool = True


# Sweep grid
SWEEP_GRID = {
    "deadzone": [0.2, 0.3, 0.4, 0.5],
    "min_hold": [4, 8, 12, 16],
    "zscore_window": [180, 360, 480, 720],
}


def load_15m_data(symbol: str = "ETHUSDT") -> pd.DataFrame:
    """Load 15m kline data."""
    path = f"data_files/{symbol}_15m.csv"
    if not Path(path).exists():
        log.error("Missing data: %s", path)
        log.error("Run: python3 -m scripts.data.download_15m_klines")
        sys.exit(1)
    return pd.read_csv(path)


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute enriched features on 15m bars."""
    comp = EnrichedFeatureComputer()
    records = []
    for _, row in df.iterrows():
        feats = comp.on_bar(
            "ETHUSDT",
            close=float(row["close"]),
            volume=float(row.get("volume", 0)),
            high=float(row.get("high", row["close"])),
            low=float(row.get("low", row["close"])),
            open_=float(row.get("open", row["close"])),
            trades=float(row.get("trades", 0) or 0),
            taker_buy_volume=float(row.get("taker_buy_volume", 0) or 0),
            quote_volume=float(row.get("quote_volume", 0) or 0),
            taker_buy_quote_volume=float(row.get("taker_buy_quote_volume", 0) or 0),
        )
        records.append(feats)
    return pd.DataFrame(records)


def evaluate_params(
    feat_df: pd.DataFrame,
    predictions: np.ndarray,
    close: np.ndarray,
    params: ParamSet,
) -> Dict[str, float]:
    """Evaluate signal pipeline with given params, return metrics."""
    n = len(predictions)
    signals = apply_signal_constraints(
        predictions,
        zscore_window=params.zscore_window,
        zscore_warmup=params.zscore_warmup,
        deadzone=params.deadzone,
        min_hold=params.min_hold,
        max_hold=params.max_hold,
        long_only=params.long_only,
    )

    # Compute returns
    forward_ret = np.zeros(n)
    forward_ret[:-1] = close[1:] / close[:-1] - 1.0
    pnl = signals * forward_ret

    # Metrics
    total_ret = float(np.sum(pnl))
    n_trades = int(np.sum(np.diff(signals) != 0))
    mean_pnl = float(np.mean(pnl))
    std_pnl = float(np.std(pnl)) if np.std(pnl) > 0 else 1e-8
    sharpe = mean_pnl / std_pnl * np.sqrt(365 * 24 * 4)  # 15m bars annualized

    # Win rate
    trade_pnls = []
    in_trade = False
    trade_start_pnl = 0.0
    cum_pnl = 0.0
    for i in range(n):
        cum_pnl += pnl[i]
        if signals[i] != 0 and not in_trade:
            in_trade = True
            trade_start_pnl = cum_pnl
        elif signals[i] == 0 and in_trade:
            in_trade = False
            trade_pnls.append(cum_pnl - trade_start_pnl)

    wins = sum(1 for p in trade_pnls if p > 0)
    win_rate = wins / max(len(trade_pnls), 1) * 100

    return {
        "sharpe": round(sharpe, 3),
        "total_return": round(total_ret * 100, 2),
        "n_trades": n_trades,
        "win_rate": round(win_rate, 1),
        "mean_pnl_bps": round(mean_pnl * 10000, 2),
        "deadzone": params.deadzone,
        "min_hold": params.min_hold,
        "zscore_window": params.zscore_window,
    }


def param_sweep(
    feat_df: pd.DataFrame,
    predictions: np.ndarray,
    close: np.ndarray,
) -> List[Dict]:
    """Sweep parameter grid and rank by Sharpe."""
    results = []
    for dz, mh, zw in product(
        SWEEP_GRID["deadzone"],
        SWEEP_GRID["min_hold"],
        SWEEP_GRID["zscore_window"],
    ):
        params = ParamSet(
            deadzone=dz,
            min_hold=mh,
            zscore_window=zw,
            zscore_warmup=min(zw // 4, 180),
        )
        metrics = evaluate_params(feat_df, predictions, close, params)
        results.append(metrics)

    results.sort(key=lambda x: x["sharpe"], reverse=True)
    return results


def main():
    parser = argparse.ArgumentParser(description="ETH 15m Alpha Optimizer")
    parser.add_argument("--sweep", action="store_true", help="Run full param sweep")
    parser.add_argument("--top-n", type=int, default=10, help="Show top N results")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    log.info("Loading ETH 15m data...")
    df = load_15m_data()
    log.info("Loaded %d bars", len(df))

    log.info("Computing features...")
    feat_df = compute_features(df)

    close = df["close"].values.astype(float)

    # Use simple momentum as prediction proxy (replace with actual model in production)
    # This validates the pipeline; actual performance depends on model quality
    predictions = feat_df.get("ret_12", pd.Series(np.zeros(len(feat_df)))).values.astype(float)
    predictions = np.nan_to_num(predictions, 0.0)

    if args.sweep:
        log.info("Running parameter sweep (%d combinations)...",
                 len(SWEEP_GRID["deadzone"]) * len(SWEEP_GRID["min_hold"]) * len(SWEEP_GRID["zscore_window"]))
        results = param_sweep(feat_df, predictions, close)

        print()
        print("=" * 90)
        print("  ETH 15m Parameter Sweep Results (ranked by Sharpe)")
        print("=" * 90)
        print(f"  {'Rank':>4} {'Sharpe':>8} {'Return%':>8} {'Trades':>7} {'WR%':>6} "
              f"{'DZ':>5} {'MinH':>5} {'ZW':>5}")
        print(f"  {'-'*4} {'-'*8} {'-'*8} {'-'*7} {'-'*6} {'-'*5} {'-'*5} {'-'*5}")

        for i, r in enumerate(results[:args.top_n]):
            print(f"  {i+1:>4} {r['sharpe']:>8.3f} {r['total_return']:>8.2f} "
                  f"{r['n_trades']:>7} {r['win_rate']:>6.1f} "
                  f"{r['deadzone']:>5.2f} {r['min_hold']:>5} {r['zscore_window']:>5}")
        print("=" * 90)
    else:
        # Single evaluation with optimized defaults
        params = ParamSet()
        metrics = evaluate_params(feat_df, predictions, close, params)

        print()
        print("ETH 15m Alpha Evaluation (optimized params)")
        print(f"  Params: deadzone={params.deadzone}, min_hold={params.min_hold}, "
              f"zscore_window={params.zscore_window}")
        for k, v in metrics.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
