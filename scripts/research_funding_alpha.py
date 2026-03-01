#!/usr/bin/env python3
"""Research: Funding rate reversal alpha.

Tests hypothesis: extreme positive funding → short (crowded long reversal).
Computes IC, win rate, and conditional Sharpe for funding-based signals.

Usage:
    python3 -m scripts.research_funding_alpha
    python3 -m scripts.research_funding_alpha --symbol ETHUSDT
"""
from __future__ import annotations

import argparse
import csv
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from features.enriched_computer import EnrichedFeatureComputer

logger = logging.getLogger(__name__)

FUNDING_FEATURES = [
    "funding_rate", "funding_ma8", "funding_zscore_24",
    "funding_momentum", "funding_extreme", "funding_cumulative_8",
    "funding_sign_persist",
]

HORIZONS = [1, 3, 5, 8, 24]


def _load_funding_schedule(path: Path) -> Dict[int, float]:
    schedule: Dict[int, float] = {}
    if not path.exists():
        return schedule
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            schedule[int(row["timestamp"])] = float(row["funding_rate"])
    return schedule


def compute_features(symbol: str) -> pd.DataFrame:
    csv_path = Path(f"data_files/{symbol}_1h.csv")
    funding_path = Path(f"data_files/{symbol}_funding.csv")
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path}")

    df = pd.read_csv(csv_path)
    funding_schedule = _load_funding_schedule(funding_path)
    funding_times = sorted(funding_schedule.keys())
    funding_idx = 0

    comp = EnrichedFeatureComputer()
    records = []

    for _, row in df.iterrows():
        close = float(row["close"])
        volume = float(row.get("volume", 0))
        high = float(row.get("high", close))
        low = float(row.get("low", close))
        open_ = float(row.get("open", close))
        trades = float(row.get("trades", 0) or 0)
        taker_buy_volume = float(row.get("taker_buy_volume", 0) or 0)
        quote_volume = float(row.get("quote_volume", 0) or 0)

        ts_raw = row.get("timestamp") or row.get("open_time", "")
        hour, dow, ts_ms = -1, -1, 0
        if ts_raw:
            try:
                ts_ms = int(ts_raw)
                dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
                hour, dow = dt.hour, dt.weekday()
            except (ValueError, OSError):
                pass

        funding_rate = None
        while funding_idx < len(funding_times) and funding_times[funding_idx] <= ts_ms:
            funding_rate = funding_schedule[funding_times[funding_idx]]
            funding_idx += 1
        if funding_rate is None and funding_idx > 0:
            funding_rate = funding_schedule[funding_times[funding_idx - 1]]

        feats = comp.on_bar(
            symbol, close=close, volume=volume, high=high, low=low,
            open_=open_, hour=hour, dow=dow, funding_rate=funding_rate,
            trades=trades, taker_buy_volume=taker_buy_volume,
            quote_volume=quote_volume,
        )
        feats["close"] = close
        records.append(feats)

    return pd.DataFrame(records)


def analyze_extremes(feat_df: pd.DataFrame, horizons: List[int]) -> None:
    """Analyze forward returns after extreme funding events."""
    closes = feat_df["close"].values
    zscore = feat_df["funding_zscore_24"].values.astype(float)
    valid = ~np.isnan(zscore)

    for h in horizons:
        fwd = np.full(len(closes), np.nan)
        for i in range(len(closes) - h):
            if closes[i] != 0:
                fwd[i] = closes[i + h] / closes[i] - 1.0

        mask = valid & ~np.isnan(fwd)
        z = zscore[mask]
        r = fwd[mask]

        if len(z) < 50:
            continue

        # Top 10% funding → crowded longs
        top_pct = np.percentile(z, 90)
        bot_pct = np.percentile(z, 10)

        top_mask = z >= top_pct
        bot_mask = z <= bot_pct
        mid_mask = ~top_mask & ~bot_mask

        top_ret = r[top_mask]
        bot_ret = r[bot_mask]
        mid_ret = r[mid_mask]

        print(f"\n    Horizon {h}h:")
        for label, rets in [("Top 10% (crowded long)", top_ret),
                            ("Bot 10% (crowded short)", bot_ret),
                            ("Middle 80%", mid_ret)]:
            if len(rets) < 5:
                continue
            mean_r = float(np.mean(rets))
            std_r = float(np.std(rets))
            sharpe = mean_r / std_r * np.sqrt(8760 / h) if std_r > 0 else 0
            win = float(np.mean(rets > 0))
            print(f"      {label:<25s}: mean={mean_r*100:>+6.3f}%  "
                  f"win={win*100:>5.1f}%  sharpe={sharpe:>6.2f}  n={len(rets)}")

        # Reversal test: short when top, long when bottom
        reversal_ret = np.concatenate([-top_ret, bot_ret])
        if len(reversal_ret) > 10:
            rev_mean = float(np.mean(reversal_ret))
            rev_std = float(np.std(reversal_ret))
            rev_sharpe = rev_mean / rev_std * np.sqrt(8760 / h) if rev_std > 0 else 0
            rev_win = float(np.mean(reversal_ret > 0))
            print(f"      {'REVERSAL STRATEGY':<25s}: mean={rev_mean*100:>+6.3f}%  "
                  f"win={rev_win*100:>5.1f}%  sharpe={rev_sharpe:>6.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Funding alpha research")
    parser.add_argument("--symbol", default="BTCUSDT")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    symbol = args.symbol.upper()

    print(f"\n{'='*70}")
    print(f"  Funding Alpha Research: {symbol}")
    print(f"{'='*70}")

    feat_df = compute_features(symbol)
    print(f"  Total bars: {len(feat_df)}")

    # IC for all funding features
    print(f"\n  IC Analysis:")
    print(f"  {'Feature':<25s}", end="")
    for h in HORIZONS:
        print(f"  IC_h{h:>2d}", end="")
    print()

    closes = feat_df["close"].values
    for feat_name in FUNDING_FEATURES:
        vals = feat_df[feat_name].values
        print(f"  {feat_name:<25s}", end="")
        for h in HORIZONS:
            fwd = np.full(len(closes), np.nan)
            for i in range(len(closes) - h):
                if closes[i] != 0:
                    fwd[i] = closes[i + h] / closes[i] - 1.0
            mask = ~np.isnan(vals.astype(float)) & ~np.isnan(fwd)
            x, y = vals[mask].astype(float), fwd[mask]
            ic = float(np.corrcoef(x, y)[0, 1]) if len(x) > 20 else 0.0
            marker = " *" if abs(ic) > 0.02 else "  "
            print(f"  {ic:>6.4f}{marker}", end="")
        print()

    # Extreme funding analysis
    print(f"\n  Extreme Funding Reversal Analysis:")
    analyze_extremes(feat_df, HORIZONS)

    out_dir = Path("output/research")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n  Done. Results printed above.")


if __name__ == "__main__":
    main()
