#!/usr/bin/env python3
"""Research: IC decay analysis for all features.

Computes IC at multiple horizons to identify fast-decay vs slow-decay features.
Fast-decay → suited for short-term; slow-decay → suited for longer holding.

Usage:
    python3 -m scripts.research_ic_decay
    python3 -m scripts.research_ic_decay --symbol ETHUSDT
"""
from __future__ import annotations

import argparse
import csv
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from features.enriched_computer import EnrichedFeatureComputer, ENRICHED_FEATURE_NAMES

logger = logging.getLogger(__name__)

HORIZONS = [1, 2, 3, 5, 8, 12, 24, 48]


def _load_funding_schedule(path: Path) -> Dict[int, float]:
    schedule: Dict[int, float] = {}
    if not path.exists():
        return schedule
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            schedule[int(row["timestamp"])] = float(row["funding_rate"])
    return schedule


def compute_all_features(symbol: str) -> pd.DataFrame:
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


def ic_decay_analysis(feat_df: pd.DataFrame) -> pd.DataFrame:
    closes = feat_df["close"].values
    feature_names = [f for f in ENRICHED_FEATURE_NAMES if f in feat_df.columns]
    results = []

    for feat_name in feature_names:
        vals = feat_df[feat_name].values
        row = {"feature": feat_name}
        ic_values = []

        for h in HORIZONS:
            fwd_ret = np.full(len(closes), np.nan)
            for i in range(len(closes) - h):
                if closes[i] != 0:
                    fwd_ret[i] = closes[i + h] / closes[i] - 1.0
            mask = ~np.isnan(vals.astype(float)) & ~np.isnan(fwd_ret)
            x = vals[mask].astype(float)
            y = fwd_ret[mask]
            if len(x) > 20:
                ic = float(np.corrcoef(x, y)[0, 1])
            else:
                ic = 0.0
            row[f"IC_h{h}"] = ic
            ic_values.append(abs(ic))

        # Classify decay speed
        if len(ic_values) >= 3:
            short_ic = np.mean(ic_values[:3])  # h1-h3
            long_ic = np.mean(ic_values[-3:])   # h12-h48
            if short_ic > 0:
                row["decay_ratio"] = long_ic / short_ic
            else:
                row["decay_ratio"] = 0.0
            row["peak_horizon"] = HORIZONS[np.argmax(ic_values)]
            row["max_abs_ic"] = max(ic_values)
        else:
            row["decay_ratio"] = 0.0
            row["peak_horizon"] = 0
            row["max_abs_ic"] = 0.0

        results.append(row)

    return pd.DataFrame(results)


def main() -> None:
    parser = argparse.ArgumentParser(description="IC decay analysis")
    parser.add_argument("--symbol", default="BTCUSDT")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    symbol = args.symbol.upper()

    print(f"\n{'='*80}")
    print(f"  IC Decay Analysis: {symbol}")
    print(f"{'='*80}")

    feat_df = compute_all_features(symbol)
    print(f"  Total bars: {len(feat_df)}")

    decay_df = ic_decay_analysis(feat_df)

    # Sort by max IC
    decay_df = decay_df.sort_values("max_abs_ic", ascending=False)

    # Print header
    print(f"\n  {'Feature':<25s}", end="")
    for h in HORIZONS:
        print(f" h{h:>2d}", end="")
    print(f" {'Peak':>5s} {'Decay':>6s} {'Type':<12s}")
    print(f"  {'-'*100}")

    for _, row in decay_df.iterrows():
        feat = row["feature"]
        print(f"  {feat:<25s}", end="")
        for h in HORIZONS:
            val = row[f"IC_h{h}"]
            print(f" {val:>+.3f}" if abs(val) >= 0.005 else f" {'.'*4:>4s}", end="")

        peak = int(row["peak_horizon"])
        decay = row["decay_ratio"]
        if decay < 0.3:
            decay_type = "FAST-DECAY"
        elif decay > 0.7:
            decay_type = "SLOW-DECAY"
        else:
            decay_type = "MODERATE"
        print(f" h{peak:>3d} {decay:>6.2f} {decay_type:<12s}")

    # Save
    out_dir = Path("output/research")
    out_dir.mkdir(parents=True, exist_ok=True)
    decay_df.to_csv(out_dir / f"{symbol}_ic_decay.csv", index=False)
    print(f"\n  Saved to {out_dir}/{symbol}_ic_decay.csv")


if __name__ == "__main__":
    main()
