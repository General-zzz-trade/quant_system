#!/usr/bin/env python3
"""Research: IC analysis for kline microstructure features.

Computes Information Coefficient (IC) at multiple horizons for each new
microstructure feature, plus correlation analysis against existing features.

Usage:
    python3 -m scripts.research_microstructure_alpha
    python3 -m scripts.research_microstructure_alpha --symbol ETHUSDT
"""
from __future__ import annotations

import argparse
import csv
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from features.enriched_computer import EnrichedFeatureComputer, ENRICHED_FEATURE_NAMES

logger = logging.getLogger(__name__)

MICRO_FEATURES = [
    "trade_intensity", "taker_buy_ratio", "taker_buy_ratio_ma10",
    "taker_imbalance", "avg_trade_size", "avg_trade_size_ratio",
    "volume_per_trade", "trade_count_regime",
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


def compute_all_features(symbol: str) -> pd.DataFrame:
    csv_path = Path(f"data_files/{symbol}_1h.csv")
    funding_path = Path(f"data_files/{symbol}_funding.csv")
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found")

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


def ic_analysis(feat_df: pd.DataFrame, feature_names: List[str],
                horizons: List[int]) -> pd.DataFrame:
    """Compute IC for each feature at each horizon."""
    closes = feat_df["close"].values
    results = []

    for feat_name in feature_names:
        vals = feat_df[feat_name].values
        row = {"feature": feat_name}
        for h in horizons:
            fwd_ret = np.empty(len(closes))
            fwd_ret[:] = np.nan
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
        results.append(row)

    return pd.DataFrame(results)


def correlation_analysis(feat_df: pd.DataFrame, micro_features: List[str],
                         all_features: List[str]) -> pd.DataFrame:
    """Correlation between micro features and all features."""
    corrs = []
    for mf in micro_features:
        mvals = feat_df[mf].values
        row = {"feature": mf}
        for af in all_features:
            if af in micro_features:
                continue
            avals = feat_df[af].values
            mask = ~pd.isna(mvals) & ~pd.isna(avals)
            if mask.sum() > 20:
                c = float(np.corrcoef(mvals[mask].astype(float),
                                      avals[mask].astype(float))[0, 1])
            else:
                c = 0.0
            row[af] = c
        corrs.append(row)
    return pd.DataFrame(corrs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Microstructure IC research")
    parser.add_argument("--symbol", default="BTCUSDT")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    symbol = args.symbol.upper()

    print(f"\n{'='*70}")
    print(f"  Microstructure Alpha Research: {symbol}")
    print(f"{'='*70}")

    feat_df = compute_all_features(symbol)
    print(f"  Total bars: {len(feat_df)}")

    # IC analysis
    print(f"\n  IC Analysis (horizons: {HORIZONS})")
    print(f"  {'-'*60}")
    ic_df = ic_analysis(feat_df, MICRO_FEATURES, HORIZONS)
    print(f"  {'Feature':<25s}", end="")
    for h in HORIZONS:
        print(f"  IC_h{h:>2d}", end="")
    print()
    for _, row in ic_df.iterrows():
        print(f"  {row['feature']:<25s}", end="")
        for h in HORIZONS:
            val = row[f"IC_h{h}"]
            marker = " *" if abs(val) > 0.02 else "  "
            print(f"  {val:>6.4f}{marker}", end="")
        print()

    # Correlation with existing features
    existing = [f for f in ENRICHED_FEATURE_NAMES if f not in MICRO_FEATURES]
    corr_df = correlation_analysis(feat_df, MICRO_FEATURES, existing)

    print(f"\n  Top Correlated Existing Features (|r| > 0.3):")
    print(f"  {'-'*60}")
    for _, row in corr_df.iterrows():
        feat = row["feature"]
        other_cols = [c for c in row.index if c != "feature"]
        sorted_corrs = sorted(other_cols, key=lambda c: abs(row[c]), reverse=True)
        high = [(c, row[c]) for c in sorted_corrs[:3] if abs(row[c]) > 0.3]
        if high:
            pairs = ", ".join(f"{c}={v:.3f}" for c, v in high)
            print(f"  {feat:<25s} → {pairs}")
        else:
            print(f"  {feat:<25s} → (low correlation, good!)")

    # Save results
    out_dir = Path("output/research")
    out_dir.mkdir(parents=True, exist_ok=True)
    ic_df.to_csv(out_dir / f"{symbol}_microstructure_ic.csv", index=False)
    print(f"\n  Results saved to {out_dir}/{symbol}_microstructure_ic.csv")


if __name__ == "__main__":
    main()
