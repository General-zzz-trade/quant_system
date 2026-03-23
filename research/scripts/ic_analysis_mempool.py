#!/usr/bin/env python3
"""IC analysis for mempool features — Spearman IC vs 5-bar forward return.

Requires:
    - data_files/BTCUSDT_1h.csv (OHLCV)
    - data_files/btc_mempool_fees.csv (from download_mempool.py)
    - Standard schedule CSVs

Usage:
    python scripts/ic_analysis_mempool.py
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import pandas as pd
from scipy.stats import spearmanr

from features.enriched_computer import EnrichedFeatureComputer


def load_schedule(path: str, ts_col: str, val_col: str) -> Dict[int, float]:
    schedule: Dict[int, float] = {}
    if not Path(path).exists():
        return schedule
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            raw_ts = row[ts_col]
            if "T" in raw_ts or "-" in raw_ts:
                dt = datetime.fromisoformat(raw_ts.replace("Z", "+00:00"))
                ts_ms = int(dt.timestamp() * 1000)
            else:
                ts_ms = int(raw_ts)
            val_str = row[val_col]
            if val_str == "":
                continue
            schedule[ts_ms] = float(val_str)
    return schedule


def load_mempool(path: str) -> Dict[int, Dict[str, float]]:
    result: Dict[int, Dict[str, float]] = {}
    if not Path(path).exists():
        return result
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            ts_ms = int(row["timestamp"])
            result[ts_ms] = {
                "fastest_fee": float(row.get("max_fee", row.get("avg_fee", 0))),
                "economy_fee": float(row.get("min_fee", 1)),
                "mempool_size": float(row.get("avg_fee", 0)) * 1000,  # proxy
            }
    return result


def main():
    symbol = "BTCUSDT"

    df = pd.read_csv(f"data_files/{symbol}_1h.csv")
    funding = load_schedule(f"data_files/{symbol}_funding.csv", "timestamp", "funding_rate")
    oi = load_schedule(f"data_files/{symbol}_open_interest.csv", "timestamp", "sum_open_interest")
    ls = load_schedule(f"data_files/{symbol}_ls_ratio.csv", "timestamp", "long_short_ratio")
    spot = load_schedule(f"data_files/{symbol}_spot_1h.csv", "open_time", "close")
    fgi = load_schedule("data_files/fear_greed_index.csv", "timestamp", "value")

    mempool = load_mempool("data_files/btc_mempool_fees.csv")
    if not mempool:
        print("ERROR: No mempool data. Run: python scripts/download_mempool.py")
        return

    print(f"Mempool data: {len(mempool)} records")

    all_schedules = [
        (sorted(funding.keys()), funding),
        (sorted(oi.keys()), oi),
        (sorted(ls.keys()), ls),
        (sorted(spot.keys()), spot),
        (sorted(fgi.keys()), fgi),
    ]
    idxs = [0] * len(all_schedules)

    mp_times = sorted(mempool.keys())
    mp_idx = 0

    comp = EnrichedFeatureComputer()
    records = []

    for _, row in df.iterrows():
        close = float(row["close"])
        volume = float(row.get("volume", 0))
        high = float(row.get("high", close))
        low = float(row.get("low", close))
        open_ = float(row.get("open", close))
        trades = float(row.get("trades", 0) or 0)
        tbv = float(row.get("taker_buy_volume", 0) or 0)
        qv = float(row.get("quote_volume", 0) or 0)
        tbqv = float(row.get("taker_buy_quote_volume", 0) or 0)

        ts_raw = row.get("timestamp") or row.get("open_time", "")
        hour, dow, ts_ms = -1, -1, 0
        if ts_raw:
            ts_ms = int(ts_raw)
            dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
            hour, dow = dt.hour, dt.weekday()

        vals = []
        for i, (times, sched) in enumerate(all_schedules):
            val = None
            while idxs[i] < len(times) and times[idxs[i]] <= ts_ms:
                val = sched[times[idxs[i]]]
                idxs[i] += 1
            if val is None and idxs[i] > 0:
                val = sched[times[idxs[i] - 1]]
            vals.append(val)

        fr, oi_val, ls_val, sp_val, fg_val = vals

        # Advance mempool pointer
        mp_val = None
        while mp_idx < len(mp_times) and mp_times[mp_idx] <= ts_ms:
            mp_val = mempool[mp_times[mp_idx]]
            mp_idx += 1
        if mp_val is None and mp_idx > 0:
            mp_val = mempool[mp_times[mp_idx - 1]]

        feats = comp.on_bar(
            symbol, close=close, volume=volume, high=high, low=low,
            open_=open_, hour=hour, dow=dow, funding_rate=fr,
            trades=trades, taker_buy_volume=tbv, quote_volume=qv,
            taker_buy_quote_volume=tbqv,
            open_interest=oi_val, ls_ratio=ls_val,
            spot_close=sp_val, fear_greed=fg_val,
            mempool_metrics=mp_val,
        )
        records.append(feats)

    feat_df = pd.DataFrame(records)
    feat_df["close"] = df["close"].values
    target = feat_df["close"].shift(-5) / feat_df["close"] - 1.0

    features = [
        "mempool_fee_zscore_24",
        "mempool_size_zscore_24",
        "fee_urgency_ratio",
    ]

    print()
    print("=" * 70)
    print("  Mempool Feature IC Analysis (Spearman, BTC 5-bar forward return)")
    print("=" * 70)
    print(f"  Total bars: {len(feat_df):,}")
    print()
    print(f"  {'Feature':<30} {'Non-null':>8} {'IC':>8} {'|IC|':>8} {'PASS':>6}")
    print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*8} {'-'*6}")

    passed = []
    for feat_name in features:
        if feat_name not in feat_df.columns:
            print(f"  {feat_name:<30} {'MISSING':>8}")
            continue
        col = feat_df[feat_name].astype(float)
        valid = col.notna() & target.notna()
        n_valid = int(valid.sum())
        if n_valid < 100:
            print(f"  {feat_name:<30} {n_valid:>8} {'N/A':>8} {'N/A':>8} {'SKIP':>6}")
            continue
        ic, _ = spearmanr(col[valid], target[valid])
        abs_ic = abs(ic)
        ok = abs_ic > 0.02
        if ok:
            passed.append((feat_name, ic))
        print(f"  {feat_name:<30} {n_valid:>8} {ic:>8.4f} {abs_ic:>8.4f} {'YES' if ok else 'NO':>6}")

    print()
    print(f"  Passed features (|IC| > 0.02): {len(passed)}/{len(features)}")
    for f, ic in passed:
        print(f"    - {f} (IC={ic:.4f})")
    print("=" * 70)

    return passed


if __name__ == "__main__":
    main()
