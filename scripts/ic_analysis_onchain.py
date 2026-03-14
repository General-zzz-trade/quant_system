#!/usr/bin/env python3
"""IC analysis for on-chain features — Spearman IC vs 5-bar forward return.

Requires:
    - data_files/BTCUSDT_1h.csv (OHLCV)
    - data/onchain/btc_onchain_combined.csv (from download_onchain_metrics.py)
    - Standard schedule CSVs (funding, OI, LS, spot, FGI, IV)

Usage:
    python scripts/ic_analysis_onchain.py
"""
from __future__ import annotations

import csv
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

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


def load_onchain_combined(path: str) -> Dict[int, Dict[str, float]]:
    """Load combined on-chain CSV into {ts_ms -> {metric -> value}} dict."""
    result: Dict[int, Dict[str, float]] = {}
    if not Path(path).exists():
        return result
    metrics = ["FlowInExUSD", "FlowOutExUSD", "SplyExNtv", "AdrActCnt", "TxTfrCnt", "HashRate"]
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            ts_ms = int(row["timestamp"])
            d: Dict[str, float] = {}
            for m in metrics:
                val = row.get(m, "")
                if val != "":
                    d[m] = float(val)
            if d:
                result[ts_ms] = d
    return result


def main() -> None:
    symbol = "BTCUSDT"
    asset = "btc"

    # Load 1h OHLCV
    df = pd.read_csv(f"data_files/{symbol}_1h.csv")

    # Load standard schedules
    funding = load_schedule(f"data_files/{symbol}_funding.csv", "timestamp", "funding_rate")
    oi = load_schedule(f"data_files/{symbol}_open_interest.csv", "timestamp", "sum_open_interest")
    ls = load_schedule(f"data_files/{symbol}_ls_ratio.csv", "timestamp", "long_short_ratio")
    spot = load_schedule(f"data_files/{symbol}_spot_1h.csv", "open_time", "close")
    fgi = load_schedule("data_files/fear_greed_index.csv", "timestamp", "value")
    iv = load_schedule(f"data_files/{symbol}_deribit_iv.csv", "timestamp", "implied_vol")
    pcr = load_schedule(f"data_files/{symbol}_deribit_iv.csv", "timestamp", "put_call_ratio")

    # Load on-chain data
    onchain_path = f"data/onchain/{asset}_onchain_combined.csv"
    onchain = load_onchain_combined(onchain_path)
    if not onchain:
        print(f"ERROR: No on-chain data found at {onchain_path}")
        print(f"Run: python scripts/download_onchain_metrics.py --asset {asset} --start 2020-01-01")
        sys.exit(1)

    onchain_times = sorted(onchain.keys())
    print(f"On-chain data: {len(onchain_times)} daily records")
    print(f"  Range: {datetime.fromtimestamp(onchain_times[0]/1000, tz=timezone.utc).date()} "
          f"to {datetime.fromtimestamp(onchain_times[-1]/1000, tz=timezone.utc).date()}")

    all_schedules = [
        (sorted(funding.keys()), funding),
        (sorted(oi.keys()), oi),
        (sorted(ls.keys()), ls),
        (sorted(spot.keys()), spot),
        (sorted(fgi.keys()), fgi),
        (sorted(iv.keys()), iv),
        (sorted(pcr.keys()), pcr),
    ]
    idxs = [0] * len(all_schedules)

    # On-chain schedule pointer
    oc_idx = 0

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

        # Advance standard schedule pointers
        vals = []
        for i, (times, sched) in enumerate(all_schedules):
            val = None
            while idxs[i] < len(times) and times[idxs[i]] <= ts_ms:
                val = sched[times[idxs[i]]]
                idxs[i] += 1
            if val is None and idxs[i] > 0:
                val = sched[times[idxs[i] - 1]]
            vals.append(val)

        fr, oi_val, ls_val, sp_val, fg_val, iv_val, pcr_val = vals

        # Advance on-chain pointer (forward-fill daily data)
        oc_val: Optional[Dict[str, float]] = None
        while oc_idx < len(onchain_times) and onchain_times[oc_idx] <= ts_ms:
            oc_val = onchain[onchain_times[oc_idx]]
            oc_idx += 1
        if oc_val is None and oc_idx > 0:
            oc_val = onchain[onchain_times[oc_idx - 1]]

        feats = comp.on_bar(
            symbol, close=close, volume=volume, high=high, low=low,
            open_=open_, hour=hour, dow=dow, funding_rate=fr,
            trades=trades, taker_buy_volume=tbv, quote_volume=qv,
            taker_buy_quote_volume=tbqv,
            open_interest=oi_val, ls_ratio=ls_val,
            spot_close=sp_val, fear_greed=fg_val,
            implied_vol=iv_val, put_call_ratio=pcr_val,
            onchain_metrics=oc_val,
        )
        records.append(feats)

    feat_df = pd.DataFrame(records)
    feat_df["close"] = df["close"].values
    target = feat_df["close"].shift(-5) / feat_df["close"] - 1.0

    onchain_features = [
        "exchange_netflow_zscore",
        "exchange_supply_change",
        "exchange_supply_zscore_30",
        "active_addr_zscore_14",
        "tx_count_zscore_14",
        "hashrate_momentum",
    ]

    print()
    print("=" * 70)
    print("  On-Chain Feature IC Analysis (Spearman, BTC 5-bar forward return)")
    print("=" * 70)
    print(f"  Total bars: {len(feat_df):,}")
    print()
    print(f"  {'Feature':<30} {'Non-null':>8} {'IC':>8} {'|IC|':>8} {'PASS':>6}")
    print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*8} {'-'*6}")

    passed = []
    for feat_name in onchain_features:
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
    print(f"  Passed features (|IC| > 0.02): {len(passed)}/{len(onchain_features)}")
    for f, ic in passed:
        print(f"    - {f} (IC={ic:.4f})")

    # Reference ICs
    print()
    print("  Reference IC (existing features):")
    ref_feats = ["basis", "funding_zscore_24", "rsi_14", "atr_norm_14", "cvd_20", "parkinson_vol"]
    for feat_name in ref_feats:
        if feat_name not in feat_df.columns:
            continue
        col = feat_df[feat_name].astype(float)
        valid = col.notna() & target.notna()
        n_valid = int(valid.sum())
        if n_valid < 100:
            continue
        ic, _ = spearmanr(col[valid], target[valid])
        print(f"    {feat_name:<30} IC={ic:>8.4f}")
    print("=" * 70)

    return passed


if __name__ == "__main__":
    main()
