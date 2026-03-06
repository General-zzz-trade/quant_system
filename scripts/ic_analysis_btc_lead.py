#!/usr/bin/env python3
"""IC analysis: BTC features as leading indicators for SOL 5-bar forward return.

Computes BTC enriched features, aligns to SOL timestamps, then tests
Spearman IC of each BTC feature against SOL's forward return.

Usage:
    python scripts/ic_analysis_btc_lead.py
"""
from __future__ import annotations

import csv
from collections import deque
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


def load_liquidation_proxy(path: str) -> Dict[int, Dict[str, float]]:
    result: Dict[int, Dict[str, float]] = {}
    if not Path(path).exists():
        return result
    df = pd.read_csv(path)
    for _, row in df.iterrows():
        ts = int(row["ts"])
        result[ts] = {
            "liq_total_volume": float(row.get("liq_proxy_volume", 0)),
            "liq_buy_volume": float(row.get("liq_proxy_buy", 0)),
            "liq_sell_volume": float(row.get("liq_proxy_sell", 0)),
            "liq_count": 1.0 if float(row.get("liq_proxy_volume", 0)) > 0 else 0.0,
        }
    return result


def compute_btc_features(btc_df: pd.DataFrame) -> pd.DataFrame:
    """Compute full enriched features for BTC, return DataFrame indexed by ts_ms."""
    symbol = "BTCUSDT"

    funding = load_schedule(f"data_files/{symbol}_funding.csv", "timestamp", "funding_rate")
    oi = load_schedule(f"data_files/{symbol}_open_interest.csv", "timestamp", "sum_open_interest")
    ls = load_schedule(f"data_files/{symbol}_ls_ratio.csv", "timestamp", "long_short_ratio")
    spot = load_schedule(f"data_files/{symbol}_spot_1h.csv", "open_time", "close")
    fgi = load_schedule("data_files/fear_greed_index.csv", "timestamp", "value")
    liq_proxy = load_liquidation_proxy(f"data_files/{symbol}_liquidation_proxy.csv")

    all_schedules = [
        (sorted(funding.keys()), funding),
        (sorted(oi.keys()), oi),
        (sorted(ls.keys()), ls),
        (sorted(spot.keys()), spot),
        (sorted(fgi.keys()), fgi),
    ]
    idxs = [0] * len(all_schedules)
    liq_times = sorted(liq_proxy.keys())
    liq_idx = 0

    comp = EnrichedFeatureComputer()
    records = []
    timestamps = []

    for _, row in btc_df.iterrows():
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

        liq_val = None
        while liq_idx < len(liq_times) and liq_times[liq_idx] <= ts_ms:
            liq_val = liq_proxy[liq_times[liq_idx]]
            liq_idx += 1
        if liq_val is None and liq_idx > 0:
            liq_val = liq_proxy[liq_times[liq_idx - 1]]

        feats = comp.on_bar(
            symbol, close=close, volume=volume, high=high, low=low,
            open_=open_, hour=hour, dow=dow, funding_rate=fr,
            trades=trades, taker_buy_volume=tbv, quote_volume=qv,
            taker_buy_quote_volume=tbqv,
            open_interest=oi_val, ls_ratio=ls_val,
            spot_close=sp_val, fear_greed=fg_val,
            liquidation_metrics=liq_val,
        )
        feats["close"] = close
        records.append(feats)
        timestamps.append(ts_ms)

    result = pd.DataFrame(records)
    result["ts_ms"] = timestamps
    result.set_index("ts_ms", inplace=True)
    return result


def main():
    print("Loading BTC data...")
    btc_df = pd.read_csv("data_files/BTCUSDT_1h.csv")
    print(f"  BTC bars: {len(btc_df):,}")

    print("Loading SOL data...")
    sol_df = pd.read_csv("data_files/SOLUSDT_1h.csv")
    print(f"  SOL bars: {len(sol_df):,}")

    # Get SOL timestamps and forward returns
    sol_ts = []
    sol_closes = []
    for _, row in sol_df.iterrows():
        ts_raw = row.get("timestamp") or row.get("open_time", "")
        ts_ms = int(ts_raw) if ts_raw else 0
        sol_ts.append(ts_ms)
        sol_closes.append(float(row["close"]))

    sol_series = pd.Series(sol_closes, index=sol_ts, name="sol_close")
    sol_fwd = sol_series.shift(-5) / sol_series - 1.0
    sol_fwd.name = "sol_fwd_5"

    # Compute BTC features
    print("Computing BTC enriched features...")
    btc_feats = compute_btc_features(btc_df)
    print(f"  BTC features computed: {len(btc_feats):,} bars, {len(btc_feats.columns)} columns")

    # Align BTC features to SOL timestamps (inner join)
    common_ts = sorted(set(btc_feats.index) & set(sol_fwd.index))
    print(f"  Overlapping bars: {len(common_ts):,}")

    btc_aligned = btc_feats.loc[common_ts]
    sol_target = sol_fwd.loc[common_ts]

    # BTC-lead candidate features — BTC's own indicators that might predict SOL
    btc_lead_candidates = [
        # BTC returns at various horizons
        "ret_1", "ret_3", "ret_6", "ret_12", "ret_24",
        # BTC momentum/trend
        "rsi_14", "macd_line", "macd_hist",
        "sma_cross_20_50",
        # BTC volatility
        "atr_norm_14", "bb_width_20", "parkinson_vol",
        "vol_ma_ratio_5_20",
        # BTC volume/microstructure
        "volume_zscore_20", "taker_buy_ratio", "taker_imbalance",
        "trade_intensity",
        # BTC funding/basis
        "funding_rate", "funding_zscore_24", "funding_extreme",
        "basis", "basis_zscore_24", "basis_momentum",
        # BTC sentiment/regime
        "fgi_normalized", "fgi_extreme", "fgi_momentum",
        # BTC OI/LS
        "open_interest_zscore", "ls_ratio_zscore_24",
        "cvd_20", "cvd_momentum",
        # BTC mean-reversion
        "mean_reversion_20",
        # BTC liquidation
        "liquidation_cascade_score",
        # BTC time features
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        # BTC 4h trend
        "tf4h_close_vs_ma20", "tf4h_atr_norm_14",
        # BTC IV (if available)
        "implied_vol_zscore_24", "iv_rv_spread",
        # BTC mempool/macro
        "mempool_size_zscore_24", "spx_overnight_ret",
    ]

    # Filter to features that actually exist
    available = [f for f in btc_lead_candidates if f in btc_aligned.columns]
    missing = [f for f in btc_lead_candidates if f not in btc_aligned.columns]

    print(f"\n  Candidate features: {len(available)} available, {len(missing)} missing")
    if missing:
        print(f"  Missing: {', '.join(missing[:10])}{'...' if len(missing) > 10 else ''}")

    # IC analysis
    print()
    print("=" * 78)
    print("  BTC Feature → SOL 5-bar Forward Return IC Analysis (Spearman)")
    print("=" * 78)
    print(f"  Total overlapping bars: {len(common_ts):,}")
    print()
    print(f"  {'BTC Feature':<35} {'Non-null':>8} {'IC':>8} {'|IC|':>8} {'PASS':>6}")
    print(f"  {'-'*35} {'-'*8} {'-'*8} {'-'*8} {'-'*6}")

    results = []
    for feat_name in sorted(available):
        col = btc_aligned[feat_name].astype(float)
        valid = col.notna() & sol_target.notna()
        n_valid = int(valid.sum())
        if n_valid < 100:
            print(f"  {feat_name:<35} {n_valid:>8} {'N/A':>8} {'N/A':>8} {'SKIP':>6}")
            continue
        ic, pval = spearmanr(col[valid], sol_target[valid])
        abs_ic = abs(ic)
        ok = abs_ic > 0.02
        results.append((feat_name, ic, abs_ic, n_valid, ok))
        print(f"  {feat_name:<35} {n_valid:>8} {ic:>8.4f} {abs_ic:>8.4f} {'YES' if ok else 'NO':>6}")

    # Summary
    passed = [(f, ic) for f, ic, _, _, ok in results if ok]
    failed = [(f, ic) for f, ic, _, _, ok in results if not ok]

    print()
    print(f"  PASSED (|IC| > 0.02): {len(passed)}/{len(results)}")
    # Sort by |IC| descending
    passed.sort(key=lambda x: abs(x[1]), reverse=True)
    for f, ic in passed:
        print(f"    - btc_lead_{f}: IC={ic:.4f}")
    print()
    print(f"  FAILED: {len(failed)}/{len(results)}")
    for f, ic in failed:
        print(f"    - {f}: IC={ic:.4f}")
    print("=" * 78)

    # Also test existing cross-asset features that are already available
    # by computing them fresh
    print("\n\n--- Existing Cross-Asset Features (for reference) ---")
    from features.cross_asset_computer import CrossAssetComputer, CROSS_ASSET_FEATURE_NAMES

    cross_comp = CrossAssetComputer()
    cross_records = []
    cross_ts_list = []

    # Need both BTC and SOL aligned
    btc_data = {}
    btc_funding = load_schedule("data_files/BTCUSDT_funding.csv", "timestamp", "funding_rate")
    for _, row in btc_df.iterrows():
        ts_ms = int(row.get("timestamp") or row.get("open_time", 0))
        btc_data[ts_ms] = (float(row["close"]), btc_funding.get(ts_ms))

    sol_funding = load_schedule("data_files/SOLUSDT_funding.csv", "timestamp", "funding_rate")
    for _, row in sol_df.iterrows():
        ts_ms = int(row.get("timestamp") or row.get("open_time", 0))
        sol_close = float(row["close"])
        sol_fr = sol_funding.get(ts_ms)

        # Push BTC first (benchmark)
        btc_row = btc_data.get(ts_ms)
        if btc_row is not None:
            cross_comp.on_bar("BTCUSDT", close=btc_row[0], funding_rate=btc_row[1])
        # Then SOL
        cross_comp.on_bar("SOLUSDT", close=sol_close, funding_rate=sol_fr)

        feats = cross_comp.get_features("SOLUSDT")
        cross_records.append(feats)
        cross_ts_list.append(ts_ms)

    cross_df = pd.DataFrame(cross_records, index=cross_ts_list)
    cross_target = sol_fwd.reindex(cross_ts_list)

    print(f"\n  {'Cross-Asset Feature':<35} {'Non-null':>8} {'IC':>8} {'|IC|':>8} {'PASS':>6}")
    print(f"  {'-'*35} {'-'*8} {'-'*8} {'-'*8} {'-'*6}")
    for feat_name in CROSS_ASSET_FEATURE_NAMES:
        col = cross_df[feat_name].astype(float)
        valid = col.notna() & cross_target.notna()
        n_valid = int(valid.sum())
        if n_valid < 100:
            print(f"  {feat_name:<35} {n_valid:>8} {'N/A':>8} {'N/A':>8} {'SKIP':>6}")
            continue
        ic, _ = spearmanr(col[valid], cross_target[valid])
        abs_ic = abs(ic)
        ok = abs_ic > 0.02
        print(f"  {feat_name:<35} {n_valid:>8} {ic:>8.4f} {abs_ic:>8.4f} {'YES' if ok else 'NO':>6}")

    print("=" * 78)
    return passed


if __name__ == "__main__":
    main()
