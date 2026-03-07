#!/usr/bin/env python3
"""IC analysis for V12 untested features — Spearman IC vs 24-bar forward return.

Tests V11 features that were implemented but never IC-validated:
  - liquidation_volume_zscore_24, liquidation_imbalance, liquidation_volume_ratio
  - fee_urgency_ratio (mempool)
  - cross_tf_regime_sync (V9)
  - social_volume_zscore_24 (V11, if data exists)
  - liquidation_cluster_flag, mempool_fee_zscore_24, mempool_size_zscore_24
  - spx_btc_corr_30d, dxy_change_5d, vix_zscore_14

Usage:
    python3 scripts/ic_analysis_v12_untested.py --symbol BTCUSDT
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from features.batch_feature_engine import compute_features_batch


def main() -> None:
    parser = argparse.ArgumentParser(description="V12 IC analysis for untested features")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--horizon", type=int, default=24, help="Forward return horizon (bars)")
    parser.add_argument("--threshold", type=float, default=0.02, help="|IC| pass threshold")
    args = parser.parse_args()

    symbol = args.symbol.upper()
    horizon = args.horizon
    threshold = args.threshold

    csv_path = Path(f"data_files/{symbol}_1h.csv")
    if not csv_path.exists():
        print(f"  Data not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    print(f"\n  V12 IC Analysis: {symbol}")
    print(f"  Total bars: {len(df):,}")
    print(f"  Forward return horizon: {horizon} bars")

    print("  Computing features (include_v11=True, include_iv=True, include_onchain=True)...")
    t0 = time.time()
    feat_df = compute_features_batch(
        symbol, df, include_iv=True, include_onchain=True, include_v11=True)
    print(f"  Features computed in {time.time()-t0:.1f}s ({len(feat_df.columns)} columns)")

    # Forward return target
    closes = feat_df["close"].values.astype(np.float64)
    target = pd.Series(np.roll(closes, -horizon) / closes - 1.0, index=feat_df.index)
    # Invalidate last `horizon` bars (rolled-over values)
    target.iloc[-horizon:] = np.nan

    # V12 untested features
    v12_features = [
        # V11 Liquidation
        "liquidation_volume_zscore_24",
        "liquidation_imbalance",
        "liquidation_volume_ratio",
        "liquidation_cluster_flag",
        # V11 Mempool
        "fee_urgency_ratio",
        "mempool_fee_zscore_24",
        "mempool_size_zscore_24",
        # V9 Cross-TF
        "cross_tf_regime_sync",
        # V11 Macro (some already tested, include for completeness)
        "spx_btc_corr_30d",
        "dxy_change_5d",
        "vix_zscore_14",
        # V11 Social (if data exists)
        "social_volume_zscore_24",
        "social_sentiment_score",
        "social_volume_price_div",
        # V10 On-chain (some untested)
        "exchange_netflow_zscore",
        "exchange_supply_change",
        "active_addr_zscore_14",
        "tx_count_zscore_14",
        "hashrate_momentum",
    ]

    print()
    print("=" * 78)
    print(f"  V12 Feature IC Analysis (Spearman, {symbol} {horizon}-bar forward return)")
    print("=" * 78)
    print(f"  {'Feature':<35} {'Non-null':>8} {'IC':>8} {'|IC|':>8} {'PASS':>6}")
    print(f"  {'-'*35} {'-'*8} {'-'*8} {'-'*8} {'-'*6}")

    passed = []
    for feat_name in v12_features:
        if feat_name not in feat_df.columns:
            print(f"  {feat_name:<35} {'MISSING':>8}")
            continue
        col = feat_df[feat_name].astype(float)
        valid = col.notna() & target.notna() & (col != 0)
        n_valid = int(valid.sum())
        if n_valid < 200:
            print(f"  {feat_name:<35} {n_valid:>8} {'N/A':>8} {'N/A':>8} {'SKIP':>6}")
            continue
        ic, pval = spearmanr(col[valid], target[valid])
        abs_ic = abs(ic)
        ok = abs_ic > threshold
        if ok:
            passed.append((feat_name, ic, pval))
        print(f"  {feat_name:<35} {n_valid:>8} {ic:>8.4f} {abs_ic:>8.4f} {'YES' if ok else 'NO':>6}")

    print()
    print(f"  Passed features (|IC| > {threshold}): {len(passed)}/{len(v12_features)}")
    for f, ic, pval in passed:
        print(f"    - {f} (IC={ic:.4f}, p={pval:.4f})")

    # Reference ICs for context
    print()
    print("  Reference IC (existing features):")
    ref_feats = ["basis", "funding_zscore_24", "rsi_14", "atr_norm_14", "cvd_20",
                 "parkinson_vol", "implied_vol_zscore_24", "spx_overnight_ret"]
    for feat_name in ref_feats:
        if feat_name not in feat_df.columns:
            continue
        col = feat_df[feat_name].astype(float)
        valid = col.notna() & target.notna() & (col != 0)
        n_valid = int(valid.sum())
        if n_valid < 200:
            continue
        ic, _ = spearmanr(col[valid], target[valid])
        print(f"    {feat_name:<35} IC={ic:>8.4f}")
    print("=" * 78)

    return passed


if __name__ == "__main__":
    main()
