#!/usr/bin/env python3
"""Bear IC analysis for short model feature candidates.

Measures Spearman IC on negative-return bars only (bear IC) and all bars,
across multiple horizons. Reports features ranked by bear IC strength.

Usage:
    python3 -m scripts.ic_analysis_short_features --symbol BTCUSDT
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from features.dynamic_selector import _rankdata, _spearman_ic
from scripts.signal_postprocess import _compute_bear_mask as _shared_compute_bear_mask
from scripts.train_v7_alpha import _load_and_compute_features, _compute_target, BLACKLIST


MA_WINDOW = 480
WARMUP = 65
HORIZONS = [5, 12, 24]


def _compute_bear_mask(closes: np.ndarray, ma_window: int = MA_WINDOW) -> np.ndarray:
    mask = _shared_compute_bear_mask(closes, ma_window).copy()
    if len(closes) < ma_window:
        return np.zeros(len(closes), dtype=bool)
    mask[: ma_window - 1] = False
    return mask


def _ic(x: np.ndarray, y: np.ndarray) -> float:
    valid = ~(np.isnan(x) | np.isnan(y))
    if valid.sum() < 20:
        return float("nan")
    xv, yv = x[valid], y[valid]
    if np.std(xv) < 1e-12 or np.std(yv) < 1e-12:
        return float("nan")
    return float(_spearman_ic(_rankdata(xv), _rankdata(yv)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BTCUSDT")
    args = parser.parse_args()
    symbol = args.symbol.upper()

    csv_path = Path(f"data_files/{symbol}_1h.csv")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df):,} bars for {symbol}")

    print("Computing features...")
    feat_df = _load_and_compute_features(symbol, df)
    closes = feat_df["close"].values.astype(np.float64)

    bear_mask = _compute_bear_mask(closes)
    neg_ret_24 = _compute_target(closes, 24, "clipped")
    neg_mask = np.zeros(len(closes), dtype=bool)
    neg_mask[WARMUP:] = True
    valid_24 = ~np.isnan(neg_ret_24)
    neg_ret_bars = valid_24 & (neg_ret_24 < 0) & neg_mask

    n_bear = int(bear_mask[WARMUP:].sum())
    n_neg = int(neg_ret_bars.sum())
    print(f"Bear regime bars: {n_bear:,} / {len(closes) - WARMUP:,}")
    print(f"Negative 24h return bars: {n_neg:,}")

    # Candidate features to test (NOT currently in short model fixed features)
    # Current fixed: funding_zscore_24, funding_momentum, basis, basis_zscore_24,
    #   basis_momentum, parkinson_vol, atr_norm_14, leverage_proxy, oi_acceleration, fgi_normalized
    # Current candidates: funding_sign_persist, funding_extreme, funding_cumulative_8,
    #   vol_regime, vol_20, rsi_14, bb_pctb_20, liquidation_cascade_score,
    #   implied_vol_zscore_24, iv_rv_spread, cvd_20, vol_of_vol
    candidates = [
        # Liquidation (V11)
        "liquidation_volume_zscore_24", "liquidation_imbalance",
        "liquidation_volume_ratio", "liquidation_cluster_flag",
        # IV/Options
        "put_call_ratio", "implied_vol_zscore_24", "iv_rv_spread",
        # On-chain
        "exchange_netflow_zscore", "exchange_supply_change",
        "exchange_supply_zscore_30", "active_addr_zscore_14",
        # Macro
        "dxy_change_5d", "spx_btc_corr_30d", "spx_overnight_ret", "vix_zscore_14",
        # Mempool
        "mempool_fee_zscore_24", "mempool_size_zscore_24", "fee_urgency_ratio",
        # Order flow
        "aggressive_flow_zscore", "cvd_price_divergence",
        # Cross-TF / interaction
        "cross_tf_regime_sync", "funding_term_slope",
        "liquidation_cascade_score",
        # Volatility
        "vol_of_vol", "mom_vol_divergence", "rv_acceleration", "range_vs_rv",
        # Current model features (for comparison baseline)
        "funding_zscore_24", "basis", "parkinson_vol", "fgi_normalized",
        "leverage_proxy", "oi_acceleration",
    ]

    # Also test current fixed + candidate pool features
    all_features = [f for f in candidates if f in feat_df.columns and f not in BLACKLIST]
    missing = [f for f in candidates if f not in feat_df.columns]
    if missing:
        print(f"\nMissing features ({len(missing)}): {missing}")

    print(f"\nTesting {len(all_features)} features across {len(HORIZONS)} horizons")
    print(f"{'='*100}")

    results = []
    for feat_name in all_features:
        x = feat_df[feat_name].values.astype(np.float64)
        row = {"feature": feat_name}

        for h in HORIZONS:
            y = _compute_target(closes, h, "clipped")

            # All-bar IC
            ic_all = _ic(x[WARMUP:], y[WARMUP:])

            # Bear regime IC
            bear_x = x[bear_mask & neg_mask]
            bear_y = y[bear_mask & neg_mask]
            ic_bear = _ic(bear_x, bear_y)

            # Negative return IC (only bars where 24h fwd return < 0)
            if h == 24:
                neg_x = x[neg_ret_bars]
                neg_y = neg_ret_24[neg_ret_bars]
                ic_neg = _ic(neg_x, neg_y)
            else:
                neg_h = _compute_target(closes, h, "clipped")
                neg_h_mask = valid_24 & (neg_ret_24 < 0) & neg_mask  # use 24h as regime
                neg_x = x[neg_h_mask]
                neg_y = neg_h[neg_h_mask]
                ic_neg = _ic(neg_x, neg_y)

            row[f"ic_all_{h}"] = ic_all
            row[f"ic_bear_{h}"] = ic_bear
            row[f"ic_neg_{h}"] = ic_neg

        results.append(row)

    df_results = pd.DataFrame(results)

    # Sort by bear IC at 24h horizon (primary metric for short model)
    df_results["abs_ic_bear_24"] = df_results["ic_bear_24"].abs()
    df_results = df_results.sort_values("abs_ic_bear_24", ascending=False)

    print(f"\n{'Feature':<35} {'IC_all_24':>10} {'IC_bear_24':>11} {'IC_neg_24':>10} "
          f"{'IC_all_5':>9} {'IC_bear_5':>10}")
    print("-" * 100)

    for _, r in df_results.iterrows():
        mark = ""
        # Mark features NOT in current short model
        current = {"funding_zscore_24", "funding_momentum", "basis", "basis_zscore_24",
                    "basis_momentum", "parkinson_vol", "atr_norm_14", "leverage_proxy",
                    "oi_acceleration", "fgi_normalized", "funding_sign_persist",
                    "funding_extreme", "funding_cumulative_8", "vol_regime", "vol_20",
                    "rsi_14", "bb_pctb_20", "liquidation_cascade_score",
                    "implied_vol_zscore_24", "iv_rv_spread", "cvd_20", "vol_of_vol"}
        if r["feature"] not in current:
            mark = " ***NEW"

        ic_all_24 = r["ic_all_24"]
        ic_bear_24 = r["ic_bear_24"]
        ic_neg_24 = r["ic_neg_24"]
        ic_all_5 = r["ic_all_5"]
        ic_bear_5 = r["ic_bear_5"]

        print(f"  {r['feature']:<33} {ic_all_24:>+.4f}   {ic_bear_24:>+.4f}    "
              f"{ic_neg_24:>+.4f}   {ic_all_5:>+.4f}   {ic_bear_5:>+.4f}{mark}")

    # Summary: top 10 NEW features by bear IC
    print(f"\n{'='*60}")
    print("TOP NEW FEATURES FOR SHORT MODEL (by |bear IC @ 24h|)")
    print(f"{'='*60}")
    new_only = df_results[~df_results["feature"].isin(current)]
    for i, (_, r) in enumerate(new_only.head(10).iterrows()):
        print(f"  {i+1:2d}. {r['feature']:<35} bear_IC={r['ic_bear_24']:+.4f}  "
              f"neg_IC={r['ic_neg_24']:+.4f}  all_IC={r['ic_all_24']:+.4f}")


if __name__ == "__main__":
    main()
