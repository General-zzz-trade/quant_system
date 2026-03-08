#!/usr/bin/env python3
"""Screen all available features by IC and IC_IR to find unused high-value features."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def _compute_ic_stats(feature: np.ndarray, target: np.ndarray, window: int = 720):
    """Compute rank IC and rolling IC_IR (mean/std of rolling IC)."""
    valid = ~(np.isnan(feature) | np.isnan(target))
    f, t = feature[valid], target[valid]
    if len(f) < 100:
        return np.nan, np.nan
    ic, _ = spearmanr(f, t)
    # Rolling IC
    n = len(f)
    ics = []
    for start in range(0, n - window, window // 2):
        end = start + window
        if end > n:
            break
        r, _ = spearmanr(f[start:end], t[start:end])
        if not np.isnan(r):
            ics.append(r)
    if len(ics) < 3:
        return ic, np.nan
    ic_ir = np.mean(ics) / (np.std(ics, ddof=1) + 1e-10)
    return ic, ic_ir


def screen_symbol(symbol: str, horizon: int, used_features: set):
    """Screen all features for a symbol."""
    csv_path = Path(f"data_files/{symbol}_1h.csv")
    if not csv_path.exists():
        print(f"  [{symbol}] No data file")
        return []

    df = pd.read_csv(csv_path)

    # Compute enriched features (include_v11 for macro/mempool/liquidation)
    from features.batch_feature_engine import compute_features_batch
    feat_df = compute_features_batch(symbol, df, include_v11=True)

    # Cross-asset features
    if symbol != "BTCUSDT":
        try:
            from features.batch_cross_asset import build_cross_features_batch
            cross_map = build_cross_features_batch([symbol])
            if cross_map and symbol in cross_map:
                ts_col = "timestamp" if "timestamp" in df.columns else "open_time"
                oos_ts = df[ts_col].values.astype(np.int64)
                cross_df = cross_map[symbol]
                cross_aligned = cross_df.reindex(oos_ts)
                for col in cross_aligned.columns:
                    if col not in feat_df.columns:
                        feat_df[col] = cross_aligned[col].values
        except Exception as e:
            print(f"  [{symbol}] Cross-asset error: {e}")

    # Forward return target
    closes = feat_df["close"].values.astype(np.float64)
    fwd_ret = np.full(len(closes), np.nan)
    fwd_ret[:-horizon] = (closes[horizon:] - closes[:-horizon]) / closes[:-horizon]

    # Skip non-feature columns
    skip = {"open", "high", "low", "close", "volume", "timestamp", "open_time",
            "close_time", "quote_volume", "count", "taker_buy_volume",
            "taker_buy_quote_volume", "ignore"}

    results = []
    for col in feat_df.columns:
        if col in skip:
            continue
        vals = feat_df[col].values.astype(np.float64)
        ic, ic_ir = _compute_ic_stats(vals[65:], fwd_ret[65:])  # skip warmup
        if np.isnan(ic):
            continue
        results.append({
            "feature": col,
            "ic": ic,
            "abs_ic": abs(ic),
            "ic_ir": ic_ir if not np.isnan(ic_ir) else 0.0,
            "used": col in used_features,
        })

    results.sort(key=lambda x: -x["abs_ic"])
    return results


def main():
    out_dir = Path("results/feature_screen")
    out_dir.mkdir(parents=True, exist_ok=True)

    symbols_config = {
        "BTCUSDT": (24, "models_v8/BTCUSDT_gate_v2/config.json"),
        "ETHUSDT": (24, "models_v8/ETHUSDT_gate_v2/config.json"),
        "SOLUSDT": (5,  "models_v8/SOLUSDT_gate_v2/config.json"),
    }

    for symbol, (horizon, cfg_path) in symbols_config.items():
        print(f"\n{'='*60}")
        print(f"  {symbol} (horizon={horizon}h)")
        print(f"{'='*60}")

        with open(cfg_path) as f:
            cfg = json.load(f)
        used = set(cfg["features"])

        results = screen_symbol(symbol, horizon, used)
        if not results:
            continue

        # Save full results
        pd.DataFrame(results).to_csv(out_dir / f"{symbol}_ic.csv", index=False)

        # Print top 30 unused
        unused = [r for r in results if not r["used"]]
        print(f"\n  Top 20 UNUSED features (of {len(unused)} total unused):")
        print(f"  {'Feature':<30} {'IC':>8} {'|IC|':>8} {'IC_IR':>8}")
        print(f"  {'-'*58}")
        for r in unused[:20]:
            print(f"  {r['feature']:<30} {r['ic']:>+8.4f} {r['abs_ic']:>8.4f} {r['ic_ir']:>8.3f}")

        # Also show used features for comparison
        used_list = [r for r in results if r["used"]]
        print(f"\n  Currently used features ({len(used_list)}):")
        print(f"  {'Feature':<30} {'IC':>8} {'|IC|':>8} {'IC_IR':>8}")
        print(f"  {'-'*58}")
        for r in used_list:
            print(f"  {r['feature']:<30} {r['ic']:>+8.4f} {r['abs_ic']:>8.4f} {r['ic_ir']:>8.3f}")


if __name__ == "__main__":
    main()
