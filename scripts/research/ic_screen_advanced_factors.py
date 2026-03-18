#!/usr/bin/env python3
"""IC screening for advanced alpha factors: IV, liquidation, orderbook proxy.

Uses batch_feature_engine with IV and liquidation data to compute features,
then screens IC across all symbols and horizons.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, "/quant_system")

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SUIUSDT", "AXSUSDT"]
HORIZONS = [1, 3, 6, 12, 24]


def load_klines(symbol: str) -> pd.DataFrame:
    df = pd.read_csv(f"data_files/{symbol}_1h.csv")
    if "open_time" in df.columns:
        df = df.sort_values("open_time").reset_index(drop=True)
    for c in ["close", "open", "high", "low", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_iv_data(symbol: str, kline_df: pd.DataFrame) -> pd.DataFrame:
    """Load Deribit IV and align to kline timestamps."""
    from pathlib import Path
    iv_path = Path(f"data_files/{symbol}_deribit_iv.csv")
    if not iv_path.exists():
        return pd.DataFrame()
    iv = pd.read_csv(iv_path)
    if "timestamp" not in iv.columns:
        return pd.DataFrame()
    # Merge on nearest timestamp
    # Parse timestamp: may be ISO string or numeric
    try:
        iv["ts_ms"] = pd.to_datetime(iv["timestamp"]).astype(np.int64) // 10**6
    except Exception:
        iv["ts_ms"] = pd.to_numeric(iv["timestamp"], errors="coerce")
    iv = iv.dropna(subset=["ts_ms"])
    if iv.empty:
        return pd.DataFrame()
    # Create features from IV
    result = pd.DataFrame(index=kline_df.index)
    if "implied_vol" in iv.columns:
        iv_sorted = iv.sort_values("ts_ms")
        ts_kline = kline_df["open_time"].values
        iv_vals = np.interp(ts_kline, iv_sorted["ts_ms"].values,
                           iv_sorted["implied_vol"].values)
        result["iv_raw"] = iv_vals
        # 24-bar z-score
        result["iv_zscore_24"] = _rolling_zscore(iv_vals, 24)
        # IV - realized vol spread
        closes = kline_df["close"].values
        rets = np.diff(np.log(closes), prepend=np.nan)
        rv_20 = pd.Series(rets).rolling(20).std().values
        result["iv_rv_spread"] = iv_vals - rv_20
        # IV momentum
        result["iv_momentum_12"] = pd.Series(iv_vals).diff(12).values
        # IV acceleration
        result["iv_accel_6"] = pd.Series(iv_vals).diff(6).diff(6).values
    if "put_call_ratio" in iv.columns:
        pcr_vals = np.interp(ts_kline, iv_sorted["ts_ms"].values,
                            iv_sorted["put_call_ratio"].values)
        result["pcr_raw"] = pcr_vals
        result["pcr_zscore_24"] = _rolling_zscore(pcr_vals, 24)
        result["pcr_momentum_12"] = pd.Series(pcr_vals).diff(12).values
    return result


def load_liq_data(symbol: str, kline_df: pd.DataFrame) -> pd.DataFrame:
    """Load liquidation proxy and create features."""
    from pathlib import Path
    liq_path = Path(f"data_files/{symbol}_liquidation_proxy.csv")
    if not liq_path.exists():
        return pd.DataFrame()
    liq = pd.read_csv(liq_path)
    result = pd.DataFrame(index=kline_df.index)
    n = min(len(liq), len(kline_df))
    offset = len(kline_df) - n

    if "liq_proxy_volume" in liq.columns:
        vol = liq["liq_proxy_volume"].values[-n:]
        padded = np.concatenate([np.full(offset, np.nan), vol])
        result["liq_volume"] = padded
        result["liq_volume_zscore_24"] = _rolling_zscore(padded, 24)
        # Log volume (handles zeros)
        result["liq_log_volume"] = np.log1p(np.maximum(padded, 0))

    if "liq_proxy_imbalance" in liq.columns:
        imb = liq["liq_proxy_imbalance"].values[-n:]
        padded = np.concatenate([np.full(offset, np.nan), imb])
        result["liq_imbalance"] = padded
        result["liq_imbalance_zscore_12"] = _rolling_zscore(padded, 12)

    if "liq_proxy_cluster" in liq.columns:
        cl = liq["liq_proxy_cluster"].values[-n:]
        padded = np.concatenate([np.full(offset, np.nan), cl])
        result["liq_cluster"] = padded
        # Rolling cluster count (how many cluster events in last 12 bars)
        result["liq_cluster_count_12"] = pd.Series(padded).rolling(12).sum().values

    # Interaction: liquidation × volatility
    if "liq_volume" in result.columns:
        closes = kline_df["close"].values
        rets = np.diff(np.log(closes), prepend=np.nan)
        rv_20 = pd.Series(rets).rolling(20).std().values
        result["liq_x_vol"] = result["liq_volume"].values * rv_20

    return result


def create_orderbook_proxy_features(kline_df: pd.DataFrame) -> pd.DataFrame:
    """Create orderbook-like features from kline microstructure data.

    Since we don't have historical depth data, use proxies:
    - Taker buy ratio as bid/ask pressure proxy
    - Volume per trade as depth proxy
    - Spread proxy from high-low range
    """
    result = pd.DataFrame(index=kline_df.index)
    closes = kline_df["close"].values
    highs = kline_df["high"].values
    lows = kline_df["low"].values
    volumes = kline_df["volume"].values if "volume" in kline_df else np.ones(len(closes))

    # 1. Spread proxy: (high - low) / close — intrabar range as spread measure
    spread_proxy = (highs - lows) / np.maximum(closes, 1e-10)
    result["ob_spread_proxy"] = spread_proxy
    result["ob_spread_zscore_20"] = _rolling_zscore(spread_proxy, 20)

    # 2. Taker pressure accumulation (CVD-like but normalized)
    if "taker_buy_volume" in kline_df.columns:
        tbv = kline_df["taker_buy_volume"].values.astype(float)
        tsv = volumes - tbv  # taker sell volume
        # Normalized imbalance
        total = tbv + tsv
        imb = np.where(total > 0, (tbv - tsv) / total, 0.0)
        result["ob_imbalance_proxy"] = imb
        result["ob_imbalance_ma10"] = pd.Series(imb).rolling(10).mean().values
        # Cumulative 6-bar imbalance (short-term order flow)
        result["ob_imbalance_cum6"] = pd.Series(imb).rolling(6).sum().values
        # Imbalance × volume (strength-weighted)
        vol_norm = volumes / pd.Series(volumes).rolling(20).mean().values
        result["ob_imbalance_x_vol"] = imb * vol_norm

    # 3. Trade size anomaly (large trades = institutional flow)
    if "trades" in kline_df.columns and "quote_volume" in kline_df.columns:
        trades = kline_df["trades"].values.astype(float)
        qv = kline_df["quote_volume"].values.astype(float)
        avg_size = np.where(trades > 0, qv / trades, 0.0)
        avg_size_ma = pd.Series(avg_size).rolling(20).mean().values
        result["ob_trade_size_anomaly"] = np.where(
            avg_size_ma > 0, avg_size / avg_size_ma - 1.0, 0.0
        )

    # 4. Volume clock (acceleration of trading activity)
    result["ob_volume_clock"] = pd.Series(volumes).rolling(6).mean().values / \
        np.maximum(pd.Series(volumes).rolling(24).mean().values, 1e-10) - 1.0

    return result


def _rolling_zscore(arr: np.ndarray, window: int) -> np.ndarray:
    s = pd.Series(arr)
    mu = s.rolling(window).mean()
    std = s.rolling(window).std()
    return ((s - mu) / std.replace(0, np.nan)).values


def compute_ic(feat_df: pd.DataFrame, closes: np.ndarray) -> pd.DataFrame:
    results = []
    n = len(closes)
    for h in HORIZONS:
        if n <= h + 100:
            continue
        fwd = (closes[h:] - closes[:-h]) / closes[:-h]
        for col in feat_df.columns:
            vals = pd.to_numeric(feat_df[col], errors="coerce").values[:-h]
            mask = np.isfinite(vals) & np.isfinite(fwd)
            if mask.sum() < 200:
                continue
            ic, pval = spearmanr(vals[mask], fwd[mask])
            if np.isfinite(ic):
                results.append({
                    "feature": col, "horizon": h, "ic": ic,
                    "pval": pval, "abs_ic": abs(ic), "n": int(mask.sum()),
                    "sig": pval < 0.01 and abs(ic) > 0.02,
                })
    return pd.DataFrame(results)


def main():
    print("=" * 80)
    print("  ADVANCED FACTOR IC SCREEN: IV + Liquidation + Orderbook Proxy")
    print("=" * 80)

    all_ics = []

    for symbol in SYMBOLS:
        print(f"\n{'─'*80}")
        print(f"  {symbol}")
        print(f"{'─'*80}")

        df = load_klines(symbol)
        closes = df["close"].values
        print(f"  Bars: {len(df)}")

        # Collect all advanced features
        feat_frames = []

        # IV features
        iv_feat = load_iv_data(symbol, df)
        if not iv_feat.empty:
            feat_frames.append(iv_feat)
            print(f"  IV features: {len(iv_feat.columns)} ({list(iv_feat.columns)})")
        else:
            print("  IV features: NO DATA")

        # Liquidation features
        liq_feat = load_liq_data(symbol, df)
        if not liq_feat.empty:
            feat_frames.append(liq_feat)
            print(f"  Liq features: {len(liq_feat.columns)} ({list(liq_feat.columns)})")
        else:
            print("  Liq features: NO DATA")

        # Orderbook proxy features
        ob_feat = create_orderbook_proxy_features(df)
        if not ob_feat.empty:
            feat_frames.append(ob_feat)
            print(f"  OB proxy features: {len(ob_feat.columns)} ({list(ob_feat.columns)})")

        if not feat_frames:
            print("  No features to screen!")
            continue

        feat_df = pd.concat(feat_frames, axis=1)
        print(f"  Total advanced features: {len(feat_df.columns)}")

        # IC screening
        ic_df = compute_ic(feat_df, closes)
        if ic_df.empty:
            continue

        ic_df["symbol"] = symbol
        all_ics.append(ic_df)

        # Print significant features
        sig = ic_df[ic_df["sig"]].sort_values("abs_ic", ascending=False)
        if not sig.empty:
            print("\n  SIGNIFICANT (p<0.01, |IC|>0.02):")
            print(f"  {'Feature':<30} {'H':>3} {'IC':>8} {'p-val':>8}")
            print(f"  {'-'*52}")
            for _, r in sig.head(15).iterrows():
                print(f"  {r['feature']:<30} {r['horizon']:>3} {r['ic']:>+8.4f} {r['pval']:>8.4f}")

    # Cross-symbol summary
    if all_ics:
        all_df = pd.concat(all_ics)
        print(f"\n{'='*80}")
        print("  CROSS-SYMBOL SUMMARY")
        print(f"{'='*80}")

        # Features significant in >= 2 symbols
        sig_all = all_df[all_df["sig"]]
        if not sig_all.empty:
            feat_counts = sig_all.groupby("feature").agg(
                n_symbols=("symbol", "nunique"),
                avg_ic=("ic", "mean"),
                max_abs_ic=("abs_ic", "max"),
                best_h=("abs_ic", "idxmax"),
            ).sort_values("n_symbols", ascending=False)

            print(f"\n  {'Feature':<30} {'#Sym':>5} {'AvgIC':>8} {'Max|IC|':>8}")
            print(f"  {'-'*54}")
            for feat, row in feat_counts.iterrows():
                marker = "★★" if row["n_symbols"] >= 3 else ("★" if row["n_symbols"] >= 2 else " ")
                print(f"  {feat:<30} {row['n_symbols']:>5} {row['avg_ic']:>+8.4f} "
                      f"{row['max_abs_ic']:>8.4f} {marker}")


if __name__ == "__main__":
    main()
