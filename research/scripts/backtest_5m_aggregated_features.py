#!/usr/bin/env python3
# ruff: noqa: E402,E501,E701,E702,E741,F841,E401
"""Research: Aggregate 5m microstructure into 1h features for model training.

Step 1: Compute 5m-level microstructure features
Step 2: Aggregate to 1h granularity (using 12 x 5m bars per 1h bar)
Step 3: IC scan against 1h forward returns
Step 4: Compare with existing 1h features
Step 5: Walk-forward validation with augmented feature set
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

DATA_DIR = Path("data_files")
SYMBOLS = ["BTCUSDT", "ETHUSDT"]


def load_5m(symbol: str) -> pd.DataFrame:
    path = DATA_DIR / f"{symbol}_5m.csv"
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["open_time"], unit="ms")
    df = df.sort_values("open_time").reset_index(drop=True)
    df["taker_sell_volume"] = df["volume"] - df["taker_buy_volume"]
    df["taker_sell_quote"] = df["quote_volume"] - df["taker_buy_quote_volume"]
    df["ret"] = df["close"].pct_change()
    df["log_ret"] = np.log(df["close"] / df["close"].shift(1))
    return df


def load_1h(symbol: str) -> pd.DataFrame:
    path = DATA_DIR / f"{symbol}_1h.csv"
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["open_time"], unit="ms")
    df = df.sort_values("open_time").reset_index(drop=True)
    df["ret"] = df["close"].pct_change()
    return df


def compute_5m_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Compute raw 5m microstructure metrics (before aggregation)."""
    out = pd.DataFrame(index=df.index)
    out["datetime"] = df["datetime"]
    out["hour_key"] = df["datetime"].dt.floor("h")

    vol = df["volume"]
    tbv = df["taker_buy_volume"]
    tsv = df["taker_sell_volume"]
    qv = df["quote_volume"]
    trades = df["trades"]
    close = df["close"]
    high = df["high"]
    low = df["low"]
    ret = df["ret"]

    # --- Taker flow ---
    tbr = tbv / vol.replace(0, np.nan)
    out["taker_buy_ratio"] = tbr
    out["taker_imbalance"] = 2 * tbr - 1  # -1 to +1
    out["volume_delta"] = tbv - tsv  # raw CVD per bar
    out["volume_delta_abs"] = (tbv - tsv).abs()

    # --- Trade intensity ---
    out["trades"] = trades
    out["avg_trade_size"] = qv / trades.replace(0, np.nan)
    out["volume"] = vol
    out["quote_volume"] = qv

    # --- Price micro ---
    out["bar_range"] = (high - low) / close  # intra-bar volatility
    out["body_ratio"] = (close - df["open"]).abs() / (high - low).replace(0, np.nan)
    out["upper_wick"] = (high - df[["close", "open"]].max(axis=1)) / (high - low).replace(0, np.nan)
    out["lower_wick"] = (df[["close", "open"]].min(axis=1) - low) / (high - low).replace(0, np.nan)
    out["ret"] = ret
    out["ret_sq"] = ret ** 2  # for realized variance
    out["log_ret"] = df["log_ret"]
    out["log_ret_sq"] = df["log_ret"] ** 2

    # --- Signed volume (for OFI proxy) ---
    out["signed_volume"] = np.sign(ret) * vol
    out["signed_quote_vol"] = np.sign(ret) * qv

    # --- Large trade flag: avg_trade_size > 2x rolling median ---
    median_size = out["avg_trade_size"].rolling(288).median()  # 1-day rolling
    out["large_trade_flag"] = (out["avg_trade_size"] > 2 * median_size).astype(float)
    out["large_trade_volume"] = out["large_trade_flag"] * vol

    return out


def aggregate_to_1h(raw_5m: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 5m metrics to 1h features using diverse aggregation functions."""
    grouped = raw_5m.groupby("hour_key")

    feats = pd.DataFrame()

    # --- CATEGORY 1: Taker Flow Distribution (within the hour) ---
    # Sum of taker imbalance (net buy pressure over the hour)
    feats["micro_taker_imb_sum"] = grouped["taker_imbalance"].sum()
    # Mean taker buy ratio
    feats["micro_taker_br_mean"] = grouped["taker_buy_ratio"].mean()
    # Std of taker buy ratio (flow consistency)
    feats["micro_taker_br_std"] = grouped["taker_buy_ratio"].std()
    # Skewness of taker imbalance (asymmetric buying/selling)
    feats["micro_taker_imb_skew"] = grouped["taker_imbalance"].apply(
        lambda x: x.skew() if len(x) >= 3 else 0)
    # Max absolute taker imbalance (spike detection)
    feats["micro_taker_imb_max"] = grouped["taker_imbalance"].apply(lambda x: x.abs().max())
    # Fraction of bars with buy pressure > 0.6
    feats["micro_buy_pressure_pct"] = grouped["taker_buy_ratio"].apply(
        lambda x: (x > 0.6).mean() if len(x) > 0 else 0.5)
    # Last bar taker imbalance vs hour mean (momentum within hour)
    feats["micro_taker_last_vs_mean"] = grouped["taker_imbalance"].apply(
        lambda x: x.iloc[-1] - x.mean() if len(x) > 0 else 0)

    # --- CATEGORY 2: CVD (Cumulative Volume Delta) ---
    feats["micro_cvd_1h"] = grouped["volume_delta"].sum()
    # CVD normalized by total volume
    total_vol = grouped["volume"].sum()
    feats["micro_cvd_norm"] = feats["micro_cvd_1h"] / total_vol.replace(0, np.nan)
    # CVD path: did CVD accumulate steadily or spike?
    feats["micro_cvd_path_std"] = grouped["volume_delta"].apply(
        lambda x: x.cumsum().std() if len(x) >= 3 else 0)
    # Max CVD excursion within the hour
    feats["micro_cvd_max_excursion"] = grouped["volume_delta"].apply(
        lambda x: x.cumsum().abs().max() if len(x) > 0 else 0)

    # --- CATEGORY 3: VPIN proxy (informed trading) ---
    # VPIN = sum(|buy_vol - sell_vol|) / sum(total_vol)
    feats["micro_vpin"] = grouped["volume_delta_abs"].sum() / total_vol.replace(0, np.nan)

    # --- CATEGORY 4: Trade Intensity Distribution ---
    feats["micro_trades_total"] = grouped["trades"].sum()
    feats["micro_trades_std"] = grouped["trades"].std()
    # Trade intensity trend: last 3 bars vs first 3 bars
    feats["micro_trade_trend"] = grouped["trades"].apply(
        lambda x: x.iloc[-3:].mean() / x.iloc[:3].mean() - 1 if len(x) >= 6 and x.iloc[:3].mean() > 0 else 0)
    # Peak trade bar fraction (max trades bar / mean)
    feats["micro_trade_spike"] = grouped["trades"].apply(
        lambda x: x.max() / x.mean() if x.mean() > 0 else 1)

    # --- CATEGORY 5: Avg Trade Size Distribution ---
    feats["micro_avg_size_mean"] = grouped["avg_trade_size"].mean()
    feats["micro_avg_size_std"] = grouped["avg_trade_size"].std()
    feats["micro_avg_size_max"] = grouped["avg_trade_size"].max()
    # Large trade fraction
    feats["micro_large_trade_pct"] = grouped["large_trade_flag"].mean()
    # Large trade volume concentration
    large_vol = grouped["large_trade_volume"].sum()
    feats["micro_large_vol_pct"] = large_vol / total_vol.replace(0, np.nan)

    # --- CATEGORY 6: Intra-hour Volatility Microstructure ---
    # Realized variance from 5m returns (more precise than 1h bar range)
    feats["micro_rv_5m"] = grouped["log_ret_sq"].sum().apply(np.sqrt) * np.sqrt(288 * 365)
    # Parkinson-like: max range within hour
    feats["micro_max_bar_range"] = grouped["bar_range"].max()
    feats["micro_mean_bar_range"] = grouped["bar_range"].mean()
    # Range concentration: max_range / mean_range (kurtosis proxy)
    feats["micro_range_spike"] = feats["micro_max_bar_range"] / feats["micro_mean_bar_range"].replace(0, np.nan)
    # Volatility path: was vol front-loaded or back-loaded?
    feats["micro_vol_trend"] = grouped["ret_sq"].apply(
        lambda x: x.iloc[-3:].mean() / x.iloc[:3].mean() - 1 if len(x) >= 6 and x.iloc[:3].mean() > 0 else 0)

    # --- CATEGORY 7: Candle Microstructure ---
    feats["micro_body_ratio_mean"] = grouped["body_ratio"].mean()
    feats["micro_upper_wick_mean"] = grouped["upper_wick"].mean()
    feats["micro_lower_wick_mean"] = grouped["lower_wick"].mean()
    # Wick asymmetry (upper - lower): positive = rejection of highs
    feats["micro_wick_asym"] = feats["micro_upper_wick_mean"] - feats["micro_lower_wick_mean"]

    # --- CATEGORY 8: Intra-hour Return Path ---
    # Return autocorrelation within hour (mean-reverting or trending intra-hour?)
    feats["micro_ret_autocorr"] = grouped["ret"].apply(
        lambda x: x.autocorr(lag=1) if len(x) >= 4 else 0)
    # Max consecutive direction (trend strength within hour)
    def max_run(x):
        if len(x) == 0:
            return 0
        signs = np.sign(x.values)
        max_r = 0
        current = 0
        for s in signs:
            if s == np.sign(current) and s != 0:
                current += s
            else:
                current = s
            max_r = max(max_r, abs(current))
        return max_r
    feats["micro_max_run"] = grouped["ret"].apply(max_run)
    # Intra-hour drawdown
    feats["micro_intra_dd"] = grouped["ret"].apply(
        lambda x: x.cumsum().min() if len(x) > 0 else 0)
    # Intra-hour rally
    feats["micro_intra_rally"] = grouped["ret"].apply(
        lambda x: x.cumsum().max() if len(x) > 0 else 0)

    # --- CATEGORY 9: Order Flow Imbalance ---
    feats["micro_ofi"] = grouped["signed_volume"].sum()
    feats["micro_ofi_norm"] = feats["micro_ofi"] / total_vol.replace(0, np.nan)

    # --- CATEGORY 10: Volume Distribution ---
    feats["micro_vol_total"] = total_vol
    feats["micro_vol_std"] = grouped["volume"].std()
    # Volume concentration: did volume come in bursts?
    feats["micro_vol_concentration"] = grouped["volume"].apply(
        lambda x: (x.max() / x.sum()) if x.sum() > 0 else 0)
    # Volume U-shape (first 3 + last 3 vs middle)
    feats["micro_vol_u_shape"] = grouped["volume"].apply(
        lambda x: (x.iloc[:3].sum() + x.iloc[-3:].sum()) / x.iloc[3:-3].sum()
        if len(x) >= 9 and x.iloc[3:-3].sum() > 0 else 1.0)

    return feats


def compute_rolling_features(feats_1h: pd.DataFrame) -> pd.DataFrame:
    """Add rolling windows to the 1h-aggregated features for temporal context."""
    out = feats_1h.copy()

    # Key features to add rolling context
    key_feats = [
        "micro_taker_imb_sum", "micro_cvd_norm", "micro_vpin",
        "micro_rv_5m", "micro_ofi_norm", "micro_large_trade_pct",
    ]

    for feat in key_feats:
        if feat in out.columns:
            # Rolling mean (smooth)
            out[f"{feat}_ma5"] = out[feat].rolling(5).mean()
            # Z-score (relative to recent history)
            ma20 = out[feat].rolling(20).mean()
            std20 = out[feat].rolling(20).std()
            out[f"{feat}_z20"] = (out[feat] - ma20) / std20.replace(0, np.nan)
            # Change from previous bar
            out[f"{feat}_diff"] = out[feat].diff()

    return out


def ic_scan(feats_1h: pd.DataFrame, df_1h: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """IC scan of aggregated 5m features against 1h forward returns."""
    results = []

    # Align by datetime
    feats_1h = feats_1h.copy()
    feats_1h.index = pd.to_datetime(feats_1h.index)

    df_1h_indexed = df_1h.set_index("datetime")

    # Common dates
    common = feats_1h.index.intersection(df_1h_indexed.index)
    feats_aligned = feats_1h.loc[common]
    close_aligned = df_1h_indexed.loc[common, "close"]

    for h in [1, 3, 6, 12, 24, 48, 96]:
        fwd = close_aligned.pct_change(h).shift(-h)

        for col in feats_aligned.columns:
            vals = feats_aligned[col]
            mask = vals.notna() & fwd.notna() & np.isfinite(vals) & np.isfinite(fwd)
            if mask.sum() < 500:
                continue
            ic, pval = stats.spearmanr(vals[mask], fwd[mask])
            results.append({
                "symbol": symbol,
                "feature": col,
                "horizon": f"{h}h",
                "horizon_bars": h,
                "ic": ic,
                "abs_ic": abs(ic),
                "pval": pval,
                "n": int(mask.sum()),
            })

    return pd.DataFrame(results)


def compare_with_existing(ic_results: pd.DataFrame, df_1h: pd.DataFrame, symbol: str):
    """Compare new 5m-aggregated features with existing 1h features."""
    print("\n  --- Comparison with existing 1h features ---")

    close = df_1h["close"]
    vol = df_1h["volume"]

    existing = {}
    # Standard 1h features
    existing["ret_5"] = close.pct_change(5)
    existing["ret_10"] = close.pct_change(10)
    existing["ret_20"] = close.pct_change(20)
    existing["vol_5"] = close.pct_change().rolling(5).std()
    existing["vol_20"] = close.pct_change().rolling(20).std()
    existing["vol_ratio"] = existing["vol_5"] / existing["vol_20"]

    if "taker_buy_volume" in df_1h:
        tbr = df_1h["taker_buy_volume"] / vol.replace(0, np.nan)
        existing["taker_buy_ratio_1h"] = tbr
        existing["taker_imbalance_1h"] = 2 * tbr - 1

    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    existing["rsi_14"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

    # Volume ratio
    existing["volume_ratio_5"] = vol / vol.rolling(5).mean()

    # Compute IC for existing features
    existing_results = []
    for h in [1, 3, 6, 12, 24, 48, 96]:
        fwd = close.pct_change(h).shift(-h)
        for fname, fvals in existing.items():
            mask = fvals.notna() & fwd.notna() & np.isfinite(fvals) & np.isfinite(fwd)
            if mask.sum() < 500:
                continue
            ic, pval = stats.spearmanr(fvals[mask], fwd[mask])
            existing_results.append({
                "feature": fname,
                "horizon": f"{h}h",
                "ic": ic,
                "abs_ic": abs(ic),
                "type": "existing_1h",
            })

    edf = pd.DataFrame(existing_results)

    # Best IC per feature (max across horizons)
    best_existing = edf.loc[edf.groupby("feature")["abs_ic"].idxmax()].sort_values("abs_ic", ascending=False)
    best_new = ic_results[ic_results["symbol"] == symbol]
    best_new = best_new.loc[best_new.groupby("feature")["abs_ic"].idxmax()].sort_values("abs_ic", ascending=False)

    print("\n  EXISTING 1h features (best IC):")
    for _, r in best_existing.head(10).iterrows():
        print(f"    {r['abs_ic']:.5f}  {r['feature']:30s}  h={r['horizon']}")

    print("\n  NEW 5m-aggregated features (top 20):")
    for _, r in best_new.head(20).iterrows():
        sig = "***" if r["pval"] < 0.001 else "**" if r["pval"] < 0.01 else "*" if r["pval"] < 0.05 else ""
        print(f"    {r['abs_ic']:.5f} {sig:3s}  {r['feature']:40s}  h={r['horizon']}  IC={r['ic']:+.5f}")

    # New features that BEAT best existing
    best_existing_ic = best_existing["abs_ic"].max()
    beating = best_new[best_new["abs_ic"] > best_existing_ic]
    print(f"\n  Features beating best existing ({best_existing_ic:.5f}):")
    if len(beating) > 0:
        for _, r in beating.iterrows():
            print(f"    {r['abs_ic']:.5f}  {r['feature']:40s}  h={r['horizon']}")
    else:
        print("    None")

    # New features with IC > 0.03 (practical threshold)
    strong = best_new[best_new["abs_ic"] > 0.03]
    print(f"\n  Strong new features (|IC| > 0.03): {len(strong)}")

    return edf


def walkforward_ic_test(feats_1h: pd.DataFrame, df_1h: pd.DataFrame, symbol: str,
                         top_features: list[str], h: int = 96):
    """Rolling walk-forward IC stability for top features."""
    print(f"\n  --- Walk-Forward IC Stability (h={h}h) ---")

    feats_1h = feats_1h.copy()
    feats_1h.index = pd.to_datetime(feats_1h.index)
    df_1h_indexed = df_1h.set_index("datetime")
    common = feats_1h.index.intersection(df_1h_indexed.index)
    feats = feats_1h.loc[common]
    close = df_1h_indexed.loc[common, "close"]
    fwd = close.pct_change(h).shift(-h)

    window = 720 * 2  # ~60 days
    step = 720  # ~30 day step

    for feat_name in top_features[:8]:
        if feat_name not in feats.columns:
            continue
        vals = feats[feat_name]

        monthly_ics = []
        for start in range(0, len(feats) - window, step):
            end = start + window
            mask = vals.iloc[start:end].notna() & fwd.iloc[start:end].notna()
            if mask.sum() < 100:
                continue
            ic, _ = stats.spearmanr(vals.iloc[start:end][mask], fwd.iloc[start:end][mask])
            monthly_ics.append(ic)

        if not monthly_ics:
            continue
        ics = pd.Series(monthly_ics)
        pos_rate = (np.sign(ics.mean()) == np.sign(ics)).mean() * 100
        icir = ics.mean() / ics.std() if ics.std() > 0 else 0
        print(f"    {feat_name:45s} IC={ics.mean():+.5f} ±{ics.std():.4f} ICIR={icir:.2f} "
              f"consistent={pos_rate:.0f}% ({len(ics)} windows)")


def main():
    for symbol in SYMBOLS:
        print(f"\n{'='*70}")
        print(f"  {symbol} — 5m Microstructure → 1h Feature Aggregation")
        print(f"{'='*70}")

        # Load data
        df_5m = load_5m(symbol)
        df_1h = load_1h(symbol)
        print(f"  5m bars: {len(df_5m)}, 1h bars: {len(df_1h)}")

        # Step 1: Compute 5m raw metrics
        print("  Computing 5m raw metrics...")
        raw_5m = compute_5m_raw(df_5m)

        # Step 2: Aggregate to 1h
        print("  Aggregating to 1h...")
        feats_1h = aggregate_to_1h(raw_5m)
        print(f"  Aggregated features: {len(feats_1h.columns)} columns, {len(feats_1h)} hours")

        # Step 3: Add rolling context
        print("  Adding rolling features...")
        feats_1h = compute_rolling_features(feats_1h)
        print(f"  Total features after rolling: {len(feats_1h.columns)}")

        # Step 4: IC scan
        print("  Running IC scan...")
        ic_results = ic_scan(feats_1h, df_1h, symbol)

        # Step 5: Summary
        print(f"\n{'='*60}")
        print(f"  {symbol} — IC Scan Results")
        print(f"{'='*60}")

        # Category summary
        def get_category(fname):
            if fname.startswith("micro_taker") or fname.startswith("micro_buy_pressure"):
                return "taker_flow"
            elif "cvd" in fname or "ofi" in fname:
                return "order_flow"
            elif "vpin" in fname:
                return "vpin"
            elif "trade" in fname or "avg_size" in fname or "large" in fname:
                return "trade_intensity"
            elif "rv" in fname or "range" in fname or "vol_trend" in fname:
                return "volatility"
            elif "body" in fname or "wick" in fname:
                return "candle"
            elif "ret" in fname or "run" in fname or "dd" in fname or "rally" in fname:
                return "return_path"
            elif "vol" in fname:
                return "volume"
            else:
                return "other"

        ic_results["category"] = ic_results["feature"].apply(get_category)

        cat_summary = ic_results.groupby(["category", "horizon"]).agg(
            mean_ic=("abs_ic", "mean"),
            max_ic=("abs_ic", "max"),
            n_feats=("feature", "nunique"),
        ).round(5)
        print("\n  Category × Horizon (mean |IC|):")
        pivot = ic_results.groupby(["category", "horizon_bars"])["abs_ic"].mean().unstack()
        pivot.columns = [f"{c}h" for c in pivot.columns]
        print(pivot.round(5).to_string())

        # Step 6: Compare with existing
        compare_with_existing(ic_results, df_1h, symbol)

        # Step 7: WF stability for top features
        top = ic_results.loc[ic_results.groupby("feature")["abs_ic"].idxmax()]
        top = top.sort_values("abs_ic", ascending=False)
        top_names = top["feature"].head(12).tolist()
        walkforward_ic_test(feats_1h, df_1h, symbol, top_names, h=96)

    # Save
    print("\n\nDone. Results ready for model training integration.")


if __name__ == "__main__":
    main()
