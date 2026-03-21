#!/usr/bin/env python3
"""IC screening for implemented-but-untested alpha factors.

Tests:
1. Multi-exchange funding spread (3 features)
2. On-chain exchange flow (2 features)
3. 4h multi-timeframe features (10 features)
4. Feature interaction terms
5. Existing features at different lookback windows

Computes rank IC (Spearman) at multiple forward horizons.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, "/quant_system")


def load_data(symbol: str) -> pd.DataFrame:
    path = f"data_files/{symbol}_1h.csv"
    df = pd.read_csv(path)
    if "open_time" in df.columns:
        df = df.sort_values("open_time").reset_index(drop=True)
    for c in ["close", "open", "high", "low", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def compute_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute base features via RustFeatureEngine."""
    from _quant_hotpath import RustFeatureEngine
    eng = RustFeatureEngine()
    feats = []
    for _, row in df.iterrows():
        eng.push_bar(
            close=float(row["close"]), volume=float(row.get("volume", 0)),
            high=float(row.get("high", row["close"])),
            low=float(row.get("low", row["close"])),
            open=float(row.get("open", row["close"])),
        )
        feats.append(eng.get_features())
    return pd.DataFrame(feats, index=df.index)


def add_interaction_features(feat_df: pd.DataFrame) -> pd.DataFrame:
    """Add feature interactions and alternative lookback windows."""
    df = feat_df.copy()

    # ── Interaction terms ──
    pairs = [
        ("rsi_14", "vol_20", "rsi_x_vol"),
        ("rsi_14", "atr_norm_14", "rsi_x_atr"),
        ("macd_hist", "vol_20", "macd_x_vol"),
        ("ret_1", "vol_20", "ret1_x_vol"),
        ("close_vs_ma50", "vol_20", "trend_x_vol"),
        ("bb_pctb_20", "vol_20", "bb_x_vol"),
        ("rsi_14", "close_vs_ma50", "rsi_x_trend"),
        ("adx_14", "vol_20", "adx_x_vol"),
    ]
    for a, b, name in pairs:
        if a in df.columns and b in df.columns:
            va = df[a].fillna(0).values
            vb = df[b].fillna(0).values
            df[name] = va * vb

    # ── Momentum persistence (autocorrelation of returns) ──
    if "ret_1" in df.columns:
        ret = df["ret_1"].fillna(0).values
        # Rolling 24-bar autocorrelation of returns
        ac = np.full(len(ret), np.nan)
        for i in range(24, len(ret)):
            chunk = ret[i - 24:i]
            chunk1 = chunk[:-1]
            chunk2 = chunk[1:]
            if np.std(chunk1) > 1e-10 and np.std(chunk2) > 1e-10:
                ac[i] = np.corrcoef(chunk1, chunk2)[0, 1]
        df["ret_autocorr_24"] = ac

    # ── Volume surprise (current vol / rolling median) ──
    if "vol_ratio_20" in df.columns:
        vr = df["vol_ratio_20"].fillna(1.0).values
        # Log volume surprise
        df["log_vol_surprise"] = np.log1p(np.maximum(vr, 0))

    # ── Price efficiency ratio (net move / total path) ──
    # Already partially captured by ranging detector, but expose as feature
    if "ret_1" in df.columns:
        ret = df["ret_1"].fillna(0).values
        eff = np.full(len(ret), np.nan)
        for i in range(24, len(ret)):
            chunk = ret[i - 24:i]
            net = abs(np.sum(chunk))
            total = np.sum(np.abs(chunk))
            eff[i] = net / total if total > 1e-10 else 0
        df["price_efficiency_24"] = eff

    # ── Skewness of returns (24-bar) ──
    if "ret_1" in df.columns:
        ret = df["ret_1"].fillna(0).values
        skew = np.full(len(ret), np.nan)
        for i in range(24, len(ret)):
            chunk = ret[i - 24:i]
            mu = np.mean(chunk)
            std = np.std(chunk)
            if std > 1e-10:
                skew[i] = np.mean(((chunk - mu) / std) ** 3)
        df["ret_skew_24"] = skew

    # ── Kurtosis of returns (24-bar) ──
    if "ret_1" in df.columns:
        ret = df["ret_1"].fillna(0).values
        kurt = np.full(len(ret), np.nan)
        for i in range(24, len(ret)):
            chunk = ret[i - 24:i]
            mu = np.mean(chunk)
            std = np.std(chunk)
            if std > 1e-10:
                kurt[i] = np.mean(((chunk - mu) / std) ** 4) - 3.0
        df["ret_kurtosis_24"] = kurt

    # ── Overnight gap (close-to-open return) ──
    # Simulated: ret_1 minus intrabar return
    if "body_ratio" in df.columns and "ret_1" in df.columns:
        # body_ratio ≈ (close-open)/(high-low), proxy for intrabar direction
        atr = df.get("atr_norm_14", pd.Series(0, index=df.index)).fillna(0)
        df["gap_proxy"] = df["ret_1"].fillna(0) - df["body_ratio"].fillna(0) * atr

    return df


def compute_ic_table(feat_df: pd.DataFrame, closes: np.ndarray,
                     horizons: list[int] | None = None) -> pd.DataFrame:
    if horizons is None:
        horizons = [1, 3, 6, 12, 24]
    """Compute rank IC for each feature at each horizon."""
    results = []
    n = len(closes)

    for h in horizons:
        if n <= h + 100:
            continue
        fwd_ret = (closes[h:] - closes[:-h]) / closes[:-h]

        for col in feat_df.columns:
            vals = pd.to_numeric(feat_df[col], errors="coerce").values[:-h]
            mask = np.isfinite(vals) & np.isfinite(fwd_ret)
            if mask.sum() < 100:
                continue
            ic, pval = spearmanr(vals[mask], fwd_ret[mask])
            if not np.isfinite(ic):
                continue
            results.append({
                "feature": col, "horizon": h,
                "ic": ic, "pval": pval, "abs_ic": abs(ic),
                "n_obs": int(mask.sum()),
                "significant": pval < 0.05 and abs(ic) > 0.02,
            })

    return pd.DataFrame(results)


def main():
    symbols = ["BTCUSDT", "ETHUSDT", "SUIUSDT", "AXSUSDT"]

    for symbol in symbols:
        print(f"\n{'='*80}")
        print(f"  IC SCREEN: {symbol}")
        print(f"{'='*80}")

        df = load_data(symbol)
        print(f"  Data: {len(df)} bars")

        print("  Computing features...")
        feat_df = compute_all_features(df)

        print("  Adding interaction & experimental features...")
        feat_df = add_interaction_features(feat_df)

        closes = df["close"].values
        total_features = len(feat_df.columns)
        print(f"  Total features: {total_features}")

        print("  Computing IC at h=1,3,6,12,24...")
        ic_df = compute_ic_table(feat_df, closes, horizons=[1, 3, 6, 12, 24])

        if ic_df.empty:
            print("  No IC results!")
            continue

        # ── Top features by max |IC| across horizons ──
        best = ic_df.loc[ic_df.groupby("feature")["abs_ic"].idxmax()]
        best = best.sort_values("abs_ic", ascending=False)

        print("\n  TOP 30 FEATURES (by max |IC| across horizons):")
        print(f"  {'Feature':<30} {'H':>3} {'IC':>8} {'|IC|':>6} {'p-val':>8} {'Sig':>4}")
        print(f"  {'-'*62}")
        for _, row in best.head(30).iterrows():
            sig = "★" if row["significant"] else " "
            print(f"  {row['feature']:<30} {row['horizon']:>3} {row['ic']:>+8.4f} "
                  f"{row['abs_ic']:>6.4f} {row['pval']:>8.4f} {sig}")

        # ── New/experimental features only ──
        new_features = [
            "rsi_x_vol", "rsi_x_atr", "macd_x_vol", "ret1_x_vol",
            "trend_x_vol", "bb_x_vol", "rsi_x_trend", "adx_x_vol",
            "ret_autocorr_24", "log_vol_surprise", "price_efficiency_24",
            "ret_skew_24", "ret_kurtosis_24", "gap_proxy",
        ]
        new_ic = ic_df[ic_df["feature"].isin(new_features)]
        if not new_ic.empty:
            best_new = new_ic.loc[new_ic.groupby("feature")["abs_ic"].idxmax()]
            best_new = best_new.sort_values("abs_ic", ascending=False)

            print("\n  NEW/EXPERIMENTAL FEATURES:")
            print(f"  {'Feature':<30} {'H':>3} {'IC':>8} {'|IC|':>6} {'p-val':>8} {'Sig':>4}")
            print(f"  {'-'*62}")
            for _, row in best_new.iterrows():
                sig = "★" if row["significant"] else " "
                print(f"  {row['feature']:<30} {row['horizon']:>3} {row['ic']:>+8.4f} "
                      f"{row['abs_ic']:>6.4f} {row['pval']:>8.4f} {sig}")

        # ── Features NOT currently in production ──
        try:
            from features.feature_catalog import PRODUCTION_FEATURES
            non_prod = [f for f in feat_df.columns if f not in PRODUCTION_FEATURES]
        except ImportError:
            non_prod = []

        if non_prod:
            non_prod_ic = ic_df[ic_df["feature"].isin(non_prod)]
            if not non_prod_ic.empty:
                best_np = non_prod_ic.loc[non_prod_ic.groupby("feature")["abs_ic"].idxmax()]
                best_np = best_np[best_np["significant"]].sort_values("abs_ic", ascending=False)
                if not best_np.empty:
                    print("\n  SIGNIFICANT NON-PRODUCTION FEATURES:")
                    print(f"  {'Feature':<30} {'H':>3} {'IC':>8} {'p-val':>8}")
                    print(f"  {'-'*52}")
                    for _, row in best_np.head(15).iterrows():
                        print(f"  {row['feature']:<30} {row['horizon']:>3} {row['ic']:>+8.4f} {row['pval']:>8.4f}")

    print(f"\n{'='*80}")
    print("  DONE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
