#!/usr/bin/env python3
"""Comprehensive 5m native feature IC scan.

Scans 4 categories of genuinely 5m-frequency features:
1. Microstructure (taker flow, trade intensity, avg size)
2. Intraday seasonality (hour-of-day, session, funding settlement)
3. Volume/volatility dynamics (volume clock, vol clustering, range)
4. Multi-bar patterns (sequential patterns, momentum divergence)

Output: IC rank table by category with statistical significance.
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
FWD_BARS = [1, 3, 6, 12, 24, 48]  # 5m, 15m, 30m, 1h, 2h, 4h forward


def load_5m(symbol: str) -> pd.DataFrame:
    path = DATA_DIR / f"{symbol}_5m.csv"
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["open_time"], unit="ms")
    df = df.sort_values("open_time").reset_index(drop=True)
    # Basic derived
    df["ret"] = df["close"].pct_change()
    df["log_ret"] = np.log(df["close"] / df["close"].shift(1))
    df["taker_sell_volume"] = df["volume"] - df["taker_buy_volume"]
    df["taker_sell_quote"] = df["quote_volume"] - df["taker_buy_quote_volume"]
    return df


def add_microstructure_features(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Category 1: Microstructure — genuinely 5m frequency."""
    feats = {}

    # Taker buy ratio (fraction of volume that's aggressive buy)
    tbr = df["taker_buy_volume"] / df["volume"].replace(0, np.nan)
    feats["taker_buy_ratio"] = tbr
    feats["taker_buy_ratio_ma6"] = tbr.rolling(6).mean()
    feats["taker_buy_ratio_ma12"] = tbr.rolling(12).mean()
    feats["taker_buy_ratio_zscore_24"] = (tbr - tbr.rolling(24).mean()) / tbr.rolling(24).std()
    feats["taker_buy_ratio_zscore_72"] = (tbr - tbr.rolling(72).mean()) / tbr.rolling(72).std()

    # Taker imbalance (-1 to +1)
    ti = 2 * tbr - 1
    feats["taker_imbalance"] = ti
    feats["taker_imbalance_cum6"] = ti.rolling(6).sum()
    feats["taker_imbalance_cum12"] = ti.rolling(12).sum()
    feats["taker_imbalance_cum24"] = ti.rolling(24).sum()
    feats["taker_imbalance_cum48"] = ti.rolling(48).sum()

    # CVD (cumulative volume delta) — rolling
    vd = df["taker_buy_volume"] - df["taker_sell_volume"]
    feats["cvd_6"] = vd.rolling(6).sum()
    feats["cvd_12"] = vd.rolling(12).sum()
    feats["cvd_24"] = vd.rolling(24).sum()
    feats["cvd_48"] = vd.rolling(48).sum()

    # CVD-price divergence: CVD direction vs price direction
    for w in [6, 12, 24]:
        cvd_dir = np.sign(vd.rolling(w).sum())
        price_dir = np.sign(df["close"].diff(w))
        feats[f"cvd_price_div_{w}"] = cvd_dir * price_dir  # -1 = diverging

    # Trade intensity
    trade_cnt = df["trades"]
    feats["trade_intensity_ratio_6"] = trade_cnt / trade_cnt.rolling(6).mean()
    feats["trade_intensity_ratio_12"] = trade_cnt / trade_cnt.rolling(12).mean()
    feats["trade_intensity_ratio_24"] = trade_cnt / trade_cnt.rolling(24).mean()
    feats["trade_intensity_zscore_48"] = (trade_cnt - trade_cnt.rolling(48).mean()) / trade_cnt.rolling(48).std()

    # Average trade size
    avg_size = df["quote_volume"] / df["trades"].replace(0, np.nan)
    feats["avg_trade_size"] = avg_size
    feats["avg_trade_size_ratio_12"] = avg_size / avg_size.rolling(12).mean()
    feats["avg_trade_size_ratio_24"] = avg_size / avg_size.rolling(24).mean()
    feats["avg_trade_size_zscore_48"] = (avg_size - avg_size.rolling(48).mean()) / avg_size.rolling(48).std()

    # Large trade proxy: avg_size spike + volume spike
    vol_ratio = df["volume"] / df["volume"].rolling(12).mean()
    feats["vol_x_size_spike"] = vol_ratio * (avg_size / avg_size.rolling(12).mean())

    # VPIN proxy (simplified: |buy - sell| / total over buckets)
    for w in [12, 24, 48]:
        abs_imb = (df["taker_buy_volume"] - df["taker_sell_volume"]).abs()
        feats[f"vpin_proxy_{w}"] = abs_imb.rolling(w).sum() / df["volume"].rolling(w).sum()

    # Kyle lambda proxy: |ret| / volume (price impact per unit volume)
    for w in [12, 24]:
        ret_abs = df["log_ret"].abs()
        feats[f"kyle_lambda_{w}"] = ret_abs.rolling(w).sum() / df["volume"].rolling(w).sum()

    # Amihud illiquidity: |ret| / quote_volume
    for w in [12, 24]:
        feats[f"amihud_{w}"] = df["log_ret"].abs().rolling(w).mean() / df["quote_volume"].rolling(w).mean()

    return feats


def add_intraday_features(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Category 2: Intraday seasonality — time-based patterns."""
    feats = {}

    hour = df["datetime"].dt.hour
    minute = df["datetime"].dt.minute
    dow = df["datetime"].dt.dayofweek

    # Hour of day (cyclical encoding)
    feats["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    feats["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    # Session flags
    feats["session_asia"] = ((hour >= 0) & (hour < 8)).astype(float)
    feats["session_europe"] = ((hour >= 8) & (hour < 16)).astype(float)
    feats["session_us"] = ((hour >= 14) & (hour < 22)).astype(float)

    # Session overlap (EU+US most volatile)
    feats["session_overlap_eu_us"] = ((hour >= 14) & (hour < 16)).astype(float)

    # Funding settlement approach (every 8h: 00:00, 08:00, 16:00 UTC)
    minutes_to_funding = pd.Series(0.0, index=df.index)
    for settle_h in [0, 8, 16]:
        dist = ((settle_h * 60) - (hour * 60 + minute)) % (8 * 60)
        minutes_to_funding = np.minimum(minutes_to_funding.values if minutes_to_funding.sum() > 0
                                         else np.full(len(df), 999), dist)
    feats["minutes_to_funding"] = pd.Series(minutes_to_funding, index=df.index)
    feats["near_funding_30m"] = (feats["minutes_to_funding"] <= 30).astype(float)
    feats["near_funding_60m"] = (feats["minutes_to_funding"] <= 60).astype(float)

    # Day of week
    feats["is_weekend"] = (dow >= 5).astype(float)
    feats["is_monday"] = (dow == 0).astype(float)

    # Intraday return from midnight
    midnight_close = df.groupby(df["datetime"].dt.date)["close"].transform("first")
    feats["intraday_ret"] = df["close"] / midnight_close - 1

    # Volume seasonality: current bar volume vs same-time-of-day average
    df_copy = df.copy()
    df_copy["hm"] = hour * 100 + minute
    vol_by_time = df_copy.groupby("hm")["volume"].transform("mean")
    feats["vol_vs_time_avg"] = df["volume"] / vol_by_time

    # Trade count seasonality
    trades_by_time = df_copy.groupby("hm")["trades"].transform("mean")
    feats["trades_vs_time_avg"] = df["trades"] / trades_by_time

    return feats


def add_vol_dynamics_features(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Category 3: Volume/volatility dynamics — 5m granularity."""
    feats = {}

    # Realized vol at different windows (5m-native)
    for w in [6, 12, 24, 48, 96]:
        feats[f"rv_{w}"] = df["log_ret"].rolling(w).std() * np.sqrt(288)  # annualized from 5m

    # Vol ratio (short/long)
    feats["vol_ratio_6_24"] = df["log_ret"].rolling(6).std() / df["log_ret"].rolling(24).std()
    feats["vol_ratio_12_48"] = df["log_ret"].rolling(12).std() / df["log_ret"].rolling(48).std()
    feats["vol_ratio_24_96"] = df["log_ret"].rolling(24).std() / df["log_ret"].rolling(96).std()

    # Vol acceleration
    rv12 = df["log_ret"].rolling(12).std()
    feats["vol_accel_12"] = rv12.diff(6)

    # Range-based vol (Parkinson)
    hl = np.log(df["high"] / df["low"])
    for w in [12, 24]:
        feats[f"parkinson_vol_{w}"] = np.sqrt((hl ** 2).rolling(w).mean() / (4 * np.log(2)))

    # Range vs realized vol
    feats["range_vs_rv_12"] = hl / df["log_ret"].abs().rolling(12).mean().replace(0, np.nan)

    # Volume dynamics
    vol = df["volume"]
    for w in [6, 12, 24, 48]:
        feats[f"vol_ma_ratio_{w}"] = vol / vol.rolling(w).mean()

    # Volume acceleration
    feats["vol_accel_6"] = vol.rolling(6).mean().diff(3)

    # Quote volume ($ denominated)
    qv = df["quote_volume"]
    feats["qvol_ratio_12"] = qv / qv.rolling(12).mean()

    # Volume-price correlation (rolling)
    for w in [12, 24]:
        feats[f"vol_price_corr_{w}"] = df["volume"].rolling(w).corr(df["ret"])

    # Bar range (high-low)/close
    bar_range = (df["high"] - df["low"]) / df["close"]
    feats["bar_range"] = bar_range
    feats["bar_range_ratio_12"] = bar_range / bar_range.rolling(12).mean()
    feats["bar_range_zscore_24"] = (bar_range - bar_range.rolling(24).mean()) / bar_range.rolling(24).std()

    # Body ratio: |close-open|/(high-low) — candle body strength
    body = (df["close"] - df["open"]).abs()
    full_range = (df["high"] - df["low"]).replace(0, np.nan)
    feats["body_ratio"] = body / full_range

    # Upper/lower shadow ratio
    feats["upper_shadow"] = (df["high"] - df[["close", "open"]].max(axis=1)) / full_range
    feats["lower_shadow"] = (df[["close", "open"]].min(axis=1) - df["low"]) / full_range

    return feats


def add_multibar_features(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Category 4: Multi-bar patterns — sequential analysis."""
    feats = {}

    ret = df["ret"]

    # Momentum at various 5m-native windows
    for w in [3, 6, 12, 24, 48, 96]:
        feats[f"mom_{w}"] = df["close"].pct_change(w)

    # RSI at 5m-native windows
    for w in [6, 12, 24]:
        delta = ret.copy()
        gain = delta.clip(lower=0).rolling(w).mean()
        loss = (-delta.clip(upper=0)).rolling(w).mean()
        rs = gain / loss.replace(0, np.nan)
        feats[f"rsi_{w}"] = 100 - 100 / (1 + rs)

    # Consecutive bar direction (run length)
    direction = np.sign(ret)
    runs = direction.copy()
    for i in range(1, len(runs)):
        if direction.iloc[i] == direction.iloc[i - 1] and direction.iloc[i] != 0:
            runs.iloc[i] = runs.iloc[i - 1] + direction.iloc[i]
        else:
            runs.iloc[i] = direction.iloc[i]
    feats["bar_run_length"] = runs

    # Mean reversion signal: z-score of close vs VWAP
    for w in [12, 24, 48]:
        vwap = df["quote_volume"].rolling(w).sum() / df["volume"].rolling(w).sum()
        feats[f"close_vs_vwap_{w}"] = (df["close"] - vwap) / vwap

    # Bollinger band position
    for w in [12, 24]:
        ma = df["close"].rolling(w).mean()
        std = df["close"].rolling(w).std()
        feats[f"bb_pos_{w}"] = (df["close"] - ma) / std.replace(0, np.nan)

    # Price distance from rolling high/low
    for w in [24, 48, 96]:
        feats[f"dist_high_{w}"] = df["close"] / df["high"].rolling(w).max() - 1
        feats[f"dist_low_{w}"] = df["close"] / df["low"].rolling(w).min() - 1

    # Return autocorrelation
    for lag in [1, 3, 6]:
        feats[f"ret_autocorr_{lag}"] = ret.rolling(24).corr(ret.shift(lag))

    # Hurst exponent proxy (R/S method simplified)
    for w in [48, 96]:
        cum = (ret - ret.rolling(w).mean()).rolling(w).sum()
        r = cum.rolling(w).max() - cum.rolling(w).min()
        s = ret.rolling(w).std()
        feats[f"hurst_proxy_{w}"] = np.log(r / s.replace(0, np.nan)) / np.log(w)

    # MACD at 5m scale
    ema_fast = df["close"].ewm(span=12).mean()
    ema_slow = df["close"].ewm(span=26).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=9).mean()
    feats["macd_5m"] = macd / df["close"]
    feats["macd_hist_5m"] = (macd - macd_signal) / df["close"]

    return feats


def compute_ic(features: dict[str, pd.Series], fwd_ret: pd.Series, name: str) -> list[dict]:
    """Compute rank IC (Spearman) for each feature vs forward return."""
    results = []
    for fname, fvals in features.items():
        # Align and drop NaN
        mask = fvals.notna() & fwd_ret.notna() & np.isfinite(fvals) & np.isfinite(fwd_ret)
        if mask.sum() < 1000:
            continue
        x = fvals[mask].values
        y = fwd_ret[mask].values
        ic, pval = stats.spearmanr(x, y)
        results.append({
            "symbol": name,
            "feature": fname,
            "ic": ic,
            "abs_ic": abs(ic),
            "pval": pval,
            "n": int(mask.sum()),
        })
    return results


def main():
    all_results = []

    for symbol in SYMBOLS:
        print(f"\n{'='*60}")
        print(f"  {symbol} — 5m Native Feature IC Scan")
        print(f"{'='*60}")

        df = load_5m(symbol)
        print(f"Loaded {len(df)} bars, range: {df['datetime'].iloc[0]} → {df['datetime'].iloc[-1]}")

        # Compute all feature categories
        print("Computing features...")
        cats = {
            "microstructure": add_microstructure_features(df),
            "intraday": add_intraday_features(df),
            "vol_dynamics": add_vol_dynamics_features(df),
            "multibar": add_multibar_features(df),
        }

        # Forward returns
        for h in FWD_BARS:
            fwd = df["close"].pct_change(h).shift(-h)
            fwd_name = f"fwd_{h}bar"

            for cat_name, feats in cats.items():
                results = compute_ic(feats, fwd, symbol)
                for r in results:
                    r["category"] = cat_name
                    r["horizon"] = f"{h * 5}m"
                    r["horizon_bars"] = h
                all_results.extend(results)

        # Category summary
        rdf = pd.DataFrame(all_results)
        rdf_sym = rdf[rdf["symbol"] == symbol]

        print(f"\n--- {symbol} Category Summary (mean |IC|) ---")
        cat_summary = rdf_sym.groupby(["category", "horizon"]).agg(
            mean_abs_ic=("abs_ic", "mean"),
            max_abs_ic=("abs_ic", "max"),
            n_features=("feature", "nunique"),
        ).round(5)
        print(cat_summary.to_string())

    # Global analysis
    rdf = pd.DataFrame(all_results)

    print(f"\n\n{'='*60}")
    print("  TOP 30 FEATURES BY |IC| (across all symbols & horizons)")
    print(f"{'='*60}")

    # Best IC per feature (max across horizons)
    best = rdf.loc[rdf.groupby(["symbol", "feature"])["abs_ic"].idxmax()]
    best = best.sort_values("abs_ic", ascending=False).head(30)
    for _, row in best.iterrows():
        sig = "***" if row["pval"] < 0.001 else "**" if row["pval"] < 0.01 else "*" if row["pval"] < 0.05 else ""
        print(f"  {row['abs_ic']:.5f} {sig:3s}  {row['symbol']:10s} {row['feature']:35s} {row['category']:15s} h={row['horizon']:5s} (IC={row['ic']:+.5f})")

    # By category, best horizon
    print(f"\n\n{'='*60}")
    print("  CATEGORY × HORIZON HEATMAP (mean |IC|)")
    print(f"{'='*60}")
    pivot = rdf.groupby(["category", "horizon_bars"])["abs_ic"].mean().unstack()
    pivot.columns = [f"{c*5}m" for c in pivot.columns]
    print(pivot.round(5).to_string())

    # Significance filter: features with |IC| > 0.01 and p < 0.01
    print(f"\n\n{'='*60}")
    print("  SIGNIFICANT FEATURES (|IC| > 0.010, p < 0.01)")
    print(f"{'='*60}")
    sig_feats = rdf[(rdf["abs_ic"] > 0.010) & (rdf["pval"] < 0.01)]
    if len(sig_feats) > 0:
        sig_summary = sig_feats.groupby(["category", "feature"]).agg(
            mean_ic=("ic", "mean"),
            max_abs_ic=("abs_ic", "max"),
            best_horizon=("horizon", "first"),
            symbols=("symbol", lambda x: ",".join(sorted(set(x)))),
        ).sort_values("max_abs_ic", ascending=False)
        print(sig_summary.head(40).to_string())
    else:
        print("  No features pass threshold")

    # Save results
    rdf.to_csv("data_files/5m_feature_ic_scan.csv", index=False)
    print("\nFull results saved to data_files/5m_feature_ic_scan.csv")
    print(f"Total features scanned: {rdf['feature'].nunique()}")
    print(f"Total IC computations: {len(rdf)}")


if __name__ == "__main__":
    main()
