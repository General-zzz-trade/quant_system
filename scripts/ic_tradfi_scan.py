"""IC scan for untested traditional finance features from cross_market_daily.csv.

Computes rank IC (Spearman correlation) between each feature and forward 24h return
over 30d, 60d, 90d, 180d trailing windows.
"""
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

# Features to test
TEST_FEATURES = [
    "tlt_ret_5d", "hyg_ret_1d", "credit_spread_chg", "yield_curve_proxy",
    "uso_ret_5d", "fxi_ret_1d", "gbtc_ret_1d", "ibit_ret_1d",
    "spy_ret_5d", "xlf_ret_5d", "eem_ret_5d", "vix_chg_5d", "gbtc_premium_dev",
]

WINDOWS_DAYS = [30, 60, 90, 180]
SYMBOLS = ["BTCUSDT", "ETHUSDT"]


def load_and_merge(symbol: str) -> pd.DataFrame:
    kline = pd.read_csv(f"data_files/{symbol}_1h.csv")
    cm = pd.read_csv("data_files/cross_market_daily.csv")

    # Convert kline timestamp to date
    kline["open_time_dt"] = pd.to_datetime(kline["open_time"], unit="ms")
    kline["date"] = kline["open_time_dt"].dt.strftime("%Y-%m-%d")

    # Forward 24h return (24 bars ahead for 1h data)
    kline["fwd_ret_24h"] = kline["close"].shift(-24) / kline["close"] - 1

    # Merge cross-market data (forward-fill within each day)
    merged = kline.merge(cm, on="date", how="left")

    # Forward-fill cross-market features (they're daily, klines are hourly)
    for feat in TEST_FEATURES:
        if feat in merged.columns:
            merged[feat] = merged[feat].ffill()

    return merged


def compute_ic(df: pd.DataFrame, feature: str, window_days: int) -> float:
    """Compute rank IC over the last window_days of data."""
    # Use last N days
    if "date" not in df.columns:
        return np.nan

    all_dates = sorted(df["date"].unique())
    cutoff_idx = max(0, len(all_dates) - window_days)
    recent_dates = set(all_dates[cutoff_idx:])

    sub = df[df["date"].isin(recent_dates)].copy()

    # Drop NaN in both feature and target
    mask = sub[feature].notna() & sub["fwd_ret_24h"].notna()
    sub = sub[mask]

    if len(sub) < 50:
        return np.nan

    ic, _ = spearmanr(sub[feature], sub["fwd_ret_24h"])
    return ic


def main():
    for symbol in SYMBOLS:
        print(f"\n{'='*80}")
        print(f"  IC SCAN: {symbol}  (feature vs forward 24h return)")
        print(f"{'='*80}")

        df = load_and_merge(symbol)
        print(f"  Data: {len(df)} bars, date range: {df['date'].min()} to {df['date'].max()}")
        print(f"  Forward return coverage: {df['fwd_ret_24h'].notna().sum()} bars")
        print()

        results = []
        for feat in TEST_FEATURES:
            if feat not in df.columns:
                print(f"  SKIP {feat} — not in data")
                continue

            row = {"feature": feat}
            for w in WINDOWS_DAYS:
                ic = compute_ic(df, feat, w)
                row[f"IC_{w}d"] = ic

            # Average absolute IC across windows
            ics = [row[f"IC_{w}d"] for w in WINDOWS_DAYS if not np.isnan(row.get(f"IC_{w}d", np.nan))]
            row["avg_abs_IC"] = np.mean(np.abs(ics)) if ics else np.nan
            row["sign_stable"] = all(x > 0 for x in ics) or all(x < 0 for x in ics) if len(ics) >= 3 else False
            results.append(row)

        # Sort by avg absolute IC descending
        results.sort(key=lambda r: -r.get("avg_abs_IC", 0))

        # Print table
        header = f"{'Feature':<22} {'IC_30d':>8} {'IC_60d':>8} {'IC_90d':>8} {'IC_180d':>8} {'Avg|IC|':>8} {'Stable':>6}"
        print(header)
        print("-" * len(header))
        for r in results:
            vals = []
            for w in WINDOWS_DAYS:
                v = r.get(f"IC_{w}d", np.nan)
                vals.append(f"{v:+.4f}" if not np.isnan(v) else "    N/A")
            avg = r.get("avg_abs_IC", np.nan)
            avg_s = f"{avg:.4f}" if not np.isnan(avg) else "   N/A"
            stable = "YES" if r.get("sign_stable") else "no"
            print(f"{r['feature']:<22} {vals[0]:>8} {vals[1]:>8} {vals[2]:>8} {vals[3]:>8} {avg_s:>8} {stable:>6}")

        # Summary
        print()
        strong = [r for r in results if r.get("avg_abs_IC", 0) >= 0.05 and r.get("sign_stable")]
        if strong:
            print(f"  STRONG candidates (|IC| >= 0.05, sign-stable):")
            for r in strong:
                print(f"    {r['feature']}: avg|IC| = {r['avg_abs_IC']:.4f}")
        else:
            print("  No strong sign-stable candidates found.")

        moderate = [r for r in results if r.get("avg_abs_IC", 0) >= 0.03 and not r.get("sign_stable")]
        if moderate:
            print(f"  MODERATE candidates (|IC| >= 0.03, sign NOT stable):")
            for r in moderate:
                print(f"    {r['feature']}: avg|IC| = {r['avg_abs_IC']:.4f}")


if __name__ == "__main__":
    main()
