#!/usr/bin/env python3
"""Download macroeconomic data from FRED (Federal Reserve Economic Data).

No API key needed for basic CSV endpoint. Downloads key macro indicators
that drive BTC price through liquidity/policy expectations.

Output: data_files/fred_macro.csv

Usage:
    python3 -m scripts.data.download_fred
    python3 -m scripts.data.download_fred --start 2020-01-01
"""
from __future__ import annotations

import argparse
import logging
import urllib.request
from io import StringIO
from pathlib import Path

import pandas as pd

log = logging.getLogger("fred")

# FRED series IDs and descriptions
SERIES = {
    "DFF": "fed_funds_rate",          # Federal Funds Effective Rate (daily)
    "DGS2": "treasury_2y",            # 2-Year Treasury Yield (daily)
    "DGS10": "treasury_10y",          # 10-Year Treasury Yield (daily)
    "T10Y2Y": "yield_spread_10y2y",   # 10Y-2Y Spread (daily, recession signal)
    "DTWEXBGS": "usd_index",          # Trade Weighted USD Index (daily)
    "VIXCLS": "vix_fred",             # VIX Close (daily, cross-check Yahoo)
    "WM2NS": "m2_money_supply",       # M2 Money Supply (weekly)
    "CPIAUCSL": "cpi",                # CPI All Items (monthly)
    "UNRATE": "unemployment",         # Unemployment Rate (monthly)
    "WALCL": "fed_balance_sheet",     # Fed Total Assets (weekly, QE/QT proxy)
}

OUTPUT_PATH = Path("data_files/fred_macro.csv")
FRED_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv"


def fetch_series(series_id: str, start: str = "2019-01-01") -> pd.Series:
    """Fetch a FRED series as CSV (no API key needed)."""
    url = f"{FRED_BASE}?id={series_id}&cosd={start}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    resp = urllib.request.urlopen(req, timeout=15)
    text = resp.read().decode("utf-8")

    df = pd.read_csv(StringIO(text))
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"]).set_index("date")
    return df["value"]


def build_fred_features(start: str = "2019-01-01") -> pd.DataFrame:
    """Download all FRED series and compute derived features."""
    raw = {}
    for series_id, name in SERIES.items():
        try:
            s = fetch_series(series_id, start)
            raw[name] = s
            log.info("%s (%s): %d points", series_id, name, len(s))
        except Exception as e:
            log.warning("%s failed: %s", series_id, e)

    if not raw:
        log.error("No FRED data downloaded")
        return pd.DataFrame()

    # Combine all series (different frequencies → forward-fill)
    combined = pd.DataFrame(raw)
    combined = combined.sort_index()

    # Resample everything to daily, forward-fill
    daily = combined.resample("1D").last().ffill()

    # Compute derived features
    out = pd.DataFrame(index=daily.index)
    out.index.name = "date"

    # Raw levels
    if "fed_funds_rate" in daily:
        out["fed_rate"] = daily["fed_funds_rate"]
    if "treasury_2y" in daily:
        out["treasury_2y"] = daily["treasury_2y"]
    if "treasury_10y" in daily:
        out["treasury_10y"] = daily["treasury_10y"]

    # Yield curve spread (10Y - 2Y): negative = inverted = recession signal
    if "yield_spread_10y2y" in daily:
        out["yield_spread"] = daily["yield_spread_10y2y"]
    elif "treasury_10y" in daily and "treasury_2y" in daily:
        out["yield_spread"] = daily["treasury_10y"] - daily["treasury_2y"]

    # Rate changes (momentum)
    if "fed_funds_rate" in daily:
        out["fed_rate_chg_30d"] = daily["fed_funds_rate"].diff(30)

    # M2 growth rate (monthly → daily ffill)
    if "m2_money_supply" in daily:
        m2 = daily["m2_money_supply"]
        out["m2_yoy_pct"] = (m2 / m2.shift(365) - 1) * 100

    # Fed balance sheet change (QE/QT indicator)
    if "fed_balance_sheet" in daily:
        bs = daily["fed_balance_sheet"]
        out["fed_bs_chg_30d"] = (bs / bs.shift(30) - 1) * 100

    # CPI trend
    if "cpi" in daily:
        cpi = daily["cpi"]
        out["cpi_yoy_pct"] = (cpi / cpi.shift(365) - 1) * 100

    # USD strength
    if "usd_index" in daily:
        out["usd_chg_5d"] = daily["usd_index"].pct_change(5) * 100

    return out.dropna(how="all")


def main():
    parser = argparse.ArgumentParser(description="Download FRED macro data")
    parser.add_argument("--start", default="2019-01-01")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    features = build_fred_features(args.start)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(OUTPUT_PATH)
    log.info("Saved %d rows to %s", len(features), OUTPUT_PATH)

    print(f"\nFRED macro features: {len(features)} days")
    print(f"Columns: {list(features.columns)}")
    print(f"Range: {features.index[0].date()} → {features.index[-1].date()}")
    for col in features.columns:
        nn = features[col].notna().sum()
        last = features[col].dropna().iloc[-1] if nn > 0 else None
        print(f"  {col:>20}: {nn} values, latest={last}")


if __name__ == "__main__":
    main()
