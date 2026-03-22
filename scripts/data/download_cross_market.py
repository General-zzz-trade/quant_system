#!/usr/bin/env python3
"""Download cross-market daily data from Yahoo Finance for BTC alpha features.

Fetches SPY, QQQ, VIX, TLT, USO, COIN daily closes and computes
derived features (returns, z-scores, extremes).

Output: data_files/cross_market_daily.csv
  Columns: date, spy_ret_1d, qqq_ret_1d, spy_ret_5d, vix_level,
           tlt_ret_5d, uso_ret_5d, coin_ret_1d, spy_extreme

Usage:
    python3 -m scripts.data.download_cross_market
    python3 -m scripts.data.download_cross_market --days 30  # last 30 days only
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import urllib.request
from pathlib import Path

import pandas as pd

log = logging.getLogger("cross_market")

TICKERS = {
    "SPY": "S&P 500",
    "QQQ": "Nasdaq 100",
    "^VIX": "VIX",
    "TLT": "US Treasury 20Y",
    "USO": "Oil ETF",
    "COIN": "Coinbase",
}

OUTPUT_PATH = Path("data_files/cross_market_daily.csv")


def fetch_yahoo(ticker: str, range_str: str = "5y") -> pd.DataFrame:
    """Fetch daily OHLCV from Yahoo Finance."""
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/"
           f"{ticker}?range={range_str}&interval=1d")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    resp = urllib.request.urlopen(req, timeout=15)
    raw = json.loads(resp.read())
    result = raw["chart"]["result"][0]
    q = result["indicators"]["quote"][0]
    df = pd.DataFrame({
        "date": pd.to_datetime(result["timestamp"], unit="s"),
        "close": q["close"],
    }).dropna(subset=["close"])
    df["date"] = df["date"].dt.date
    df = df.drop_duplicates(subset=["date"], keep="last")
    return df.set_index("date")


def build_cross_market_features() -> pd.DataFrame:
    """Download all tickers and compute cross-market features."""
    data = {}
    for ticker, name in TICKERS.items():
        try:
            df = fetch_yahoo(ticker)
            data[ticker] = df["close"]
            log.info("%s (%s): %d days", ticker, name, len(df))
        except Exception as e:
            log.warning("%s failed: %s", ticker, e)

    if "SPY" not in data:
        log.error("SPY download failed — cannot compute features")
        sys.exit(1)

    # Align all to SPY dates
    combined = pd.DataFrame(data)
    combined = combined.dropna(subset=["SPY"])

    out = pd.DataFrame(index=combined.index)
    out.index.name = "date"

    # Returns
    out["spy_ret_1d"] = combined["SPY"].pct_change()
    out["qqq_ret_1d"] = combined["QQQ"].pct_change() if "QQQ" in combined else None
    out["spy_ret_5d"] = combined["SPY"].pct_change(5)
    out["coin_ret_1d"] = combined["COIN"].pct_change() if "COIN" in combined else None

    # 5d returns
    out["tlt_ret_5d"] = combined["TLT"].pct_change(5) if "TLT" in combined else None
    out["uso_ret_5d"] = combined["USO"].pct_change(5) if "USO" in combined else None

    # VIX level (not return — absolute level matters)
    out["vix_level"] = combined["^VIX"] if "^VIX" in combined else None

    # SPY extreme flag: |ret| > 2%
    if "spy_ret_1d" in out:
        out["spy_extreme"] = 0.0
        out.loc[out["spy_ret_1d"] > 0.02, "spy_extreme"] = 1.0
        out.loc[out["spy_ret_1d"] < -0.02, "spy_extreme"] = -1.0

    return out.dropna(subset=["spy_ret_1d"])


def main():
    parser = argparse.ArgumentParser(description="Download cross-market data")
    parser.add_argument("--days", type=int, default=None,
                        help="Only keep last N days")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    features = build_cross_market_features()

    if args.days:
        features = features.tail(args.days)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(OUTPUT_PATH)
    log.info("Saved %d rows to %s", len(features), OUTPUT_PATH)

    # Summary
    print(f"\nCross-market features: {len(features)} days")
    print(f"Columns: {list(features.columns)}")
    print(f"Range: {features.index[0]} → {features.index[-1]}")


if __name__ == "__main__":
    main()
