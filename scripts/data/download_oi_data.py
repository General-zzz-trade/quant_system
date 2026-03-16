#!/usr/bin/env python3
"""Download OI, Long/Short ratio, Taker buy/sell from Binance Futures API.

Supports incremental download with pagination (max 500 per request).
Stores aligned 1h data as CSV in data_files/.

Usage:
    python3 -m scripts.data.download_oi_data --symbols ETHUSDT BTCUSDT
    python3 -m scripts.data.download_oi_data --symbols ETHUSDT --days 365
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE_URL = "https://fapi.binance.com/futures/data"
RATE_LIMIT_SLEEP = 0.5  # seconds between requests


def _fetch_paginated(endpoint: str, symbol: str, period: str = "1h",
                     start_ms: int = 0, end_ms: int = 0,
                     limit: int = 500) -> list[dict]:
    """Fetch paginated data from Binance futures data API.

    Uses backward pagination with endTime (Binance doesn't support startTime
    for these endpoints beyond ~30 days).
    """
    all_data: list[dict] = []
    current_end = end_ms if end_ms > 0 else int(time.time() * 1000)

    while True:
        params = f"symbol={symbol}&period={period}&limit={limit}"
        params += f"&endTime={current_end}"

        url = f"{BASE_URL}/{endpoint}?{params}"
        req = Request(url, headers={"Accept": "application/json"})

        try:
            with urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read())
        except Exception as e:
            logger.warning("Request failed: %s, retrying...", e)
            time.sleep(2)
            try:
                with urlopen(req, timeout=15) as resp:
                    data = json.loads(resp.read())
            except Exception as e2:
                logger.error("Retry failed: %s", e2)
                break

        if not data or not isinstance(data, list):
            break

        all_data.extend(data)

        if len(data) < limit:
            break  # no more data available

        # Next page ends before first timestamp of current batch
        first_ts = int(data[0].get("timestamp", 0))
        if start_ms > 0 and first_ts <= start_ms:
            break
        current_end = first_ts - 1

        time.sleep(RATE_LIMIT_SLEEP)

    # Deduplicate and sort ascending
    seen = set()
    unique = []
    for d in all_data:
        ts = d.get("timestamp")
        if ts not in seen:
            seen.add(ts)
            unique.append(d)
    unique.sort(key=lambda x: x.get("timestamp", 0))
    return unique


def download_oi(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Download Open Interest history."""
    data = _fetch_paginated("openInterestHist", symbol, "1h", start_ms, end_ms)
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_numeric(df["timestamp"])
    df["oi"] = pd.to_numeric(df["sumOpenInterest"])
    df["oi_value"] = pd.to_numeric(df["sumOpenInterestValue"])
    return df[["timestamp", "oi", "oi_value"]].drop_duplicates("timestamp").sort_values("timestamp")


def download_ls_ratio(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Download global Long/Short account ratio."""
    data = _fetch_paginated("globalLongShortAccountRatio", symbol, "1h", start_ms, end_ms)
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_numeric(df["timestamp"])
    df["ls_ratio"] = pd.to_numeric(df["longShortRatio"])
    df["long_pct"] = pd.to_numeric(df["longAccount"])
    df["short_pct"] = pd.to_numeric(df["shortAccount"])
    return df[["timestamp", "ls_ratio", "long_pct", "short_pct"]].drop_duplicates("timestamp").sort_values("timestamp")


def download_taker_ratio(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Download taker buy/sell volume ratio."""
    data = _fetch_paginated("takerlongshortRatio", symbol, "1h", start_ms, end_ms)
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_numeric(df["timestamp"])
    df["taker_ratio"] = pd.to_numeric(df["buySellRatio"])
    df["taker_buy_vol"] = pd.to_numeric(df["buyVol"])
    df["taker_sell_vol"] = pd.to_numeric(df["sellVol"])
    return df[["timestamp", "taker_ratio", "taker_buy_vol",
        "taker_sell_vol"]].drop_duplicates("timestamp").sort_values("timestamp")


def download_top_trader_ls(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Download top trader position long/short ratio."""
    data = _fetch_paginated("topLongShortPositionRatio", symbol, "1h", start_ms, end_ms)
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_numeric(df["timestamp"])
    df["top_ls_ratio"] = pd.to_numeric(df["longShortRatio"])
    df["top_long_pct"] = pd.to_numeric(df["longAccount"])
    df["top_short_pct"] = pd.to_numeric(df["shortAccount"])
    return df[["timestamp", "top_ls_ratio", "top_long_pct",
        "top_short_pct"]].drop_duplicates("timestamp").sort_values("timestamp")


def download_all(symbol: str, days: int = 365) -> pd.DataFrame:
    """Download all OI-related data and merge into single DataFrame."""
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - days * 86400 * 1000

    logger.info("Downloading %s OI data: %d days", symbol, days)

    oi = download_oi(symbol, start_ms, end_ms)
    logger.info("  OI: %d records", len(oi))

    ls = download_ls_ratio(symbol, start_ms, end_ms)
    logger.info("  L/S ratio: %d records", len(ls))

    taker = download_taker_ratio(symbol, start_ms, end_ms)
    logger.info("  Taker ratio: %d records", len(taker))

    top_ls = download_top_trader_ls(symbol, start_ms, end_ms)
    logger.info("  Top trader L/S: %d records", len(top_ls))

    # Merge all on timestamp
    if oi.empty:
        logger.error("No OI data available")
        return pd.DataFrame()

    merged = oi
    if not ls.empty:
        merged = merged.merge(ls, on="timestamp", how="outer")
    if not taker.empty:
        merged = merged.merge(taker, on="timestamp", how="outer")
    if not top_ls.empty:
        merged = merged.merge(top_ls, on="timestamp", how="outer")

    merged = merged.sort_values("timestamp").reset_index(drop=True)

    # Add datetime column
    merged["datetime"] = pd.to_datetime(merged["timestamp"], unit="ms", utc=True)

    logger.info("  Merged: %d records, %s -> %s",
                len(merged),
                merged["datetime"].iloc[0].strftime("%Y-%m-%d"),
                merged["datetime"].iloc[-1].strftime("%Y-%m-%d"))

    return merged


def main():
    parser = argparse.ArgumentParser(description="Download OI/LS/Taker data from Binance")
    parser.add_argument("--symbols", nargs="+", default=["ETHUSDT", "BTCUSDT"])
    parser.add_argument("--days", type=int, default=730, help="Days of history to download")
    parser.add_argument("--output-dir", default="data_files", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for symbol in args.symbols:
        df = download_all(symbol, args.days)
        if df.empty:
            logger.warning("No data for %s", symbol)
            continue

        # Save
        out_path = output_dir / f"{symbol}_oi_1h.csv"

        # Incremental: merge with existing
        if out_path.exists():
            existing = pd.read_csv(out_path)
            existing["timestamp"] = pd.to_numeric(existing["timestamp"])
            combined = pd.concat([existing, df]).drop_duplicates("timestamp").sort_values("timestamp")
            combined.to_csv(out_path, index=False)
            new_count = len(combined) - len(existing)
            logger.info("  %s: updated %s (%d new, %d total)",
                        symbol, out_path, new_count, len(combined))
        else:
            df.to_csv(out_path, index=False)
            logger.info("  %s: saved %s (%d records)", symbol, out_path, len(df))


if __name__ == "__main__":
    main()
