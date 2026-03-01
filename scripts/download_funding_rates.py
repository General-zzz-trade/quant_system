#!/usr/bin/env python3
"""Download historical funding rates from Binance Futures API.

Binance settles funding every 8h (00:00, 08:00, 16:00 UTC).
API: GET /fapi/v1/fundingRate?symbol=BTCUSDT&limit=1000&startTime=xxx

Usage:
    python3 -m scripts.download_funding_rates
    python3 -m scripts.download_funding_rates --symbols BTCUSDT ETHUSDT
"""
from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List
from urllib.request import urlopen
import json

BASE_URL = "https://fapi.binance.com"
LIMIT = 1000  # max per request


def download_funding(symbol: str, start_ms: int = 1577836800000) -> List[dict]:
    """Download all funding rate records for a symbol from start_ms.

    Default start: 2020-01-01 00:00 UTC.
    """
    all_records: List[dict] = []
    cursor = start_ms

    while True:
        url = (
            f"{BASE_URL}/fapi/v1/fundingRate"
            f"?symbol={symbol}&limit={LIMIT}&startTime={cursor}"
        )
        try:
            with urlopen(url, timeout=30) as resp:
                data = json.loads(resp.read().decode())
        except Exception as e:
            print(f"  Error fetching {symbol} at {cursor}: {e}")
            time.sleep(5)
            continue

        if not data:
            break

        all_records.extend(data)
        last_ts = data[-1]["fundingTime"]
        print(f"  {symbol}: {len(all_records)} records (last: {datetime.fromtimestamp(last_ts/1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M')})")

        if len(data) < LIMIT:
            break

        cursor = last_ts + 1
        time.sleep(0.2)  # rate limit

    return all_records


def save_csv(records: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "symbol", "funding_rate", "mark_price"])
        for r in records:
            writer.writerow([
                r["fundingTime"],
                r["symbol"],
                r["fundingRate"],
                r.get("markPrice", "0"),
            ])
    print(f"  Saved {len(records)} records to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Binance funding rates")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    parser.add_argument("--out", default="data_files", help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out)

    for symbol in args.symbols:
        print(f"\nDownloading funding rates for {symbol}...")
        records = download_funding(symbol)
        if records:
            save_csv(records, out_dir / f"{symbol}_funding.csv")
        else:
            print(f"  No records for {symbol}")


if __name__ == "__main__":
    main()
