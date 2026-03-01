#!/usr/bin/env python3
"""Download historical Open Interest data from Binance Futures API.

API: GET /futures/data/openInterestHist?symbol=BTCUSDT&period=1h&limit=500
Returns OI at 1h intervals.

Usage:
    python3 -m scripts.download_open_interest
    python3 -m scripts.download_open_interest --symbols BTCUSDT ETHUSDT
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List
from urllib.request import urlopen

BASE_URL = "https://fapi.binance.com"
LIMIT = 500  # max per request


def download_oi(symbol: str) -> List[dict]:
    """Download all available OI records for a symbol.

    Binance /futures/data/openInterestHist only keeps ~30 days of history.
    We first fetch without startTime, then paginate forward from the last record.
    """
    all_records: List[dict] = []

    # First request: no startTime to get earliest available
    url = f"{BASE_URL}/futures/data/openInterestHist?symbol={symbol}&period=1h&limit={LIMIT}"
    try:
        with urlopen(url, timeout=30) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        print(f"  Error fetching {symbol}: {e}")
        return all_records

    if not data:
        return all_records

    all_records.extend(data)
    first_dt = datetime.fromtimestamp(data[0]["timestamp"] / 1000, tz=timezone.utc)
    last_dt = datetime.fromtimestamp(data[-1]["timestamp"] / 1000, tz=timezone.utc)
    print(f"  {symbol}: {len(all_records)} records ({first_dt.strftime('%Y-%m-%d')} to {last_dt.strftime('%Y-%m-%d %H:%M')})")

    # Paginate forward if we got a full page
    while len(data) == LIMIT:
        cursor = data[-1]["timestamp"] + 1
        url = (
            f"{BASE_URL}/futures/data/openInterestHist"
            f"?symbol={symbol}&period=1h&limit={LIMIT}&startTime={cursor}"
        )
        try:
            with urlopen(url, timeout=30) as resp:
                data = json.loads(resp.read().decode())
        except Exception as e:
            print(f"  Error paginating {symbol}: {e}")
            break

        if not data:
            break

        all_records.extend(data)
        last_dt = datetime.fromtimestamp(data[-1]["timestamp"] / 1000, tz=timezone.utc)
        print(f"  {symbol}: {len(all_records)} records (to {last_dt.strftime('%Y-%m-%d %H:%M')})")
        time.sleep(0.5)

    return all_records


def save_csv(records: List[dict], path: Path) -> None:
    """Save records, merging with existing CSV to avoid duplicates."""
    path.parent.mkdir(parents=True, exist_ok=True)

    existing_ts: set = set()
    if path.exists():
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_ts.add(int(row["timestamp"]))

    new_records = [r for r in records if r["timestamp"] not in existing_ts]

    write_header = not path.exists() or not existing_ts
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["timestamp", "symbol", "sum_open_interest", "sum_open_interest_value"])
        for r in new_records:
            writer.writerow([
                r["timestamp"],
                r["symbol"],
                r["sumOpenInterest"],
                r["sumOpenInterestValue"],
            ])
    print(f"  Saved {len(new_records)} new records to {path} ({len(existing_ts)} existing)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Binance Open Interest")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    parser.add_argument("--out", default="data_files", help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out)

    for symbol in args.symbols:
        print(f"\nDownloading OI for {symbol}...")
        records = download_oi(symbol)
        if records:
            save_csv(records, out_dir / f"{symbol}_open_interest.csv")
        else:
            print(f"  No records for {symbol}")


if __name__ == "__main__":
    main()
