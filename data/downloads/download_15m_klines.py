#!/usr/bin/env python3
"""Download 15m kline data from Binance Futures for ETHUSDT and SOLUSDT.

Incremental: reads the last timestamp from existing CSV and only fetches new bars.
If no CSV exists, downloads from 2024-01-01 onward.

Usage:
    python3 -m data.downloads.download_15m_klines
    python3 -m data.downloads.download_15m_klines --symbols ETHUSDT
    python3 -m data.downloads.download_15m_klines --symbols ETHUSDT,SOLUSDT,BTCUSDT
    python3 -m data.downloads.download_15m_klines --full   # re-download everything
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import HTTPError

BASE_URL = "https://fapi.binance.com/fapi/v1/klines"
INTERVAL = "15m"
LIMIT = 1500
INTERVAL_MS = 15 * 60 * 1000  # 900_000 ms per bar
DEFAULT_START_MS = 1704067200000  # 2024-01-01 00:00:00 UTC
DEFAULT_SYMBOLS = ["ETHUSDT", "SOLUSDT"]
DATA_DIR = Path("/quant_system/data_files")

COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades",
    "taker_buy_volume", "taker_buy_quote_volume", "ignore",
]


def _ts_str(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def _read_last_timestamp(csv_path: Path) -> int | None:
    """Read the last open_time from an existing CSV file."""
    if not csv_path.exists():
        return None
    last_line = None
    with open(csv_path, "r") as f:
        for line in f:
            stripped = line.strip()
            if stripped and not stripped.startswith("open_time"):
                last_line = stripped
    if last_line is None:
        return None
    try:
        return int(last_line.split(",")[0])
    except (ValueError, IndexError):
        return None


def _fetch_with_retry(url: str, max_retries: int = 5) -> list:
    """Fetch JSON from URL with exponential backoff retry."""
    for attempt in range(max_retries):
        try:
            req = Request(url)
            req.add_header("User-Agent", "Mozilla/5.0")
            with urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode())
        except HTTPError as e:
            if e.code == 429:
                wait = 2 ** (attempt + 2)
                print(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
            elif attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"  HTTP {e.code}, retry {attempt+1} in {wait}s...")
                time.sleep(wait)
            else:
                raise
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"  Error: {e}, retry {attempt+1} in {wait}s...")
                time.sleep(wait)
            else:
                raise
    return []


def download_symbol(symbol: str, *, full: bool = False) -> int:
    """Download 15m klines for a single symbol. Returns number of new bars."""
    csv_path = DATA_DIR / f"{symbol}_15m.csv"
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Determine start time
    if full or not csv_path.exists():
        start_ms = DEFAULT_START_MS
        existing_rows = []
        print(f"  Full download from {_ts_str(start_ms)}")
    else:
        last_ts = _read_last_timestamp(csv_path)
        if last_ts is None:
            start_ms = DEFAULT_START_MS
            existing_rows = []
            print(f"  No valid data found, full download from {_ts_str(start_ms)}")
        else:
            # Start from next bar after last known
            start_ms = last_ts + INTERVAL_MS
            # Read existing rows for merge
            existing_rows = []
            seen_times = set()
            with open(csv_path, "r") as f:
                reader = csv.reader(f)
                next(reader, None)  # skip header
                for row in reader:
                    if row:
                        ot = int(row[0])
                        if ot not in seen_times:
                            seen_times.add(ot)
                            existing_rows.append(row)
            print(f"  Incremental from {_ts_str(start_ms)} ({len(existing_rows):,} existing bars)")

    now_ms = int(time.time() * 1000)
    if start_ms >= now_ms:
        print("  Already up to date.")
        return 0

    # Fetch new bars
    new_rows = []
    current = start_ms
    batch = 0

    while current < now_ms:
        url = f"{BASE_URL}?symbol={symbol}&interval={INTERVAL}&limit={LIMIT}&startTime={current}"
        data = _fetch_with_retry(url)

        if not data:
            break

        batch += 1
        for candle in data:
            new_rows.append([str(v) for v in candle[:12]])

        last_close_time = data[-1][6]
        current = last_close_time + 1

        if batch % 10 == 0:
            print(f"    Batch {batch}: {len(new_rows):,} new bars, up to {_ts_str(data[-1][0])}")

        if len(data) < LIMIT:
            break

        # Stay under rate limit (1200 req/min weight)
        time.sleep(0.15)

    if not new_rows and not full:
        print("  No new bars available.")
        return 0

    # Merge and deduplicate
    all_rows = existing_rows + new_rows
    seen = set()
    unique = []
    for row in all_rows:
        ot = row[0]
        if ot not in seen:
            seen.add(ot)
            unique.append(row)
    unique.sort(key=lambda r: int(r[0]))

    # Write CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)
        for row in unique:
            writer.writerow(row)

    print(f"  Saved {len(unique):,} total bars to {csv_path}")
    print(f"  Range: {_ts_str(int(unique[0][0]))} -> {_ts_str(int(unique[-1][0]))}")
    print(f"  New bars: {len(new_rows):,}")

    return len(new_rows)


def main():
    parser = argparse.ArgumentParser(description="Download 15m klines from Binance Futures")
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS),
                        help=f"Comma-separated symbols (default: {','.join(DEFAULT_SYMBOLS)})")
    parser.add_argument("--full", action="store_true",
                        help="Re-download everything (ignore existing data)")
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",")]

    print("=" * 60)
    print("  15m KLINE DOWNLOAD")
    print(f"  Symbols: {symbols}")
    print(f"  Mode:    {'full' if args.full else 'incremental'}")
    print(f"  Time:    {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 60)

    total_new = 0
    for symbol in symbols:
        print(f"\n  {symbol}:")
        try:
            n = download_symbol(symbol, full=args.full)
            total_new += n
        except Exception as e:
            print(f"  ERROR: {e}")

    print(f"\n  Done. Total new bars: {total_new:,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
