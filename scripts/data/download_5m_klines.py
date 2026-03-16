#!/usr/bin/env python3
"""Download 5m kline data from Binance Futures.

Incremental: reads the last timestamp from existing CSV and only fetches new bars.
If no CSV exists, downloads from 2024-01-01 onward.

Usage:
    python3 -m scripts.data.download_5m_klines
    python3 -m scripts.data.download_5m_klines --symbols ETHUSDT
    python3 -m scripts.data.download_5m_klines --start 2023-01-01
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen, Request

BASE_URL = "https://fapi.binance.com/fapi/v1/klines"
INTERVAL = "5m"
LIMIT = 1500
INTERVAL_MS = 5 * 60 * 1000  # 300_000 ms per bar
DEFAULT_START_MS = 1704067200000  # 2024-01-01 00:00:00 UTC
DEFAULT_SYMBOLS = ["ETHUSDT"]
DATA_DIR = Path("/quant_system/data_files")

COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades",
    "taker_buy_volume", "taker_buy_quote_volume", "ignore",
]


def _ts_str(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def _read_last_timestamp(csv_path: Path) -> int | None:
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


def download_symbol(symbol: str, start_ms: int | None = None) -> int:
    csv_path = DATA_DIR / f"{symbol}_5m.csv"
    last_ts = _read_last_timestamp(csv_path)
    is_new = last_ts is None

    if start_ms and is_new:
        cursor = start_ms
    elif last_ts is not None:
        cursor = last_ts + INTERVAL_MS
    else:
        cursor = DEFAULT_START_MS

    now_ms = int(time.time() * 1000)
    if cursor >= now_ms:
        print(f"  {symbol}: already up to date ({_ts_str(last_ts or cursor)})")
        return 0

    mode = "a" if not is_new else "w"
    total = 0

    print(f"  {symbol}: downloading from {_ts_str(cursor)} {'(new file)' if is_new else '(incremental)'}...")

    with open(csv_path, mode, newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(COLUMNS)

        while cursor < now_ms:
            url = f"{BASE_URL}?symbol={symbol}&interval={INTERVAL}&startTime={cursor}&limit={LIMIT}"
            req = Request(url, headers={"Accept": "application/json"})
            try:
                with urlopen(req, timeout=15) as resp:
                    bars = json.loads(resp.read())
            except Exception as e:
                print(f"    Error at {_ts_str(cursor)}: {e}, retrying...")
                time.sleep(2)
                try:
                    with urlopen(req, timeout=15) as resp:
                        bars = json.loads(resp.read())
                except Exception as e2:
                    print(f"    Retry failed: {e2}, stopping")
                    break

            if not bars:
                break

            for bar in bars:
                writer.writerow(bar[:len(COLUMNS)])
            total += len(bars)

            last_bar_ts = int(bars[-1][0])
            cursor = last_bar_ts + INTERVAL_MS

            if total % 15000 == 0:
                print(f"    {total:,} bars... ({_ts_str(last_bar_ts)})")

            time.sleep(0.3)

    print(f"  {symbol}: {total:,} new bars → {csv_path.name}")
    return total


def main():
    parser = argparse.ArgumentParser(description="Download 5m kline data from Binance")
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    parser.add_argument("--start", type=str, default=None, help="Start date (YYYY-MM-DD)")
    args = parser.parse_args()

    start_ms = None
    if args.start:
        dt = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        start_ms = int(dt.timestamp() * 1000)

    print(f"Downloading 5m klines for {args.symbols}")
    for symbol in args.symbols:
        download_symbol(symbol, start_ms)


if __name__ == "__main__":
    main()
