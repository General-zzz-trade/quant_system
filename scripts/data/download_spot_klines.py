"""Download Binance Spot historical klines.

Same pattern as download_binance_klines.py but uses the spot API endpoint.

Usage:
    python scripts/download_spot_klines.py BTCUSDT 1h data_files/BTCUSDT_spot_1h.csv
    python scripts/download_spot_klines.py --batch BTCUSDT,ETHUSDT,SOLUSDT
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from datetime import datetime, timezone
from urllib.request import urlopen, Request

BASE_URL = "https://api.binance.com"
ENDPOINT = "/api/v3/klines"
LIMIT = 1000  # spot API max per request


def fetch_klines(symbol: str, interval: str, start_time: int, limit: int = LIMIT) -> list:
    url = f"{BASE_URL}{ENDPOINT}?symbol={symbol}&interval={interval}&startTime={start_time}&limit={limit}"
    req = Request(url)
    req.add_header("User-Agent", "Mozilla/5.0")
    with urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def interval_ms(interval: str) -> int:
    units = {"m": 60_000, "h": 3_600_000, "d": 86_400_000}
    return int(interval[:-1]) * units[interval[-1]]


def _ts_str(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def _parse_date_ms(date_str: str) -> int:
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def download_all(
    symbol: str, interval: str, output_path: str, *, start_ms: int | None = None,
) -> int:
    if start_ms is None:
        start_ms = 1567900800000  # 2019-09-08

    all_rows: list = []
    current = start_ms
    now_ms = int(time.time() * 1000)

    print(f"Downloading {symbol} spot {interval} klines from {_ts_str(start_ms)} to now...")

    while current < now_ms:
        try:
            data = fetch_klines(symbol, interval, current, LIMIT)
        except Exception as e:
            print(f"  Error at {current}: {e}, retrying in 3s...")
            time.sleep(3)
            continue

        if not data:
            break

        all_rows.extend(data)
        last_open_time = data[-1][0]

        pct = min(100, (last_open_time - start_ms) / max(1, now_ms - start_ms) * 100)
        print(f"  Fetched {len(all_rows):>7,} bars | up to {_ts_str(last_open_time)} | {pct:.1f}%")

        if len(data) < LIMIT:
            break

        current = last_open_time + interval_ms(interval)
        time.sleep(0.2)

    # deduplicate by open_time
    seen: set = set()
    unique: list = []
    for row in all_rows:
        if row[0] not in seen:
            seen.add(row[0])
            unique.append(row)
    unique.sort(key=lambda r: r[0])

    # write CSV
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["open_time", "open", "high", "low", "close", "volume",
                     "close_time", "quote_volume", "trades", "taker_buy_volume",
                     "taker_buy_quote_volume", "ignore"])
        for row in unique:
            w.writerow(row)

    print(f"\nDone! {len(unique):,} bars saved to {output_path}")
    return len(unique)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Binance Spot klines")
    parser.add_argument("symbol", nargs="?", help="Symbol (e.g. BTCUSDT)")
    parser.add_argument("interval", nargs="?", default="1h", help="Interval (default: 1h)")
    parser.add_argument("output", nargs="?", help="Output CSV path")
    parser.add_argument("--start", help="Start date YYYY-MM-DD")
    parser.add_argument("--batch", help="Comma-separated symbols for batch download")
    parser.add_argument("--interval", dest="interval_flag", default="1h", help="Interval for batch mode")
    args = parser.parse_args()

    start_ms = _parse_date_ms(args.start) if args.start else None

    if args.batch:
        symbols = [s.strip() for s in args.batch.split(",")]
        interval = args.interval_flag
        for sym in symbols:
            out = f"data_files/{sym}_spot_{interval}.csv"
            download_all(sym, interval, out, start_ms=start_ms)
            print()
    elif args.symbol:
        out = args.output or f"data_files/{args.symbol}_spot_{args.interval}.csv"
        download_all(args.symbol, args.interval, out, start_ms=start_ms)
    else:
        parser.print_help()
