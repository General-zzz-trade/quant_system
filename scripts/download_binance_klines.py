"""Download Binance BTCUSDT perpetual futures historical klines."""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from urllib.request import urlopen, Request

BASE_URL = "https://fapi.binance.com"
ENDPOINT = "/fapi/v1/klines"
LIMIT = 1500  # max per request


def fetch_klines(symbol: str, interval: str, start_time: int, limit: int = LIMIT) -> list:
    url = f"{BASE_URL}{ENDPOINT}?symbol={symbol}&interval={interval}&startTime={start_time}&limit={limit}"
    req = Request(url)
    req.add_header("User-Agent", "Mozilla/5.0")
    with urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def interval_ms(interval: str) -> int:
    """Convert interval string to milliseconds."""
    units = {"m": 60_000, "h": 3_600_000, "d": 86_400_000}
    return int(interval[:-1]) * units[interval[-1]]


def download_all(symbol: str, interval: str, output_path: str):
    # BTCUSDT perp started 2019-09-08
    start_ms = 1567900800000  # 2019-09-08 00:00:00 UTC
    step = interval_ms(interval) * LIMIT

    all_rows = []
    current = start_ms
    now_ms = int(time.time() * 1000)

    print(f"Downloading {symbol} {interval} klines from 2019-09-08 to now...")

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

        # progress
        pct = min(100, (last_open_time - start_ms) / max(1, now_ms - start_ms) * 100)
        print(f"  Fetched {len(all_rows):>7,} bars | up to {_ts_str(last_open_time)} | {pct:.1f}%")

        if len(data) < LIMIT:
            break

        # next batch starts after last bar
        current = last_open_time + interval_ms(interval)
        time.sleep(0.2)  # be polite

    # deduplicate by open_time
    seen = set()
    unique = []
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


def _ts_str(ms: int) -> str:
    from datetime import datetime, timezone
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


if __name__ == "__main__":
    symbol = sys.argv[1] if len(sys.argv) > 1 else "BTCUSDT"
    interval = sys.argv[2] if len(sys.argv) > 2 else "1h"
    out = sys.argv[3] if len(sys.argv) > 3 else f"data_files/{symbol}_{interval}.csv"
    download_all(symbol, interval, out)
