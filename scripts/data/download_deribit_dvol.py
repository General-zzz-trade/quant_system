"""Download full Deribit DVOL history (hourly) for BTC and ETH.

DVOL = Deribit Volatility Index, hourly OHLC from 2022-01-01.
Paginated in 7-day chunks to avoid API limits.

Usage:
    python3 -m scripts.data.download_deribit_dvol --currency BTC
    python3 -m scripts.data.download_deribit_dvol --currency ETH
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
import urllib.request
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

log = logging.getLogger("deribit_dvol")

BASE_URL = "https://www.deribit.com/api/v2/public"
CHUNK_DAYS = 7
RATE_LIMIT_S = 0.5  # 0.5s between requests


def download_dvol(currency: str, start_date: str = "2022-01-01") -> list[dict]:
    """Download all hourly DVOL data from start_date to now."""
    start = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = datetime.now(timezone.utc)

    all_bars = []
    cursor = start

    while cursor < end:
        chunk_end = min(cursor + timedelta(days=CHUNK_DAYS), end)
        start_ms = int(cursor.timestamp() * 1000)
        end_ms = int(chunk_end.timestamp() * 1000)

        url = (f"{BASE_URL}/get_volatility_index_data?"
               f"currency={currency}&resolution=3600"
               f"&start_timestamp={start_ms}&end_timestamp={end_ms}")

        try:
            resp = json.loads(urllib.request.urlopen(url, timeout=15).read())
            data = resp.get("result", {}).get("data", [])

            for bar in data:
                # bar = [timestamp, open, high, low, close]
                all_bars.append({
                    "timestamp": bar[0],
                    "datetime": datetime.fromtimestamp(bar[0]/1000, tz=timezone.utc).isoformat(),
                    "open": bar[1],
                    "high": bar[2],
                    "low": bar[3],
                    "close": bar[4],
                })

            log.info("%s DVOL %s → %s: %d bars (total %d)",
                     currency, cursor.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d"),
                     len(data), len(all_bars))
        except Exception as e:
            log.warning("Failed %s → %s: %s", cursor.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d"), e)

        cursor = chunk_end
        time.sleep(RATE_LIMIT_S)

    return all_bars


def save_csv(bars: list[dict], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "datetime", "open", "high", "low", "close"])
        writer.writeheader()
        writer.writerows(bars)
    log.info("Saved %d bars to %s", len(bars), path)


def main():
    parser = argparse.ArgumentParser(description="Download Deribit DVOL history")
    parser.add_argument("--currency", default="BTC", choices=["BTC", "ETH"])
    parser.add_argument("--start", default="2022-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--out-dir", default="data_files")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    log.info("Downloading %s DVOL from %s", args.currency, args.start)
    bars = download_dvol(args.currency, args.start)

    if bars:
        symbol = f"{args.currency}USDT"
        path = os.path.join(args.out_dir, f"{symbol}_dvol_1h.csv")
        save_csv(bars, path)

        d0 = bars[0]["datetime"][:10]
        d1 = bars[-1]["datetime"][:10]
        print(f"\n{args.currency} DVOL: {len(bars)} bars, {d0} → {d1}")
        print(f"  Saved to: {path}")
    else:
        print("No data downloaded")


if __name__ == "__main__":
    main()
