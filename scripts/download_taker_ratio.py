"""Download Taker Buy/Sell Ratio and Top Trader Position Ratio from Binance.

Reuses the merge-with-existing pattern from download_ls_ratio.py.
These endpoints only keep ~30 days of data, so this script is for accumulation.

Usage:
    python3 -m scripts.download_taker_ratio
    python3 -m scripts.download_taker_ratio --symbols BTCUSDT ETHUSDT
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
LIMIT = 500


def _download_ratio(symbol: str, endpoint: str, label: str) -> List[dict]:
    """Download ratio records from a Binance /futures/data/ endpoint."""
    all_records: List[dict] = []

    url = f"{BASE_URL}{endpoint}?symbol={symbol}&period=1h&limit={LIMIT}"
    try:
        with urlopen(url, timeout=30) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        print(f"  Error fetching {label} for {symbol}: {e}")
        return all_records

    if not data:
        return all_records

    all_records.extend(data)
    first_dt = datetime.fromtimestamp(data[0]["timestamp"] / 1000, tz=timezone.utc)
    last_dt = datetime.fromtimestamp(data[-1]["timestamp"] / 1000, tz=timezone.utc)
    print(f"  {symbol} {label}: {len(all_records)} records "
          f"({first_dt.strftime('%Y-%m-%d')} to {last_dt.strftime('%Y-%m-%d %H:%M')})")

    while len(data) == LIMIT:
        cursor = data[-1]["timestamp"] + 1
        url = f"{BASE_URL}{endpoint}?symbol={symbol}&period=1h&limit={LIMIT}&startTime={cursor}"
        try:
            with urlopen(url, timeout=30) as resp:
                data = json.loads(resp.read().decode())
        except Exception as e:
            print(f"  Error paginating {label} for {symbol}: {e}")
            break

        if not data:
            break

        all_records.extend(data)
        last_dt = datetime.fromtimestamp(data[-1]["timestamp"] / 1000, tz=timezone.utc)
        print(f"  {symbol} {label}: {len(all_records)} records "
              f"(to {last_dt.strftime('%Y-%m-%d %H:%M')})")
        time.sleep(0.5)

    return all_records


def _save_csv(records: List[dict], path: Path, columns: List[str], key_map: dict) -> None:
    """Save records with merge-with-existing pattern."""
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
            writer.writerow(columns)
        for r in new_records:
            writer.writerow([r.get(key_map.get(c, c), "") for c in columns])

    print(f"  Saved {len(new_records)} new records to {path} ({len(existing_ts)} existing)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Taker & Top Trader Ratios")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    parser.add_argument("--out", default="data_files", help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out)

    for symbol in args.symbols:
        # Taker Buy/Sell Ratio
        print(f"\nDownloading taker ratio for {symbol}...")
        taker_records = _download_ratio(
            symbol, "/futures/data/takerlongshortRatio", "taker_ratio")
        if taker_records:
            _save_csv(
                taker_records,
                out_dir / f"{symbol}_taker_ratio.csv",
                columns=["timestamp", "symbol", "buy_sell_ratio", "buy_vol", "sell_vol"],
                key_map={
                    "buy_sell_ratio": "buySellRatio",
                    "buy_vol": "buyVol",
                    "sell_vol": "sellVol",
                },
            )

        time.sleep(0.5)

        # Top Trader Position Ratio
        print(f"Downloading top trader ratio for {symbol}...")
        top_records = _download_ratio(
            symbol, "/futures/data/topLongShortPositionRatio", "top_trader")
        if top_records:
            _save_csv(
                top_records,
                out_dir / f"{symbol}_top_trader_ratio.csv",
                columns=["timestamp", "symbol", "long_short_ratio", "long_account", "short_account"],
                key_map={
                    "long_short_ratio": "longShortRatio",
                    "long_account": "longAccount",
                    "short_account": "shortAccount",
                },
            )


if __name__ == "__main__":
    main()
