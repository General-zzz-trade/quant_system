"""Download Bitcoin Fear & Greed Index full history.

API: GET https://api.alternative.me/fng/?limit=0&format=json

Usage:
    python scripts/download_fear_greed.py
    python scripts/download_fear_greed.py --out data_files/fear_greed_index.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from urllib.request import urlopen, Request


API_URL = "https://api.alternative.me/fng/?limit=0&format=json"


def download_fgi(output_path: str) -> int:
    print("Downloading Fear & Greed Index full history...")

    req = Request(API_URL)
    req.add_header("User-Agent", "Mozilla/5.0")
    with urlopen(req, timeout=30) as resp:
        result = json.loads(resp.read())

    data = result.get("data", [])
    if not data:
        print("  No data returned")
        return 0

    # API returns newest first, each entry: {"value": "73", "value_classification": "Greed", "timestamp": "1677628800"}
    # timestamp is UNIX seconds → convert to ms, align to UTC 00:00
    records = []
    for entry in data:
        ts_sec = int(entry["timestamp"])
        ts_ms = ts_sec * 1000
        value = int(entry["value"])
        classification = entry["value_classification"]
        records.append((ts_ms, value, classification))

    # Sort by timestamp ascending
    records.sort(key=lambda r: r[0])

    # Deduplicate by timestamp
    seen: set = set()
    unique = []
    for r in records:
        if r[0] not in seen:
            seen.add(r[0])
            unique.append(r)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "value", "classification"])
        for r in unique:
            w.writerow(r)

    print(f"Done! {len(unique):,} days saved to {output_path}")
    print(f"  Range: {unique[0][0]} to {unique[-1][0]}")
    return len(unique)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Fear & Greed Index")
    parser.add_argument("--out", default="data_files/fear_greed_index.csv",
                        help="Output CSV path")
    args = parser.parse_args()
    download_fgi(args.out)
