#!/usr/bin/env python3
"""Download on-chain metrics from Coin Metrics Community API.

Downloads: FlowInExUSD, FlowOutExUSD, SplyExNtv, AdrActCnt, TxTfrCnt, HashRate
Output: one CSV per metric in data/onchain/ (or --out-dir)

Usage:
    python scripts/download_onchain_metrics.py --asset btc --start 2020-01-01
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import time
import urllib.request
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

_API_BASE = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
_METRICS = ["FlowInExUSD", "FlowOutExUSD", "SplyExNtv", "AdrActCnt", "TxTfrCnt", "HashRate"]
_PAGE_SIZE = 10000
_RATE_LIMIT_SLEEP = 0.7  # 10 req / 6s → ~0.6s per req, add margin


def _fetch_page(asset: str, metrics: str, start: str, next_page_token: str | None = None) -> dict:
    url = (
        f"{_API_BASE}?assets={asset}&metrics={metrics}"
        f"&frequency=1d&page_size={_PAGE_SIZE}&start_time={start}&sort=time"
    )
    if next_page_token:
        url += f"&next_page_token={next_page_token}"

    req = urllib.request.Request(url)
    req.add_header("User-Agent", "quant-system/1.0")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def download_metrics(asset: str, start: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_str = ",".join(_METRICS)

    # Collect all rows (paginated)
    all_rows: list[dict] = []
    next_token: str | None = None
    page = 0

    while True:
        page += 1
        logger.info("Fetching page %d (rows so far: %d)...", page, len(all_rows))
        data = _fetch_page(asset, metrics_str, start, next_token)
        rows = data.get("data", [])
        all_rows.extend(rows)

        next_token = data.get("next_page_token")
        if not next_token or not rows:
            break
        time.sleep(_RATE_LIMIT_SLEEP)

    logger.info("Total rows fetched: %d", len(all_rows))

    # Write one CSV per metric
    for metric in _METRICS:
        out_path = out_dir / f"{asset}_{metric}.csv"
        written = 0
        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["timestamp", "date", metric])
            writer.writeheader()
            for row in all_rows:
                val = row.get(metric)
                if val is None or val == "":
                    continue
                ts = row.get("time", "")
                # Parse ISO date to epoch ms for consistency with other schedules
                try:
                    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    ts_ms = int(dt.timestamp() * 1000)
                except (ValueError, AttributeError):
                    ts_ms = 0
                writer.writerow({
                    "timestamp": ts_ms,
                    "date": ts,
                    metric: val,
                })
                written += 1
        logger.info("Wrote %d rows to %s", written, out_path)

    # Also write a combined CSV for convenience
    combined_path = out_dir / f"{asset}_onchain_combined.csv"
    fields = ["timestamp", "date"] + _METRICS
    with combined_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in all_rows:
            ts = row.get("time", "")
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                ts_ms = int(dt.timestamp() * 1000)
            except (ValueError, AttributeError):
                ts_ms = 0
            out_row = {"timestamp": ts_ms, "date": ts}
            for m in _METRICS:
                out_row[m] = row.get(m, "")
            writer.writerow(out_row)
    logger.info("Wrote combined CSV: %s (%d rows)", combined_path, len(all_rows))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Download on-chain metrics from Coin Metrics")
    parser.add_argument("--asset", default="btc", help="Asset (default: btc)")
    parser.add_argument("--start", default="2020-01-01", help="Start date (ISO format)")
    parser.add_argument("--out-dir", default="data/onchain", help="Output directory")
    args = parser.parse_args()

    download_metrics(args.asset, args.start, Path(args.out_dir))
    logger.info("Done.")


if __name__ == "__main__":
    main()
