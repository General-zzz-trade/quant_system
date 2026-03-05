#!/usr/bin/env python3
"""Download historical mempool/fee data from mempool.space API.

API: mempool.space/api/v1/mining/blocks/fee-rates/{timePeriod}
Note: Limited historical data available (mempool.space stores ~2 years).

Output: data_files/btc_mempool_fees.csv

Usage:
    python scripts/download_mempool.py
"""
from __future__ import annotations

import csv
import json
import logging
import time
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

# mempool.space block fee rates — returns fee data per block
_BLOCKS_URL = "https://mempool.space/api/v1/mining/blocks/fee-rates/3y"
# Alternative: individual block stats
_BLOCK_STATS_URL = "https://mempool.space/api/v1/fees/mempool-blocks"


def download_fee_history(out_dir: str) -> Path:
    out_path = Path(out_dir) / "btc_mempool_fees.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Try the fee-rates endpoint first (gives 3y of data)
    url = _BLOCKS_URL
    req = urllib.request.Request(url, headers={"User-Agent": "quant-system/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())

        if not data:
            logger.warning("No mempool fee data returned")
            return out_path

        rows = []
        for entry in data:
            ts = entry.get("timestamp", 0)
            avg_fee = entry.get("avgFee_50", entry.get("avgFee", 0))
            min_fee = entry.get("avgFee_10", 1)
            max_fee = entry.get("avgFee_90", avg_fee)
            rows.append({
                "timestamp": ts * 1000 if ts < 2e9 else ts,  # convert to ms
                "avg_fee": avg_fee,
                "min_fee": min_fee,
                "max_fee": max_fee,
                "fee_urgency": max_fee / min_fee if min_fee > 0 else 1.0,
            })

        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["timestamp", "avg_fee", "min_fee", "max_fee", "fee_urgency"])
            writer.writeheader()
            writer.writerows(rows)

        logger.info("Wrote %s (%d rows)", out_path, len(rows))

    except urllib.error.HTTPError as e:
        logger.warning("Fee rates API returned %d, trying alternative...", e.code)
        _download_block_timestamps(out_path)

    return out_path


def _download_block_timestamps(out_path: Path) -> None:
    """Alternative: download block-level data from mempool.space blocks API."""
    url = "https://mempool.space/api/v1/blocks"
    req = urllib.request.Request(url, headers={"User-Agent": "quant-system/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        blocks = json.loads(resp.read())

    rows = []
    for block in blocks:
        ts = block.get("timestamp", 0)
        extras = block.get("extras", {})
        avg_fee = extras.get("medianFee", extras.get("avgFeeRate", 0))
        rows.append({
            "timestamp": ts * 1000,
            "avg_fee": avg_fee,
            "min_fee": extras.get("feeRange", [1])[0] if extras.get("feeRange") else 1,
            "max_fee": extras.get("feeRange", [avg_fee])[-1] if extras.get("feeRange") else avg_fee,
            "fee_urgency": 1.0,
        })

    if rows:
        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["timestamp", "avg_fee", "min_fee", "max_fee", "fee_urgency"])
            writer.writeheader()
            writer.writerows(rows)
        logger.info("Wrote %s (%d rows from blocks API)", out_path, len(rows))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    download_fee_history("data_files")


if __name__ == "__main__":
    main()
