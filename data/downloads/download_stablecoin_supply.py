"""Download stablecoin total supply history from DeFiLlama API.

Fetches USDT (id=1), USDC (id=2), and aggregate total supply.
Saves daily data to data_files/stablecoin_daily.csv.

Usage:
    python3 -m data.downloads.download_stablecoin_supply
"""
from __future__ import annotations

import csv
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

OUTPUT_PATH = Path("data_files/stablecoin_daily.csv")

# DeFiLlama stablecoin IDs
STABLECOIN_IDS = {
    "usdt": 1,
    "usdc": 2,
}

BASE_URL = "https://stablecoins.llama.fi/stablecoincharts/all"


def _fetch_json(url: str) -> list:
    """Fetch JSON from URL with error handling."""
    import urllib.request
    import json

    req = urllib.request.Request(url, headers={"User-Agent": "quant_system/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        logger.error("Failed to fetch %s: %s", url, e)
        return []


def download_stablecoin_supply() -> Path:
    """Download USDT, USDC, and total stablecoin supply history."""
    logger.info("Fetching aggregate stablecoin supply from DeFiLlama...")
    total_data = _fetch_json(BASE_URL)
    if not total_data:
        logger.error("No aggregate data returned")
        sys.exit(1)

    # Build date -> total_supply mapping
    total_by_date: dict[int, float] = {}
    for entry in total_data:
        ts = int(entry["date"])
        circ = entry.get("totalCirculating", {})
        # Sum only peggedUSD (dominant, others are tiny)
        total_by_date[ts] = circ.get("peggedUSD", 0)

    # Fetch individual stablecoins
    supply_by_coin: dict[str, dict[int, float]] = {}
    for name, sid in STABLECOIN_IDS.items():
        url = f"{BASE_URL}?stablecoin={sid}"
        logger.info("Fetching %s (id=%d)...", name.upper(), sid)
        data = _fetch_json(url)
        coin_map: dict[int, float] = {}
        for entry in data:
            ts = int(entry["date"])
            circ = entry.get("totalCirculating", {})
            coin_map[ts] = circ.get("peggedUSD", 0)
        supply_by_coin[name] = coin_map
        logger.info("  %s: %d data points", name.upper(), len(coin_map))

    # Merge all dates
    all_dates = sorted(set(total_by_date.keys()))
    logger.info("Total dates: %d (from %s to %s)",
                len(all_dates),
                datetime.fromtimestamp(all_dates[0], tz=timezone.utc).date(),
                datetime.fromtimestamp(all_dates[-1], tz=timezone.utc).date())

    # Write CSV
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "timestamp", "usdt_supply", "usdc_supply", "total_supply"])
        for ts in all_dates:
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            date_str = dt.strftime("%Y-%m-%d")
            usdt = supply_by_coin.get("usdt", {}).get(ts, "")
            usdc = supply_by_coin.get("usdc", {}).get(ts, "")
            total = total_by_date.get(ts, "")
            writer.writerow([date_str, ts * 1000, usdt, usdc, total])

    logger.info("Saved %d rows to %s", len(all_dates), OUTPUT_PATH)
    return OUTPUT_PATH


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    path = download_stablecoin_supply()
    print(f"Done: {path}")


if __name__ == "__main__":
    main()
