#!/usr/bin/env python3
"""Download BTC/ETH exchange reserve and flow data from free public APIs.

Primary source: Coin Metrics Community API (free, no API key required)
    - FlowInExUSD: daily exchange inflow (USD)
    - FlowOutExUSD: daily exchange outflow (USD)
    - SplyExNtv: exchange-held supply (native units)

Derived columns:
    - exchange_netflow = inflow - outflow
    - exchange_reserve = SplyExNtv (alias for clarity)

Fallback source: blockchain.info (BTC only, limited metrics)

Output: data_files/{asset}_onchain_daily.csv with columns:
    date, exchange_reserve, exchange_inflow, exchange_outflow, exchange_netflow

Also updates data/onchain/{asset}_onchain_combined.csv (full Coin Metrics data)
for the batch feature engine V17 pipeline.

Usage:
    python3 -m data.downloads.download_onchain                          # BTC + ETH, incremental
    python3 -m data.downloads.download_onchain --asset btc              # BTC only
    python3 -m data.downloads.download_onchain --asset eth --start 2023-01-01
    python3 -m data.downloads.download_onchain --full-refresh           # Re-download all history
    python3 -m data.downloads.download_onchain --dry-run                # Show what would be fetched
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
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Coin Metrics Community API (free tier, no key required)
# Rate limit: ~10 requests per 6 seconds
_CM_API_BASE = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
_CM_METRICS = ["FlowInExUSD", "FlowOutExUSD", "SplyExNtv", "AdrActCnt", "TxTfrCnt", "HashRate"]
_CM_PAGE_SIZE = 10000
_CM_RATE_LIMIT_SLEEP = 0.7

# blockchain.info API (BTC only, free, no key)
_BLOCKCHAIN_INFO_BASE = "https://api.blockchain.info/charts"

_DEFAULT_ASSETS = ["btc", "eth"]
_DEFAULT_START = "2020-01-01"

_DATA_FILES_DIR = Path("data_files")
_ONCHAIN_DIR = Path("data/onchain")


def _fetch_coinmetrics_page(
    asset: str,
    metrics: str,
    start: str,
    next_page_token: Optional[str] = None,
    end: Optional[str] = None,
) -> dict:
    """Fetch one page from Coin Metrics Community API."""
    url = (
        f"{_CM_API_BASE}?assets={asset}&metrics={metrics}"
        f"&page_size={_CM_PAGE_SIZE}&start_time={start}&sort=time"
    )
    if end:
        url += f"&end_time={end}"
    if next_page_token:
        url += f"&next_page_token={next_page_token}"

    req = urllib.request.Request(url)
    req.add_header("User-Agent", "quant-system/1.0")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        logger.error("Coin Metrics API error %d: %s", e.code, e.reason)
        raise
    except urllib.error.URLError as e:
        logger.error("Coin Metrics connection error: %s", e.reason)
        raise


def _fetch_all_coinmetrics(
    asset: str, start: str, end: Optional[str] = None
) -> List[dict]:
    """Fetch all pages of Coin Metrics data for the given asset."""
    metrics_str = ",".join(_CM_METRICS)
    all_rows: List[dict] = []
    next_token: Optional[str] = None
    page = 0

    while True:
        page += 1
        logger.info("Fetching Coin Metrics page %d for %s (rows so far: %d)...",
                     page, asset, len(all_rows))
        data = _fetch_coinmetrics_page(asset, metrics_str, start, next_token, end)
        rows = data.get("data", [])
        all_rows.extend(rows)

        next_token = data.get("next_page_token")
        if not next_token or not rows:
            break
        time.sleep(_CM_RATE_LIMIT_SLEEP)

    logger.info("Total rows fetched for %s: %d", asset, len(all_rows))
    return all_rows


def _parse_timestamp(iso_str: str) -> int:
    """Parse ISO datetime string to epoch milliseconds."""
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        return int(dt.timestamp() * 1000)
    except (ValueError, AttributeError):
        return 0


def _parse_date(iso_str: str) -> str:
    """Extract YYYY-MM-DD from ISO datetime string."""
    try:
        return iso_str[:10]
    except (TypeError, IndexError):
        return ""


def _get_last_date(csv_path: Path) -> Optional[str]:
    """Read the last date in a CSV file for incremental updates."""
    if not csv_path.exists():
        return None
    try:
        with csv_path.open("r") as f:
            reader = csv.DictReader(f)
            last_date = None
            for row in reader:
                last_date = row.get("date", "")
            return last_date if last_date else None
    except Exception as e:
        logger.warning("Could not read last date from %s: %s", csv_path, e)
        return None


def _get_last_date_combined(csv_path: Path) -> Optional[str]:
    """Read the last date from the combined CSV (ISO format in 'date' column)."""
    if not csv_path.exists():
        return None
    try:
        with csv_path.open("r") as f:
            reader = csv.DictReader(f)
            last_date = None
            for row in reader:
                d = row.get("date", "")
                if d:
                    last_date = d[:10]  # Extract YYYY-MM-DD
            return last_date if last_date else None
    except Exception as e:
        logger.warning("Could not read last date from %s: %s", csv_path, e)
        return None


def _write_daily_csv(rows: List[dict], out_path: Path, append: bool = False) -> int:
    """Write exchange flow data to the daily CSV format.

    Columns: date, exchange_reserve, exchange_inflow, exchange_outflow, exchange_netflow
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["date", "exchange_reserve", "exchange_inflow", "exchange_outflow", "exchange_netflow"]

    mode = "a" if append else "w"
    written = 0

    # If appending, read existing dates to avoid duplicates
    existing_dates: set = set()
    if append and out_path.exists():
        with out_path.open("r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_dates.add(row.get("date", ""))

    with out_path.open(mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not append or not out_path.exists() or out_path.stat().st_size == 0:
            writer.writeheader()

        for row in rows:
            date_str = _parse_date(row.get("time", ""))
            if not date_str or date_str in existing_dates:
                continue

            flow_in_str = row.get("FlowInExUSD", "")
            flow_out_str = row.get("FlowOutExUSD", "")
            supply_str = row.get("SplyExNtv", "")

            flow_in = float(flow_in_str) if flow_in_str else None
            flow_out = float(flow_out_str) if flow_out_str else None
            supply = float(supply_str) if supply_str else None

            netflow = None
            if flow_in is not None and flow_out is not None:
                netflow = flow_in - flow_out

            writer.writerow({
                "date": date_str,
                "exchange_reserve": f"{supply:.8f}" if supply is not None else "",
                "exchange_inflow": f"{flow_in:.2f}" if flow_in is not None else "",
                "exchange_outflow": f"{flow_out:.2f}" if flow_out is not None else "",
                "exchange_netflow": f"{netflow:.2f}" if netflow is not None else "",
            })
            existing_dates.add(date_str)
            written += 1

    return written


def _write_combined_csv(rows: List[dict], out_path: Path) -> int:
    """Write full Coin Metrics data to combined CSV (for batch feature engine)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["timestamp", "date"] + _CM_METRICS

    written = 0
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            ts = row.get("time", "")
            ts_ms = _parse_timestamp(ts)
            out_row: Dict[str, str] = {"timestamp": str(ts_ms), "date": ts}
            for m in _CM_METRICS:
                out_row[m] = row.get(m, "")
            writer.writerow(out_row)
            written += 1

    return written


def _fetch_blockchain_info_btc() -> Optional[List[dict]]:
    """Fallback: fetch BTC exchange data from blockchain.info.

    Note: blockchain.info doesn't have exchange reserve data directly,
    but provides related metrics. This is a limited fallback.
    """
    try:
        url = f"{_BLOCKCHAIN_INFO_BASE}/estimated-transaction-volume-usd?timespan=30days&format=json"
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "quant-system/1.0")
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
        return data.get("values", [])
    except Exception as e:
        logger.warning("blockchain.info fallback failed: %s", e)
        return None


def download_asset(
    asset: str,
    start: str = _DEFAULT_START,
    full_refresh: bool = False,
    dry_run: bool = False,
) -> None:
    """Download on-chain exchange flow data for a single asset."""
    asset_lower = asset.lower()
    asset_upper = asset.upper()

    # Output paths
    daily_path = _DATA_FILES_DIR / f"{asset_lower}_onchain_daily.csv"
    combined_path = _ONCHAIN_DIR / f"{asset_lower}_onchain_combined.csv"

    # Determine start date for incremental update
    effective_start = start
    if not full_refresh:
        last_date = _get_last_date_combined(combined_path)
        if last_date:
            logger.info("%s: last data date = %s, fetching incrementally", asset_upper, last_date)
            effective_start = last_date
        else:
            logger.info("%s: no existing data, fetching from %s", asset_upper, start)

    if dry_run:
        logger.info("[DRY RUN] Would fetch %s data from %s", asset_upper, effective_start)
        logger.info("[DRY RUN] Would write to: %s", daily_path)
        logger.info("[DRY RUN] Would write to: %s", combined_path)
        return

    # Fetch from Coin Metrics
    try:
        rows = _fetch_all_coinmetrics(asset_lower, effective_start)
    except Exception as e:
        logger.error("Coin Metrics fetch failed for %s: %s", asset_upper, e)
        if asset_lower == "btc":
            logger.info("Attempting blockchain.info fallback for BTC...")
            fallback = _fetch_blockchain_info_btc()
            if fallback:
                logger.info("blockchain.info returned %d values (limited metrics)", len(fallback))
            else:
                logger.error("All API sources failed for BTC")
        return

    if not rows:
        logger.warning("No data returned for %s", asset_upper)
        return

    # If incremental, merge with existing data
    if not full_refresh and combined_path.exists():
        existing_rows = _load_existing_combined(combined_path)
        existing_dates = {_parse_date(r.get("time", "")) for r in existing_rows}
        new_rows = [r for r in rows if _parse_date(r.get("time", "")) not in existing_dates]
        all_rows = existing_rows + new_rows
        logger.info("%s: %d existing + %d new = %d total rows",
                     asset_upper, len(existing_rows), len(new_rows), len(all_rows))
    else:
        all_rows = rows

    # Sort by time
    all_rows.sort(key=lambda r: r.get("time", ""))

    # Write combined CSV (full rewrite for consistency)
    combined_written = _write_combined_csv(all_rows, combined_path)
    logger.info("Wrote %d rows to %s", combined_written, combined_path)

    # Write daily exchange flow CSV
    daily_written = _write_daily_csv(all_rows, daily_path, append=False)
    logger.info("Wrote %d rows to %s", daily_written, daily_path)

    # Summary
    if all_rows:
        first_date = _parse_date(all_rows[0].get("time", ""))
        last_date = _parse_date(all_rows[-1].get("time", ""))
        logger.info("%s summary: %s to %s (%d days)", asset_upper, first_date, last_date, len(all_rows))


def _load_existing_combined(csv_path: Path) -> List[dict]:
    """Load existing combined CSV back into row dicts (same format as API response)."""
    rows: List[dict] = []
    try:
        with csv_path.open("r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                api_row: Dict[str, str] = {"time": row.get("date", "")}
                for m in _CM_METRICS:
                    val = row.get(m, "")
                    if val:
                        api_row[m] = val
                rows.append(api_row)
    except Exception as e:
        logger.warning("Could not load existing data from %s: %s", csv_path, e)
    return rows


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Download on-chain exchange flow data (Coin Metrics free API)"
    )
    parser.add_argument(
        "--asset", type=str, default=None,
        help="Asset to download (btc or eth). Default: both."
    )
    parser.add_argument(
        "--start", type=str, default=_DEFAULT_START,
        help=f"Start date for initial download (default: {_DEFAULT_START})"
    )
    parser.add_argument(
        "--full-refresh", action="store_true",
        help="Re-download all history (ignore existing data)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be fetched without downloading"
    )
    args = parser.parse_args()

    assets = [args.asset] if args.asset else _DEFAULT_ASSETS

    for asset in assets:
        logger.info("=" * 60)
        logger.info("Processing %s...", asset.upper())
        logger.info("=" * 60)
        download_asset(
            asset=asset,
            start=args.start,
            full_refresh=args.full_refresh,
            dry_run=args.dry_run,
        )

    logger.info("Done.")


if __name__ == "__main__":
    main()
