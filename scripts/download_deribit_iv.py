#!/usr/bin/env python3
"""Download historical implied volatility from Deribit public API.

Outputs: data_files/{symbol}_deribit_iv.csv
Columns: timestamp, implied_vol, put_call_ratio

Usage:
    python3 -m scripts.download_deribit_iv --currency BTC --days 365
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

DERIBIT_API = "https://www.deribit.com/api/v2/public"


def fetch_historical_volatility(currency: str = "BTC") -> list[dict]:
    """Fetch historical volatility data from Deribit."""
    url = f"{DERIBIT_API}/get_historical_volatility?currency={currency}"
    req = urllib.request.Request(url)
    req.add_header("User-Agent", "quant-system/1.0")
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
    if "result" not in data:
        raise ValueError(f"Unexpected Deribit response: {data}")
    return data["result"]


def fetch_book_summary(currency: str = "BTC", kind: str = "option") -> list[dict]:
    """Fetch current options book summary for put/call ratio."""
    url = f"{DERIBIT_API}/get_book_summary_by_currency?currency={currency}&kind={kind}"
    req = urllib.request.Request(url)
    req.add_header("User-Agent", "quant-system/1.0")
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
    return data.get("result", [])


def compute_put_call_ratio(summaries: list[dict]) -> float | None:
    """Compute put/call OI ratio from book summaries."""
    put_oi = 0.0
    call_oi = 0.0
    for s in summaries:
        name = s.get("instrument_name", "")
        oi = float(s.get("open_interest", 0))
        if "-P" in name:
            put_oi += oi
        elif "-C" in name:
            call_oi += oi
    if call_oi < 1e-8:
        return None
    return put_oi / call_oi


def download_iv(currency: str, out_dir: Path) -> Path:
    """Download IV data and save to CSV."""
    out_dir.mkdir(parents=True, exist_ok=True)
    symbol = f"{currency}USDT"
    out_path = out_dir / f"{symbol}_deribit_iv.csv"

    logger.info("Fetching historical volatility for %s...", currency)
    hv_data = fetch_historical_volatility(currency)

    logger.info("Fetching options book summary for put/call ratio...")
    try:
        summaries = fetch_book_summary(currency)
        current_pcr = compute_put_call_ratio(summaries)
    except Exception:
        logger.warning("Failed to fetch put/call ratio", exc_info=True)
        current_pcr = None

    rows = []
    for entry in hv_data:
        # Deribit returns [timestamp_ms, implied_vol]
        if isinstance(entry, list) and len(entry) >= 2:
            ts_ms = int(entry[0])
            iv = float(entry[1])
            ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
            rows.append({
                "timestamp": ts.isoformat(),
                "implied_vol": iv / 100.0,  # Deribit returns percentage
                "put_call_ratio": current_pcr if current_pcr is not None else "",
            })

    rows.sort(key=lambda r: r["timestamp"])

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "implied_vol", "put_call_ratio"])
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Wrote %d rows to %s", len(rows), out_path)
    return out_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Download Deribit IV data")
    parser.add_argument("--currency", default="BTC", help="BTC or ETH")
    parser.add_argument("--out-dir", default="data_files", help="Output directory")
    args = parser.parse_args()

    download_iv(args.currency, Path(args.out_dir))


if __name__ == "__main__":
    main()
