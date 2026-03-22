#!/usr/bin/env python3
"""Download historical implied volatility from Deribit public API.

Outputs: data_files/{currency.lower()}_iv_daily.csv
Columns: date, dvol, iv_atm_30d, iv_atm_7d, iv_skew_25d

Also retains legacy output: data_files/{SYMBOL}_deribit_iv.csv

Data sources (no auth required):
  - get_historical_volatility: hourly HV data (used to derive daily IV proxy)
  - get_volatility_index_data: DVOL index hourly OHLC (primary IV source)
  - get_book_summary_by_currency: current options book for live PCR/IV snapshot

Supports incremental updates: only fetches new data since last row in CSV.

Usage:
    python3 -m scripts.data.download_deribit_iv --currency BTC
    python3 -m scripts.data.download_deribit_iv --currency ETH
    python3 -m scripts.data.download_deribit_iv --all
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import time
import urllib.request
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

DERIBIT_API = "https://www.deribit.com/api/v2/public"
CHUNK_DAYS = 7
RATE_LIMIT_S = 0.5


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


def _fetch_dvol_chunk(
    currency: str, start_ms: int, end_ms: int
) -> List[Dict]:
    """Fetch one chunk of hourly DVOL data."""
    url = (
        f"{DERIBIT_API}/get_volatility_index_data?"
        f"currency={currency}&resolution=3600"
        f"&start_timestamp={start_ms}&end_timestamp={end_ms}"
    )
    req = urllib.request.Request(url)
    req.add_header("User-Agent", "quant-system/1.0")
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
    bars = data.get("result", {}).get("data", [])
    result = []
    for bar in bars:
        # bar = [timestamp_ms, open, high, low, close]
        result.append({
            "ts_ms": int(bar[0]),
            "open": float(bar[1]),
            "high": float(bar[2]),
            "low": float(bar[3]),
            "close": float(bar[4]),
        })
    return result


def fetch_dvol_hourly(
    currency: str, start_date: str = "2022-01-01", last_ts_ms: int = 0
) -> List[Dict]:
    """Fetch all hourly DVOL data from start_date (or after last_ts_ms for incremental)."""
    if last_ts_ms > 0:
        start = datetime.fromtimestamp(last_ts_ms / 1000 + 1, tz=timezone.utc)
    else:
        start = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = datetime.now(timezone.utc)

    if start >= end:
        logger.info("DVOL data already up to date for %s", currency)
        return []

    all_bars: List[Dict] = []
    cursor = start
    while cursor < end:
        chunk_end = min(cursor + timedelta(days=CHUNK_DAYS), end)
        start_ms = int(cursor.timestamp() * 1000)
        end_ms = int(chunk_end.timestamp() * 1000)
        try:
            bars = _fetch_dvol_chunk(currency, start_ms, end_ms)
            all_bars.extend(bars)
            logger.debug("%s DVOL %s -> %s: %d bars",
                         currency, cursor.strftime("%Y-%m-%d"),
                         chunk_end.strftime("%Y-%m-%d"), len(bars))
        except Exception as e:
            logger.warning("DVOL fetch failed %s -> %s: %s",
                           cursor.strftime("%Y-%m-%d"),
                           chunk_end.strftime("%Y-%m-%d"), e)
        cursor = chunk_end
        time.sleep(RATE_LIMIT_S)

    return all_bars


def _aggregate_dvol_to_daily(hourly_bars: List[Dict]) -> List[Dict]:
    """Aggregate hourly DVOL bars to daily rows.

    Produces per-day: dvol (close of last bar), high, low.
    """
    from collections import defaultdict

    by_date: Dict[str, List[Dict]] = defaultdict(list)
    for bar in hourly_bars:
        dt = datetime.fromtimestamp(bar["ts_ms"] / 1000, tz=timezone.utc)
        date_str = dt.strftime("%Y-%m-%d")
        by_date[date_str].append(bar)

    daily = []
    for date_str in sorted(by_date.keys()):
        bars = sorted(by_date[date_str], key=lambda b: b["ts_ms"])
        closes = [b["close"] for b in bars]
        highs = [b["high"] for b in bars]
        lows = [b["low"] for b in bars]
        daily.append({
            "date": date_str,
            "dvol": closes[-1],         # end-of-day DVOL
            "dvol_high": max(highs),
            "dvol_low": min(lows),
            "dvol_open": bars[0]["open"],
            "n_bars": len(bars),
        })
    return daily


def _load_existing_daily(path: Path) -> List[Dict]:
    """Load existing daily CSV if present."""
    if not path.exists():
        return []
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _last_date_in_rows(rows: List[Dict]) -> Optional[str]:
    if not rows:
        return None
    return rows[-1].get("date")


def download_daily_iv(currency: str, out_dir: Path) -> Path:
    """Download DVOL data, aggregate to daily, save as {currency}_iv_daily.csv.

    Supports incremental updates.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{currency.lower()}_iv_daily.csv"

    existing = _load_existing_daily(out_path)
    last_date = _last_date_in_rows(existing)

    # Determine incremental start
    last_ts_ms = 0
    if last_date:
        try:
            dt = datetime.strptime(last_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            last_ts_ms = int(dt.timestamp() * 1000) + 24 * 3600 * 1000  # day after last
            logger.info("Incremental update from %s for %s", last_date, currency)
        except ValueError:
            pass

    # Fetch hourly DVOL
    hourly = fetch_dvol_hourly(currency, last_ts_ms=last_ts_ms)
    if not hourly and existing:
        logger.info("No new DVOL data for %s — already up to date (%d rows)",
                     currency, len(existing))
        return out_path

    # If no existing data, fetch full history
    if not existing and not hourly:
        hourly = fetch_dvol_hourly(currency, start_date="2022-01-01")

    new_daily = _aggregate_dvol_to_daily(hourly)

    # Merge: keep existing rows, append new (skip overlap on last_date)
    existing_dates = {r["date"] for r in existing}
    merged = list(existing)
    for row in new_daily:
        if row["date"] not in existing_dates:
            merged.append(row)

    # Sort by date
    merged.sort(key=lambda r: r["date"])

    # Write
    fieldnames = ["date", "dvol", "dvol_high", "dvol_low", "dvol_open", "n_bars"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in merged:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    logger.info("Wrote %d daily rows to %s (new: %d)",
                len(merged), out_path, len(new_daily))
    return out_path


def download_legacy_iv(currency: str, out_dir: Path) -> Path:
    """Download IV data in legacy format (backward compat)."""
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
        if isinstance(entry, list) and len(entry) >= 2:
            ts_ms = int(entry[0])
            iv = float(entry[1])
            ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
            rows.append({
                "timestamp": ts.isoformat(),
                "implied_vol": iv / 100.0,
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
    parser.add_argument("--all", action="store_true", help="Download for both BTC and ETH")
    parser.add_argument("--legacy-only", action="store_true",
                        help="Only download legacy format (short-term IV)")
    args = parser.parse_args()

    currencies = ["BTC", "ETH"] if args.all else [args.currency]
    out = Path(args.out_dir)

    for cur in currencies:
        if args.legacy_only:
            download_legacy_iv(cur, out)
        else:
            # Daily DVOL-based IV (primary output for 4h alpha)
            path = download_daily_iv(cur, out)
            print(f"  {cur} daily IV -> {path}")
            # Also refresh legacy format
            legacy_path = download_legacy_iv(cur, out)
            print(f"  {cur} legacy IV -> {legacy_path}")


if __name__ == "__main__":
    main()
