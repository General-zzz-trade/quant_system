#!/usr/bin/env python3
"""Download macro data (DXY, SPX, VIX) from Yahoo Finance.

Uses Yahoo Finance v8 chart API — no dependencies, no auth.
Output: data_files/macro_daily.csv

Usage:
    python scripts/download_macro.py [--start 2020-01-01]
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

_YF_BASE = "https://query1.finance.yahoo.com/v8/finance/chart"
_TICKERS = {
    "dxy": "DX-Y.NYB",
    "spx": "%5EGSPC",
    "vix": "%5EVIX",
}


def _fetch_yahoo(ticker: str, period1: int, period2: int) -> list[dict]:
    """Fetch daily OHLCV from Yahoo Finance."""
    url = f"{_YF_BASE}/{ticker}?period1={period1}&period2={period2}&interval=1d"
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (quant-system/1.0)",
    })
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())

    chart = data.get("chart", {}).get("result", [{}])[0]
    timestamps = chart.get("timestamp", [])
    quote = chart.get("indicators", {}).get("quote", [{}])[0]
    closes = quote.get("close", [])

    rows = []
    for i, ts in enumerate(timestamps):
        c = closes[i] if i < len(closes) else None
        if c is None:
            continue
        rows.append({"timestamp": ts, "close": c})
    return rows


# Yahoo Finance assigns timestamps at market OPEN, but close prices are only
# available AFTER market close. Without offset correction, schedule forward-fill
# makes close prices available 6-17 hours too early → lookahead bias.
#
# Market close times (UTC):
#   NYSE (SPX, VIX): 21:00 UTC (4pm ET)
#   FOREX (DXY):     22:00 UTC (5pm ET)
# We add 1 hour margin after close to ensure data is actually available.
_CLOSE_OFFSET_SEC = {
    "dxy": 18 * 3600,   # 05:00 UTC → 23:00 UTC (FOREX close + 1h)
    "spx": 7 * 3600,    # 14:30 UTC → 21:30 UTC (NYSE close + 30min)
    "vix": 7 * 3600,    # 14:30 UTC → 21:30 UTC (NYSE close + 30min)
}


def download_macro(start: str, out_dir: str) -> Path:
    out_path = Path(out_dir) / "macro_daily.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    start_dt = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    period1 = int(start_dt.timestamp())
    period2 = int(datetime.now(timezone.utc).timestamp())

    # Fetch each ticker and apply close-time offset to prevent lookahead bias
    ticker_data: dict[str, dict[int, float]] = {}
    for key, ticker in _TICKERS.items():
        try:
            rows = _fetch_yahoo(ticker, period1, period2)
            offset = _CLOSE_OFFSET_SEC.get(key, 0)
            # Shift timestamp from market open → market close + margin
            ticker_data[key] = {r["timestamp"] + offset: r["close"] for r in rows}
            logger.info("Fetched %s: %d daily bars (offset +%dh)", key.upper(), len(rows), offset // 3600)
        except Exception:
            logger.exception("Failed to fetch %s", key)
            ticker_data[key] = {}

    # Merge all timestamps
    all_ts = set()
    for td in ticker_data.values():
        all_ts.update(td.keys())
    all_ts_sorted = sorted(all_ts)

    # Write combined CSV
    fields = ["timestamp", "timestamp_ms", "date", "dxy", "spx", "vix"]
    rows_out = []
    for ts in all_ts_sorted:
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        row = {
            "timestamp": ts,
            "timestamp_ms": ts * 1000,
            "date": dt.strftime("%Y-%m-%d"),
        }
        for key in ("dxy", "spx", "vix"):
            row[key] = ticker_data.get(key, {}).get(ts, "")
        rows_out.append(row)

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows_out)

    logger.info("Wrote %s (%d rows)", out_path, len(rows_out))
    return out_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Download macro data from Yahoo Finance")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--out-dir", default="data_files")
    args = parser.parse_args()

    download_macro(args.start, args.out_dir)


if __name__ == "__main__":
    main()
