#!/usr/bin/env python3
"""Automated data refresh — incremental update of all data sources.

Designed to run daily via cron. Fetches only new data since last update.

Usage:
    python3 -m scripts.refresh_data                  # All symbols, all sources
    python3 -m scripts.refresh_data --symbols BTCUSDT # Single symbol
    python3 -m scripts.refresh_data --skip-slow       # Skip slow sources (onchain, mempool)
    python3 -m scripts.refresh_data --dry-run          # Show what would be updated
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

DATA_DIR = Path("data_files")
DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
PYTHON = sys.executable


def _last_ts_ms(csv_path: Path, ts_col: str = "open_time") -> Optional[int]:
    """Read the last timestamp from a CSV file."""
    if not csv_path.exists():
        return None
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        if ts_col not in (reader.fieldnames or []):
            # Try 'timestamp' as fallback
            ts_col = "timestamp"
            if ts_col not in (reader.fieldnames or []):
                return None
        last = None
        for row in reader:
            last = row[ts_col]
    if last is None:
        return None
    return int(float(last))


def _ts_str(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def _run(cmd: List[str], label: str, dry_run: bool = False) -> bool:
    """Run a subprocess, return True on success."""
    print(f"  [{label}] ", end="", flush=True)
    if dry_run:
        print(f"DRY RUN: {' '.join(cmd)}")
        return True
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600, cwd="/opt/quant_system",
        )
        if result.returncode == 0:
            # Count lines of output for summary
            lines = result.stdout.strip().split("\n") if result.stdout.strip() else []
            print(f"OK ({len(lines)} lines)")
            return True
        else:
            err = result.stderr.strip().split("\n")[-1] if result.stderr.strip() else "unknown"
            print(f"FAIL: {err}")
            return False
    except subprocess.TimeoutExpired:
        print("TIMEOUT (600s)")
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False


def _append_klines(symbol: str, interval: str, dry_run: bool) -> bool:
    """Incrementally append new klines to existing CSV."""
    csv_path = DATA_DIR / f"{symbol}_{interval}.csv"
    last_ts = _last_ts_ms(csv_path, "open_time")

    if last_ts is None:
        start = "2019-09-08"
        print(f"  [{symbol} {interval}] No existing data, full download from {start}")
    else:
        age_hours = (time.time() * 1000 - last_ts) / 3_600_000
        if age_hours < 2:
            print(f"  [{symbol} {interval}] Up to date ({_ts_str(last_ts)})")
            return True
        # Start from last timestamp (download_all deduplicates)
        start_dt = datetime.fromtimestamp(last_ts / 1000, tz=timezone.utc)
        start = start_dt.strftime("%Y-%m-%d")
        print(f"  [{symbol} {interval}] Last: {_ts_str(last_ts)}, fetching from {start}")

    if dry_run:
        print(f"  [{symbol} {interval}] DRY RUN")
        return True

    # Fetch new data using the API directly (more efficient than calling subprocess)
    from scripts.download_binance_klines import fetch_klines, interval_ms, LIMIT

    if last_ts is not None:
        start_ms = last_ts  # Start from last known bar
    else:
        start_ms = 1567900800000  # 2019-09-08

    now_ms = int(time.time() * 1000)
    new_rows = []
    current = start_ms

    while current < now_ms:
        try:
            data = fetch_klines(symbol, interval, current, LIMIT)
        except Exception as e:
            print(f"    Error at {current}: {e}, retrying...")
            time.sleep(3)
            continue

        if not data:
            break
        new_rows.extend(data)
        last_open = data[-1][0]
        if len(data) < LIMIT:
            break
        current = last_open + interval_ms(interval)
        time.sleep(0.2)

    if not new_rows:
        print(f"  [{symbol} {interval}] No new data")
        return True

    # Read existing data
    existing = {}
    if csv_path.exists():
        with open(csv_path) as f:
            reader = csv.reader(f)
            header = next(reader)
            for row in reader:
                existing[int(row[0])] = row

    # Merge new rows
    added = 0
    for row in new_rows:
        ts = int(row[0])
        if ts not in existing:
            existing[ts] = row
            added += 1

    # Write back sorted
    sorted_rows = sorted(existing.values(), key=lambda r: int(r[0]))
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["open_time", "open", "high", "low", "close", "volume",
                     "close_time", "quote_volume", "trades", "taker_buy_volume",
                     "taker_buy_quote_volume", "ignore"])
        for row in sorted_rows:
            w.writerow(row)

    print(f"  [{symbol} {interval}] +{added} bars, total {len(sorted_rows)}")
    return True


def _append_spot_klines(symbol: str, dry_run: bool) -> bool:
    """Incrementally append spot klines."""
    csv_path = DATA_DIR / f"{symbol}_spot_1h.csv"
    last_ts = _last_ts_ms(csv_path, "open_time")

    if last_ts is None:
        start = "2019-01-01"
    else:
        age_hours = (time.time() * 1000 - last_ts) / 3_600_000
        if age_hours < 2:
            print(f"  [{symbol} spot] Up to date ({_ts_str(last_ts)})")
            return True
        start_dt = datetime.fromtimestamp(last_ts / 1000, tz=timezone.utc)
        start = start_dt.strftime("%Y-%m-%d")

    if dry_run:
        print(f"  [{symbol} spot] DRY RUN from {start}")
        return True

    from scripts.download_spot_klines import fetch_klines as fetch_spot, LIMIT as SPOT_LIMIT

    start_ms = last_ts if last_ts else 1546300800000  # 2019-01-01
    now_ms = int(time.time() * 1000)
    new_rows = []
    current = start_ms

    while current < now_ms:
        try:
            data = fetch_spot(symbol.replace("USDT", "") + "USDT", "1h", current, SPOT_LIMIT)
        except Exception as e:
            print(f"    Spot error: {e}, retrying...")
            time.sleep(3)
            continue
        if not data:
            break
        new_rows.extend(data)
        if len(data) < SPOT_LIMIT:
            break
        current = data[-1][0] + 3_600_000
        time.sleep(0.2)

    if not new_rows:
        print(f"  [{symbol} spot] No new data")
        return True

    existing = {}
    if csv_path.exists():
        with open(csv_path) as f:
            reader = csv.reader(f)
            header = next(reader)
            for row in reader:
                existing[int(row[0])] = row

    added = 0
    for row in new_rows:
        ts = int(row[0])
        if ts not in existing:
            existing[ts] = row
            added += 1

    sorted_rows = sorted(existing.values(), key=lambda r: int(r[0]))
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["open_time", "open", "high", "low", "close", "volume",
                     "close_time", "quote_volume", "trades", "taker_buy_volume",
                     "taker_buy_quote_volume", "ignore"])
        for row in sorted_rows:
            w.writerow(row)

    print(f"  [{symbol} spot] +{added} bars, total {len(sorted_rows)}")
    return True


def refresh_all(symbols: List[str], skip_slow: bool, dry_run: bool) -> dict:
    """Refresh all data sources for given symbols."""
    t0 = time.time()
    results = {"ok": 0, "fail": 0, "skip": 0}

    def track(ok: bool):
        results["ok" if ok else "fail"] += 1

    print(f"\n{'='*60}")
    print(f"  Data Refresh: {', '.join(symbols)}")
    print(f"  Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'='*60}")

    # ── 1. Futures klines (1h) ──
    print("\n1. Futures Klines (1h)")
    for sym in symbols:
        track(_append_klines(sym, "1h", dry_run))

    # ── 2. Spot klines ──
    print("\n2. Spot Klines (1h)")
    for sym in symbols:
        track(_append_spot_klines(sym, dry_run))

    # ── 3. Funding rates ──
    print("\n3. Funding Rates")
    track(_run(
        [PYTHON, "-m", "scripts.download_funding_rates", "--symbols"] + symbols,
        "funding", dry_run,
    ))

    # ── 4. Open Interest (30-day window, incremental) ──
    print("\n4. Open Interest")
    track(_run(
        [PYTHON, "-m", "scripts.download_open_interest", "--symbols"] + symbols,
        "OI", dry_run,
    ))

    # ── 5. Long/Short ratio (30-day window, incremental) ──
    print("\n5. Long/Short Ratio")
    track(_run(
        [PYTHON, "-m", "scripts.download_ls_ratio", "--symbols"] + symbols,
        "LS ratio", dry_run,
    ))

    # ── 6. Taker ratio ──
    print("\n6. Taker Buy/Sell Ratio")
    track(_run(
        [PYTHON, "-m", "scripts.download_taker_ratio", "--symbols"] + symbols,
        "taker ratio", dry_run,
    ))

    # ── 7. Fear & Greed Index ──
    print("\n7. Fear & Greed Index")
    track(_run(
        [PYTHON, "-m", "scripts.download_fear_greed"],
        "FGI", dry_run,
    ))

    # ── 8. Macro (DXY, SPX, VIX) ──
    print("\n8. Macro Data")
    track(_run(
        [PYTHON, "-m", "scripts.download_macro"],
        "macro", dry_run,
    ))

    # ── 9. Deribit IV + PCR ──
    print("\n9. Deribit IV / Put-Call Ratio")
    for sym in symbols:
        currency = sym.replace("USDT", "")
        if currency in ("BTC", "ETH", "SOL"):
            track(_run(
                [PYTHON, "-m", "scripts.download_deribit_iv", "--currency", currency],
                f"deribit {currency}", dry_run,
            ))

    # ── 10. Liquidation proxy (derived from klines + OI) ──
    print("\n10. Liquidation Proxy")
    for sym in symbols:
        track(_run(
            [PYTHON, "-m", "scripts.download_liquidations", "--symbol", sym],
            f"liq {sym}", dry_run,
        ))

    # ── 11. Slow sources (skip with --skip-slow) ──
    if skip_slow:
        print("\n11. Slow Sources — SKIPPED (--skip-slow)")
        results["skip"] += 2
    else:
        print("\n11. On-Chain Metrics")
        for asset in ["btc", "eth", "sol"]:
            track(_run(
                [PYTHON, "-m", "scripts.download_onchain_metrics", "--asset", asset],
                f"onchain {asset}", dry_run,
            ))

        print("\n12. Mempool Fees")
        track(_run(
            [PYTHON, "-m", "scripts.download_mempool"],
            "mempool", dry_run,
        ))

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  Refresh complete in {elapsed:.0f}s")
    print(f"  OK: {results['ok']}  FAIL: {results['fail']}  SKIP: {results['skip']}")

    # Show staleness summary
    print(f"\n  Data Staleness:")
    for sym in symbols:
        csv_path = DATA_DIR / f"{sym}_1h.csv"
        last = _last_ts_ms(csv_path, "open_time")
        if last:
            age_h = (time.time() * 1000 - last) / 3_600_000
            status = "OK" if age_h < 2 else f"STALE ({age_h:.0f}h)"
            print(f"    {sym}: {_ts_str(last)} — {status}")
        else:
            print(f"    {sym}: MISSING")

    print(f"{'='*60}\n")
    return results


def main():
    parser = argparse.ArgumentParser(description="Refresh all data sources")
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    parser.add_argument("--skip-slow", action="store_true", help="Skip onchain/mempool")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be updated")
    args = parser.parse_args()

    results = refresh_all(
        symbols=[s.upper() for s in args.symbols],
        skip_slow=args.skip_slow,
        dry_run=args.dry_run,
    )

    sys.exit(1 if results["fail"] > 0 else 0)


if __name__ == "__main__":
    main()
