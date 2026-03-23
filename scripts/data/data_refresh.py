#!/usr/bin/env python3
"""Unified data refresh -- daily incremental download of klines, funding rates, and external data.

Downloads only new data (incremental), validates integrity, and alerts on failure.

Usage:
    python3 -m scripts.data_refresh                    # All symbols, all sources
    python3 -m scripts.data_refresh --symbol BTCUSDT   # Single symbol
    python3 -m scripts.data_refresh --dry-run           # Check freshness only
    python3 -m scripts.data_refresh --alert             # Send alerts on failure
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, "/quant_system")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


SYMBOLS = ["BTCUSDT", "ETHUSDT"]
DATA_DIR = Path("data_files")
KLINE_INTERVALS = {"1h": "{symbol}_1h.csv"}  # primary timeframe
EXTERNAL_SOURCES = [
    "funding_rates",
    "open_interest",
    "fear_greed",
]

# Freshness thresholds
MAX_GAP_HOURS = 4        # max allowed gap in hourly data
MAX_STALE_HOURS = 26     # data older than this triggers alert


def check_data_freshness(symbol: str) -> Dict[str, Any]:
    """Check how fresh the data is for a symbol."""
    results = {}

    for interval, filename_template in KLINE_INTERVALS.items():
        filename = filename_template.format(symbol=symbol)
        filepath = DATA_DIR / filename

        if not filepath.exists():
            results[f"kline_{interval}"] = {
                "exists": False,
                "stale_hours": float("inf"),
                "gaps": [],
            }
            continue

        try:
            df = pd.read_csv(filepath)
            if df.empty:
                results[f"kline_{interval}"] = {
                    "exists": True,
                    "rows": 0,
                    "stale_hours": float("inf"),
                }
                continue

            # Detect timestamp column
            ts_col = None
            for col in ["open_time", "timestamp", "ts"]:
                if col in df.columns:
                    ts_col = col
                    break

            if ts_col is None:
                results[f"kline_{interval}"] = {
                    "exists": True,
                    "rows": len(df),
                    "error": "no timestamp column found",
                }
                continue

            # Check freshness
            last_ts = df[ts_col].iloc[-1]
            if last_ts > 1e12:  # milliseconds
                last_ts = last_ts / 1000
            last_dt = datetime.utcfromtimestamp(last_ts)
            stale_hours = (datetime.utcnow() - last_dt).total_seconds() / 3600

            # Gap detection
            timestamps = df[ts_col].values.astype(float)
            if timestamps[0] > 1e12:
                timestamps = timestamps / 1000
            diffs = np.diff(timestamps)
            expected_diff = 3600 if interval == "1h" else 900  # seconds
            gaps = []
            for i, d in enumerate(diffs):
                if d > expected_diff * MAX_GAP_HOURS:
                    gap_start = datetime.utcfromtimestamp(timestamps[i])
                    gap_end = datetime.utcfromtimestamp(timestamps[i + 1])
                    gaps.append({
                        "start": gap_start.isoformat(),
                        "end": gap_end.isoformat(),
                        "hours": d / 3600,
                    })

            # NaN check
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            nan_counts = df[numeric_cols].isna().sum()
            nan_total = nan_counts.sum()

            results[f"kline_{interval}"] = {
                "exists": True,
                "rows": len(df),
                "first_ts": datetime.utcfromtimestamp(
                    timestamps[0]).isoformat(),
                "last_ts": last_dt.isoformat(),
                "stale_hours": round(stale_hours, 1),
                "gaps": gaps[-5:] if gaps else [],  # last 5 gaps only
                "gap_count": len(gaps),
                "nan_count": int(nan_total),
            }
        except Exception as e:
            results[f"kline_{interval}"] = {
                "exists": True,
                "error": str(e),
            }

    return results


def _get_last_kline_ts_ms(filepath: Path) -> Optional[int]:
    """Read last open_time from existing kline CSV to enable incremental download."""
    if not filepath.exists():
        return None
    try:
        df = pd.read_csv(filepath)
        if df.empty:
            return None
        ts_col = None
        for col in ["open_time", "timestamp", "ts"]:
            if col in df.columns:
                ts_col = col
                break
        if ts_col is None:
            return None
        last_val = int(df[ts_col].iloc[-1])
        # Ensure milliseconds
        if last_val < 1e12:
            last_val = last_val * 1000
        return last_val
    except Exception:
        return None


def refresh_klines(symbol: str, interval: str = "1h", dry_run: bool = False) -> Dict[str, Any]:
    """Incrementally download new klines for a symbol."""
    result = {"symbol": symbol, "interval": interval, "success": False}

    filename = KLINE_INTERVALS.get(interval, f"{symbol}_{interval}.csv").format(symbol=symbol)
    filepath = DATA_DIR / filename

    if dry_run:
        freshness = check_data_freshness(symbol)
        result["freshness"] = freshness
        result["success"] = True
        result["dry_run"] = True
        return result

    try:
        from scripts.data.download_binance_klines import download_all, interval_ms

        # Incremental: start from last known timestamp + 1 interval
        last_ts_ms = _get_last_kline_ts_ms(filepath)
        start_ms = None
        if last_ts_ms is not None:
            start_ms = last_ts_ms + interval_ms(interval)
            logger.info(
                "%s: incremental from %s",
                symbol,
                datetime.utcfromtimestamp(start_ms / 1000).isoformat(),
            )

        # Write to temp file first, then atomically replace on success.
        # Prevents data loss if download fails mid-way (the old file survives).
        import tempfile
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".csv", dir=str(DATA_DIR))
        os.close(tmp_fd)

        try:
            n_rows = download_all(
                symbol=symbol,
                interval=interval,
                output_path=tmp_path,
                start_ms=start_ms,
            )

            # Validate: new file must have data
            if n_rows > 0:
                # If incremental, merge with existing data
                if start_ms is not None and filepath.exists():
                    existing = pd.read_csv(filepath)
                    new_data = pd.read_csv(tmp_path)
                    merged = pd.concat([existing, new_data]).drop_duplicates(
                        subset=["open_time"], keep="last"
                    ).sort_values("open_time")
                    merged.to_csv(str(filepath), index=False)
                    os.unlink(tmp_path)
                    result["rows"] = len(merged)
                else:
                    os.replace(tmp_path, str(filepath))
                    result["rows"] = n_rows
                result["success"] = True
            else:
                # Empty download — keep existing file, discard temp
                os.unlink(tmp_path)
                result["rows"] = 0
                result["success"] = True
                result["note"] = "no new bars (up to date)"
        except Exception as e:
            # Download failed — keep existing file intact
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise
    except Exception as e:
        result["error"] = str(e)

    return result


def refresh_funding_rates(symbol: str, dry_run: bool = False) -> Dict[str, Any]:
    """Download funding rate data."""
    result = {"symbol": symbol, "source": "funding_rates", "success": False}

    if dry_run:
        result["success"] = True
        result["dry_run"] = True
        return result

    try:
        import subprocess
        # download_funding_rates.py uses --symbols (nargs="+")
        cmd = [
            sys.executable, "-m", "scripts.data.download_funding_rates",
            "--symbols", symbol,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120, cwd="/quant_system")
        result["success"] = proc.returncode == 0
        if proc.returncode != 0:
            result["error"] = proc.stderr[:500]
    except Exception as e:
        result["error"] = str(e)

    return result


def refresh_external_data(dry_run: bool = False) -> List[Dict[str, Any]]:
    """Download external data sources (FGI, OI, etc.)."""
    results = []

    # Each entry: (source_name, module_path, extra_args)
    scripts: List[Tuple[str, str, List[str]]] = [
        ("fear_greed", "scripts.data.download_fear_greed", []),
        ("open_interest", "scripts.data.download_open_interest", []),
        ("dvol_btc", "scripts.data.download_deribit_dvol", ["--currency", "BTC"]),
        ("dvol_eth", "scripts.data.download_deribit_dvol", ["--currency", "ETH"]),
    ]

    for source, module, extra_args in scripts:
        result: Dict[str, Any] = {"source": source, "success": False}

        if dry_run:
            result["success"] = True
            result["dry_run"] = True
            results.append(result)
            continue

        try:
            import subprocess
            cmd = [sys.executable, "-m", module] + extra_args
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120, cwd="/quant_system")
            result["success"] = proc.returncode == 0
            if proc.returncode != 0:
                result["error"] = proc.stderr[:500]
        except Exception as e:
            result["error"] = str(e)

        results.append(result)

    return results


def send_alert(message: str, *, severity: str = "info") -> None:
    """Send alert via Telegram/webhook (same as auto_retrain.py)."""
    import urllib.request

    tg_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    tg_chat = os.environ.get("TELEGRAM_CHAT_ID", "")
    if tg_token and tg_chat:
        try:
            url = f"https://api.telegram.org/bot{tg_token}/sendMessage"
            payload = json.dumps({"chat_id": tg_chat, "text": f"[{severity.upper()}] {message}"}).encode()
            req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=10)
        except Exception as e:
            logger.warning("Telegram alert failed: %s", e)

    webhook_url = os.environ.get("RETRAIN_WEBHOOK_URL", "")
    if webhook_url:
        try:
            payload = json.dumps({"text": f"[{severity.upper()}] {message}"}).encode()
            req = urllib.request.Request(webhook_url, data=payload, headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=10)
        except Exception as e:
            logger.warning("Webhook alert failed: %s", e)


def main():
    parser = argparse.ArgumentParser(description="Unified Data Refresh")
    parser.add_argument("--symbol", default=None,
                        help="Comma-separated symbols (default: all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Check freshness only, don't download")
    parser.add_argument("--alert", action="store_true",
                        help="Send alerts on failure")
    parser.add_argument("--skip-external", action="store_true",
                        help="Skip external data sources (FGI, OI)")
    args = parser.parse_args()

    symbols = (
        [s.strip().upper() for s in args.symbol.split(",")]
        if args.symbol
        else SYMBOLS
    )

    print("=" * 70)
    print("  DATA REFRESH")
    print(f"  Date:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Symbols:  {symbols}")
    print(f"  Dry run:  {args.dry_run}")
    print("=" * 70)

    all_results: List[Dict[str, Any]] = []
    failures: List[str] = []

    # 1. Kline data
    for symbol in symbols:
        print(f"\n-- {symbol} klines --")

        # Check freshness first
        freshness = check_data_freshness(symbol)
        for key, info in freshness.items():
            stale = info.get("stale_hours", 0)
            gaps = info.get("gap_count", 0)
            nans = info.get("nan_count", 0)
            if isinstance(stale, float) and stale != float("inf"):
                print(f"  {key}: stale={stale:.1f}h, gaps={gaps}, NaNs={nans}")
            else:
                print(f"  {key}: stale=N/A, gaps={gaps}, NaNs={nans}")
            if isinstance(stale, (int, float)) and stale > MAX_STALE_HOURS:
                print(f"    WARNING: data is {stale:.1f}h stale (threshold: {MAX_STALE_HOURS}h)")

        result = refresh_klines(symbol, dry_run=args.dry_run)
        all_results.append(result)
        if not result.get("success"):
            failures.append(f"{symbol} klines: {result.get('error', 'unknown')}")
            print(f"  FAILED: {result.get('error', 'unknown')}")
        else:
            print(f"  OK (rows={result.get('rows', '?')})")

    # 2. Funding rates
    for symbol in symbols:
        print(f"\n-- {symbol} funding rates --")
        result = refresh_funding_rates(symbol, dry_run=args.dry_run)
        all_results.append(result)
        if not result.get("success"):
            failures.append(f"{symbol} funding: {result.get('error', 'unknown')}")
            print(f"  FAILED: {result.get('error', 'unknown')}")
        else:
            print("  OK")

    # 3. External data
    if not args.skip_external:
        print("\n-- External data --")
        ext_results = refresh_external_data(dry_run=args.dry_run)
        for r in ext_results:
            all_results.append(r)
            src = r.get("source", "unknown")
            if not r.get("success"):
                failures.append(f"{src}: {r.get('error', 'unknown')}")
                print(f"  {src}: FAILED ({r.get('error', 'unknown')})")
            else:
                print(f"  {src}: OK")

    # Summary
    print(f"\n{'=' * 70}")
    print("  REFRESH SUMMARY")
    print(f"{'=' * 70}")
    success_count = sum(1 for r in all_results if r.get("success"))
    total = len(all_results)
    print(f"  {success_count}/{total} succeeded")

    if failures:
        print("  FAILURES:")
        for f in failures:
            print(f"    - {f}")

        if args.alert:
            send_alert(
                f"Data refresh: {len(failures)} failures: {'; '.join(failures[:3])}",
                severity="error",
            )

    # Post-refresh validation
    if not args.dry_run and not failures:
        print("\n-- Post-refresh validation --")
        for symbol in symbols:
            freshness = check_data_freshness(symbol)
            for key, info in freshness.items():
                stale = info.get("stale_hours", 0)
                if isinstance(stale, (int, float)) and stale != float("inf"):
                    status = "OK" if stale <= MAX_STALE_HOURS else "STALE"
                    print(f"  {symbol}/{key}: {status} (stale={stale:.1f}h)")
                else:
                    print(f"  {symbol}/{key}: NO DATA")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
