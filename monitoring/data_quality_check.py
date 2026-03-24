#!/usr/bin/env python3
"""Data quality check -- validates OHLCV data for anomalies, gaps, and consistency.

Usage:
    python3 -m monitoring.data_quality_check                  # All CSV data files
    python3 -m monitoring.data_quality_check --symbol BTCUSDT # Single symbol
    python3 -m monitoring.data_quality_check --alert           # Send Telegram on failures
    python3 -m monitoring.data_quality_check --json            # JSON output
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.quality.gaps import GapDetector
from data.quality.validators import BarValidator
from data.store import Bar

logger = logging.getLogger(__name__)

DATA_DIR = Path("data_files")

# Map filename patterns to interval seconds
INTERVAL_MAP: dict[str, int] = {
    "_1h": 3600,
    "_15m": 900,
    "_5m": 300,
    "_4h": 14400,
}


def _detect_ts_column(columns: list[str]) -> str | None:
    """Find the timestamp column in a CSV."""
    for col in ["open_time", "timestamp", "ts", "time"]:
        if col in columns:
            return col
    return None


def _detect_interval_seconds(filename: str) -> int:
    """Infer interval from filename suffix."""
    for suffix, seconds in INTERVAL_MAP.items():
        if suffix in filename:
            return seconds
    # Default to 1h for files like BTCUSDT_funding.csv etc.
    return 3600


def _is_ohlcv_file(columns: list[str]) -> bool:
    """Check if a CSV has OHLCV columns."""
    required = {"open", "high", "low", "close"}
    return required.issubset(set(columns))


def check_file(filepath: Path, symbol: str = "") -> dict[str, Any]:
    """Run quality checks on a single CSV file.

    Returns a dict with validation results, gap report, and metadata.
    """
    import pandas as pd

    result: dict[str, Any] = {
        "file": filepath.name,
        "symbol": symbol,
        "pass": True,
        "errors": [],
        "warnings": [],
        "stats": {},
    }

    try:
        df = pd.read_csv(filepath)
    except Exception as exc:
        result["pass"] = False
        result["errors"].append(f"Failed to read CSV: {exc}")
        return result

    if df.empty:
        result["warnings"].append("Empty file")
        result["stats"]["rows"] = 0
        return result

    result["stats"]["rows"] = len(df)

    # Check for NaN values in numeric columns
    numeric_cols = df.select_dtypes(include=["number"]).columns
    nan_total = int(df[numeric_cols].isna().sum().sum())
    result["stats"]["nan_count"] = nan_total
    if nan_total > 0:
        result["warnings"].append(f"{nan_total} NaN values in numeric columns")

    # Only run OHLCV checks on files that have OHLCV columns
    if not _is_ohlcv_file(list(df.columns)):
        result["stats"]["type"] = "non-ohlcv"
        return result

    result["stats"]["type"] = "ohlcv"

    ts_col = _detect_ts_column(list(df.columns))
    if ts_col is None:
        result["warnings"].append("No timestamp column found — skipping time checks")
        return result

    interval_seconds = _detect_interval_seconds(filepath.name)

    # Convert to Bar objects
    store_bars: list[Bar] = []
    conversion_errors = 0
    for _, row in df.iterrows():
        try:
            ts_val = float(row[ts_col])
            if ts_val > 1e12:
                ts_val = ts_val / 1000
            ts_dt = datetime.fromtimestamp(ts_val, tz=timezone.utc)
            store_bars.append(Bar(
                ts=ts_dt,
                open=Decimal(str(row["open"])),
                high=Decimal(str(row["high"])),
                low=Decimal(str(row["low"])),
                close=Decimal(str(row["close"])),
                volume=Decimal(str(row.get("volume", 0))) if "volume" in row.index else None,
                symbol=symbol,
            ))
        except (KeyError, ValueError, InvalidOperation):
            conversion_errors += 1

    if conversion_errors > 0:
        result["warnings"].append(f"{conversion_errors} rows failed conversion")

    if not store_bars:
        result["pass"] = False
        result["errors"].append("No valid bars after conversion")
        return result

    # BarValidator
    validator = BarValidator(
        zscore_threshold=5.0,
        max_gap_seconds=interval_seconds * 2,
    )
    val_result = validator.validate(store_bars)

    result["stats"]["total_bars"] = val_result.stats.get("total_bars", len(store_bars))
    result["stats"]["anomalies"] = val_result.stats.get("anomalies", 0)
    result["stats"]["ohlc_errors"] = val_result.stats.get("ohlc_errors", 0)
    result["stats"]["time_errors"] = val_result.stats.get("time_errors", 0)

    if not val_result.valid:
        result["pass"] = False
        result["errors"].extend(list(val_result.errors)[:10])
    result["warnings"].extend(list(val_result.warnings)[:10])

    # GapDetector
    gap_detector = GapDetector(interval_seconds=interval_seconds)
    gap_report = gap_detector.detect(
        store_bars,
        start=store_bars[0].ts,
        end=store_bars[-1].ts,
    )
    result["stats"]["gaps"] = len(gap_report.gaps)
    result["stats"]["completeness_pct"] = gap_report.completeness_pct

    if gap_report.gaps:
        result["warnings"].append(
            f"{len(gap_report.gaps)} gaps detected, "
            f"{gap_report.completeness_pct:.1f}% complete"
        )

    return result


def main() -> int:
    """Run data quality checks on all CSV data files."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(description="Data quality check")
    parser.add_argument("--symbol", default=None, help="Filter to specific symbol")
    parser.add_argument("--alert", action="store_true", help="Send Telegram alert on failures")
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--data-dir", default="data_files", help="Data directory")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        return 1

    # Find all CSV files
    csv_files = sorted(data_dir.glob("*.csv"))
    if args.symbol:
        csv_files = [f for f in csv_files if args.symbol.upper() in f.name.upper()]

    if not csv_files:
        print("No CSV files found")
        return 0

    results: list[dict[str, Any]] = []
    total_pass = 0
    total_fail = 0
    total_warnings = 0

    if not args.json:
        print("=" * 70)
        print("  DATA QUALITY CHECK")
        print(f"  Date:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Files:    {len(csv_files)}")
        print("=" * 70)

    for csv_file in csv_files:
        # Infer symbol from filename (e.g., BTCUSDT_1h.csv -> BTCUSDT)
        symbol = csv_file.stem.split("_")[0] if "_" in csv_file.stem else csv_file.stem

        file_result = check_file(csv_file, symbol=symbol)
        results.append(file_result)

        if file_result["pass"]:
            total_pass += 1
        else:
            total_fail += 1
        total_warnings += len(file_result["warnings"])

        if not args.json:
            status = "PASS" if file_result["pass"] else "FAIL"
            stats = file_result["stats"]
            row_info = f"rows={stats.get('rows', '?')}"

            if stats.get("type") == "ohlcv":
                detail = (
                    f"{row_info}, "
                    f"gaps={stats.get('gaps', 0)}, "
                    f"anomalies={stats.get('anomalies', 0)}, "
                    f"complete={stats.get('completeness_pct', 100):.1f}%"
                )
            else:
                detail = f"{row_info}, NaN={stats.get('nan_count', 0)}"

            print(f"\n  {csv_file.name}: {status}")
            print(f"    {detail}")

            if file_result["errors"]:
                for err in file_result["errors"][:3]:
                    print(f"    ERROR: {err}")
            if file_result["warnings"]:
                for warn in file_result["warnings"][:3]:
                    print(f"    WARN:  {warn}")

    # Summary
    if args.json:
        summary = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "total_files": len(csv_files),
            "pass": total_pass,
            "fail": total_fail,
            "warnings": total_warnings,
            "results": results,
        }
        print(json.dumps(summary, indent=2, default=str))
    else:
        print(f"\n{'=' * 70}")
        print("  SUMMARY")
        print(f"{'=' * 70}")
        print(f"  PASS: {total_pass}  FAIL: {total_fail}  WARNINGS: {total_warnings}")

    # Send alert on failures
    if args.alert and total_fail > 0:
        try:
            from monitoring.notify import send_notification
            failed_files = [r["file"] for r in results if not r["pass"]]
            msg = (
                f"Data quality: {total_fail} files FAILED "
                f"({', '.join(failed_files[:5])})"
            )
            send_notification(msg, severity="warning")
        except ImportError:
            # Fallback: direct Telegram
            _send_telegram_alert(
                f"Data quality: {total_fail}/{len(csv_files)} files FAILED"
            )

    return 1 if total_fail > 0 else 0


def _send_telegram_alert(message: str) -> None:
    """Fallback Telegram alert."""
    import urllib.request

    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        return
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = json.dumps({"chat_id": chat_id, "text": f"[WARN] {message}"}).encode()
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10)
    except Exception as exc:
        logger.warning("Telegram alert failed: %s", exc)


if __name__ == "__main__":
    sys.exit(main())
