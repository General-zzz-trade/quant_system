#!/usr/bin/env python3
"""Polymarket maker health check.

Checks:
1. Collector service running + DB freshness
2. Maker process running
3. DB data quality (recent samples, gap detection)
4. Dryrun trade summary

Usage:
    python3 scripts/polymarket_health.py
    python3 scripts/polymarket_health.py --json
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_DIR / "data" / "polymarket" / "collector.db"
DRYRUN_PATH = PROJECT_DIR / "data" / "polymarket" / "dryrun_trades.csv"
MAKER_LOG = PROJECT_DIR / "logs" / "polymarket_maker.log"


def check_process(name: str) -> dict:
    """Check if a process matching *name* is running."""
    try:
        out = subprocess.check_output(
            ["pgrep", "-af", name], text=True, stderr=subprocess.DEVNULL
        ).strip()
        pids = [line.split()[0] for line in out.splitlines() if line]
        return {"running": True, "pids": pids}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"running": False, "pids": []}


def check_systemd(service: str) -> dict:
    """Check systemd service status."""
    try:
        result = subprocess.run(
            ["systemctl", "is-active", service],
            capture_output=True, text=True, timeout=5
        )
        active = result.stdout.strip() == "active"
        return {"service": service, "active": active, "status": result.stdout.strip()}
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return {"service": service, "active": False, "status": "unknown"}


def check_db() -> dict:
    """Check collector DB freshness and stats."""
    if not DB_PATH.exists():
        return {"exists": False, "error": f"DB not found: {DB_PATH}"}

    conn = sqlite3.connect(str(DB_PATH))
    result: dict = {"exists": True}

    try:
        # Total records
        row = conn.execute(
            "SELECT COUNT(*), MIN(timestamp_utc), MAX(timestamp_utc) FROM market_snapshots"
        ).fetchone()
        result["snapshots"] = {
            "count": row[0],
            "first": row[1],
            "last": row[2],
        }

        # Check freshness
        if row[2]:
            try:
                ts_str = row[2]
                # Handle both naive and timezone-aware timestamps
                if "+" not in ts_str and "Z" not in ts_str:
                    last_dt = datetime.fromisoformat(ts_str).replace(tzinfo=timezone.utc)
                else:
                    last_dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                age_min = (datetime.now(timezone.utc) - last_dt).total_seconds() / 60
                result["snapshots"]["age_minutes"] = round(age_min, 1)
                result["snapshots"]["fresh"] = age_min < 30  # expect data every 15m
            except (ValueError, TypeError):
                result["snapshots"]["fresh"] = None

        # Intra-window V2 samples
        v2_row = conn.execute("SELECT COUNT(*) FROM intra_window_samples_v2").fetchone()
        result["intra_v2_count"] = v2_row[0] if v2_row else 0

        # 15m samples
        s15_row = conn.execute("SELECT COUNT(*) FROM samples_15m").fetchone()
        result["samples_15m_count"] = s15_row[0] if s15_row else 0

        # Average pricing delay (data quality indicator)
        delay_row = conn.execute(
            "SELECT AVG(ABS(pricing_delay)), MAX(ABS(pricing_delay)) "
            "FROM intra_window_samples_v2 WHERE pricing_delay IS NOT NULL"
        ).fetchone()
        if delay_row and delay_row[0] is not None:
            result["pricing_delay"] = {
                "avg": round(delay_row[0], 4),
                "max": round(delay_row[1], 4),
            }

        # Settlement result distribution
        results_rows = conn.execute(
            "SELECT polymarket_result, COUNT(*) FROM market_snapshots "
            "WHERE polymarket_result IS NOT NULL GROUP BY polymarket_result"
        ).fetchall()
        result["settlement_results"] = {r[0]: r[1] for r in results_rows}

    except sqlite3.OperationalError as e:
        result["error"] = str(e)
    finally:
        conn.close()

    return result


def check_dryrun() -> dict:
    """Check dryrun trade log."""
    if not DRYRUN_PATH.exists():
        return {"exists": False}

    try:
        lines = DRYRUN_PATH.read_text().strip().splitlines()
        if len(lines) <= 1:
            return {"exists": True, "trade_count": 0}

        trade_count = len(lines) - 1  # subtract header
        last_line = lines[-1] if trade_count > 0 else ""
        return {
            "exists": True,
            "trade_count": trade_count,
            "last_trade": last_line[:200],  # truncate
        }
    except Exception as e:
        return {"exists": True, "error": str(e)}


def check_maker_log() -> dict:
    """Check maker log for recent activity and errors."""
    if not MAKER_LOG.exists():
        return {"exists": False}

    try:
        # Read last 50 lines
        result = subprocess.run(
            ["tail", "-n", "50", str(MAKER_LOG)],
            capture_output=True, text=True, timeout=5
        )
        lines = result.stdout.strip().splitlines()
        error_lines = [line for line in lines if "ERROR" in line or "CRITICAL" in line]
        signal_lines = [line for line in lines if "POLYMARKET_SIGNAL" in line]

        return {
            "exists": True,
            "recent_lines": len(lines),
            "recent_errors": len(error_lines),
            "recent_signals": len(signal_lines),
            "last_error": error_lines[-1][:200] if error_lines else None,
            "last_signal": signal_lines[-1][:200] if signal_lines else None,
        }
    except Exception as e:
        return {"exists": True, "error": str(e)}


def run_health_check() -> dict:
    """Run all health checks and return combined report."""
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "collector_service": check_systemd("polymarket-collector"),
        "maker_service": check_systemd("polymarket-maker"),
        "collector_process": check_process("polymarket.collector"),
        "maker_process": check_process("run_polymarket_maker"),
        "database": check_db(),
        "dryrun": check_dryrun(),
        "maker_log": check_maker_log(),
    }

    # Overall status
    issues = []
    if not report["collector_service"]["active"] and not report["collector_process"]["running"]:
        issues.append("Collector not running")
    db = report["database"]
    if not db.get("exists"):
        issues.append("Collector DB missing")
    elif db.get("snapshots", {}).get("fresh") is False:
        age = db["snapshots"].get("age_minutes", "?")
        issues.append(f"DB stale ({age} min since last snapshot)")
    if not report["maker_service"]["active"] and not report["maker_process"]["running"]:
        issues.append("Maker not running")

    report["overall"] = "HEALTHY" if not issues else "DEGRADED"
    report["issues"] = issues

    return report


def print_report(report: dict) -> None:
    """Print human-readable health report."""
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    RESET = "\033[0m"

    status_color = GREEN if report["overall"] == "HEALTHY" else RED
    print(f"\n{'='*50}")
    print(" Polymarket Health Check")
    print(f" {report['timestamp']}")
    print(f"{'='*50}")
    print(f"\n Status: {status_color}{report['overall']}{RESET}")

    if report["issues"]:
        for issue in report["issues"]:
            print(f"   {RED}[!]{RESET} {issue}")

    # Collector
    coll = report["collector_service"]
    color = GREEN if coll["active"] else YELLOW
    print(f"\n Collector: {color}{coll['status']}{RESET}")

    # Maker
    maker = report["maker_service"]
    color = GREEN if maker["active"] else YELLOW
    print(f" Maker:     {color}{maker['status']}{RESET}")

    # Database
    db = report["database"]
    if db.get("exists"):
        snaps = db.get("snapshots", {})
        print("\n Database:")
        print(f"   Snapshots:     {snaps.get('count', 0)}")
        print(f"   Intra V2:     {db.get('intra_v2_count', 0)}")
        print(f"   15m samples:  {db.get('samples_15m_count', 0)}")
        if snaps.get("last"):
            fresh = snaps.get("fresh")
            color = GREEN if fresh else RED
            age = snaps.get("age_minutes", "?")
            print(f"   Last data:    {snaps['last']} ({color}{age} min ago{RESET})")
        if db.get("pricing_delay"):
            pd = db["pricing_delay"]
            print(f"   Pricing delay: avg={pd['avg']:.4f} max={pd['max']:.4f}")
        if db.get("settlement_results"):
            print(f"   Settlements:  {db['settlement_results']}")
    else:
        print(f"\n {RED}Database: NOT FOUND{RESET}")

    # Dryrun
    dr = report["dryrun"]
    if dr.get("exists") and dr.get("trade_count", 0) > 0:
        print(f"\n Dryrun trades: {dr['trade_count']}")
        if dr.get("last_trade"):
            print(f"   Last: {dr['last_trade'][:100]}")

    # Maker log
    ml = report["maker_log"]
    if ml.get("exists"):
        print("\n Maker log (last 50 lines):")
        print(f"   Errors:  {ml.get('recent_errors', 0)}")
        print(f"   Signals: {ml.get('recent_signals', 0)}")
        if ml.get("last_error"):
            print(f"   Last error:  {ml['last_error'][:100]}")
        if ml.get("last_signal"):
            print(f"   Last signal: {ml['last_signal'][:100]}")

    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Polymarket maker health check")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    os.chdir(str(PROJECT_DIR))
    report = run_health_check()

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_report(report)

    # Exit code: 0 = healthy, 1 = degraded
    sys.exit(0 if report["overall"] == "HEALTHY" else 1)


if __name__ == "__main__":
    main()
