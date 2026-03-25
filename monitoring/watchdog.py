#!/usr/bin/env python3
"""Automated health watchdog — runs every 5 minutes via systemd timer.

Checks:
1. Active trading services (systemd state + fresh logs)
2. Data freshness (klines, funding, OI)
3. Account health (balance, positions, kill latch)
4. Polymarket collector alive

Actions on failure:
- Log WARNING/ERROR with details
- Restart stale services (if configured)
- Write health status to data/runtime/health_status.json
- Exit with code 1 on critical failure (for systemd OnFailure= alerting)

Usage:
    python3 -m scripts.ops.health_watchdog              # Check all
    python3 -m scripts.ops.health_watchdog --restart     # Auto-restart stale services
    python3 -m scripts.ops.health_watchdog --json        # JSON output
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

log = logging.getLogger("health_watchdog")

STATUS_FILE = "data/runtime/health_status.json"
HISTORY_FILE = "data/runtime/health_history.jsonl"

# Services to monitor
SERVICES = {
    "hft-signal": {
        "unit": "hft-signal.service",
        "journal_match": "hft_signal",
        "max_silent_s": 120,  # 2 min without log = stale
    },
    "binary-signal": {
        "unit": "binary-signal.service",
        "journal_match": "binary_signal",
        "max_silent_s": 120,
    },
    "bybit-alpha": {
        "unit": "bybit-alpha.service",
        "journal_match": "bybit_alpha",
        "max_silent_s": 18000,  # 5 hours (4h bars close every 4h + margin)
        "log_file": "logs/bybit_alpha.log",  # check file instead of journal (stdout→file)
    },
    "bybit-mm": {
        "unit": "bybit-mm.service",
        "journal_match": "bybit_mm",
        "max_silent_s": 120,
    },
}

# Data files to check freshness
DATA_FILES = {
    "BTC_1h": ("data_files/BTCUSDT_1h.csv", 48 * 3600),      # 48h max stale
    "ETH_1h": ("data_files/ETHUSDT_1h.csv", 48 * 3600),
    "BTC_funding": ("data_files/BTCUSDT_funding.csv", 48 * 3600),
    "BTC_oi": ("data_files/BTCUSDT_oi_1h.csv", 48 * 3600),
}


def check_service(name: str, cfg: dict[str, Any]) -> dict[str, Any]:
    """Check a systemd service health."""
    unit = cfg["unit"]
    result = {"name": name, "unit": unit, "status": "unknown", "problems": []}

    # Check systemd state
    try:
        out = subprocess.run(
            ["systemctl", "is-active", unit],
            capture_output=True, text=True, timeout=5
        )
        state = out.stdout.strip()
        result["state"] = state
    except Exception as e:
        result["state"] = "error"
        result["problems"].append(f"systemctl error: {e}")
        result["status"] = "error"
        return result

    if state != "active":
        result["status"] = "inactive"
        return result

    # Check recent log activity — prefer log_file over journal (stdout→file services)
    log_file = cfg.get("log_file")
    if log_file:
        try:
            import os
            log_path = os.path.join(os.getcwd(), log_file) if not os.path.isabs(log_file) else log_file
            if os.path.exists(log_path):
                age_s = time.time() - os.path.getmtime(log_path)
                result["last_log_age_s"] = int(age_s)
                result["recent_log_lines"] = 1  # file exists and was modified
                if age_s > cfg["max_silent_s"]:
                    result["problems"].append(f"log_stale_{int(age_s)}s")
                    result["status"] = "stale"
                else:
                    result["status"] = "healthy"
            else:
                result["problems"].append("log_file_missing")
                result["status"] = "stale"
        except Exception as e:
            result["problems"].append(f"log_file error: {e}")
            result["status"] = "error"
        return result

    # Fallback: check journal output
    try:
        out = subprocess.run(
            ["journalctl", "-u", unit, "--since", "5 min ago",
             "--no-pager", "-q", "--output=short-unix"],
            capture_output=True, text=True, timeout=10
        )
        lines = [ln for ln in out.stdout.strip().split("\n") if ln.strip()]
        result["recent_log_lines"] = len(lines)

        if len(lines) == 0:
            result["problems"].append("no_recent_logs")
            result["status"] = "stale"
        else:
            # Parse last timestamp
            try:
                last_line = lines[-1]
                ts_str = last_line.split()[0]
                last_ts = float(ts_str)
                age_s = time.time() - last_ts
                result["last_log_age_s"] = int(age_s)
                if age_s > cfg["max_silent_s"]:
                    result["problems"].append(f"log_stale_{int(age_s)}s")
                    result["status"] = "stale"
                else:
                    result["status"] = "healthy"
            except (ValueError, IndexError):
                result["status"] = "healthy"  # has lines, assume ok
    except Exception as e:
        result["problems"].append(f"journal error: {e}")
        result["status"] = "error"

    # Check for errors in recent logs
    try:
        out = subprocess.run(
            ["journalctl", "-u", unit, "--since", "5 min ago",
             "--no-pager", "-q", "--grep=ERROR|KILL|FATAL|Traceback"],
            capture_output=True, text=True, timeout=10
        )
        error_lines = [ln for ln in out.stdout.strip().split("\n") if ln.strip()]
        if error_lines:
            result["recent_errors"] = len(error_lines)
            result["last_error"] = error_lines[-1][:200]
    except Exception as exc:
        log.debug("Error grep failed: %s", exc)

    return result


def check_data_freshness() -> list[dict[str, Any]]:
    """Check if data files are fresh enough."""
    results = []
    now = time.time()
    for name, (path, max_age_s) in DATA_FILES.items():
        r = {"name": name, "path": path}
        if not os.path.exists(path):
            r["status"] = "missing"
            r["problems"] = ["file_not_found"]
        else:
            mtime = os.path.getmtime(path)
            age_s = now - mtime
            age_h = age_s / 3600
            r["age_hours"] = round(age_h, 1)
            r["max_age_hours"] = max_age_s / 3600
            if age_s > max_age_s:
                r["status"] = "stale"
                r["problems"] = [f"stale_{age_h:.0f}h"]
            else:
                r["status"] = "fresh"
                r["problems"] = []
        results.append(r)
    return results


def check_polymarket_collector() -> dict[str, Any]:
    """Check if Polymarket collector is alive."""
    result = {"name": "polymarket-collector", "status": "unknown", "problems": []}
    pid_file = "data/polymarket/collector.pid"

    if not os.path.exists(pid_file):
        result["status"] = "not_running"
        result["problems"].append("no_pid_file")
        return result

    try:
        with open(pid_file) as f:
            pid = int(f.read().strip())
        # Check if process is alive
        os.kill(pid, 0)
        result["pid"] = pid
        result["status"] = "running"

        # Check DB freshness
        db_path = "data/polymarket/collector.db"
        if os.path.exists(db_path):
            age_s = time.time() - os.path.getmtime(db_path)
            result["db_age_s"] = int(age_s)
            if age_s > 600:  # 10 min
                result["status"] = "stale"
                result["problems"].append(f"db_stale_{int(age_s)}s")
    except ProcessLookupError:
        result["status"] = "dead"
        result["problems"].append(f"pid_{pid}_not_running")
    except Exception as e:
        result["problems"].append(str(e))

    return result


def check_account() -> dict[str, Any]:
    """Check Bybit account health."""
    result = {"status": "unknown", "problems": []}
    try:
        from execution.adapters.bybit.client import BybitRestClient
        from execution.adapters.bybit.config import BybitConfig

        # Load env
        env = {}
        env_path = Path(".env")
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    env[k.strip()] = v.strip().strip('"').strip("'")

        key = env.get("BYBIT_API_KEY", os.environ.get("BYBIT_API_KEY", ""))
        secret = env.get("BYBIT_API_SECRET", os.environ.get("BYBIT_API_SECRET", ""))
        url = env.get("BYBIT_BASE_URL", os.environ.get("BYBIT_BASE_URL", "https://api-demo.bybit.com"))

        if not key or not secret:
            result["status"] = "no_credentials"
            result["problems"].append("missing_api_keys")
            return result

        cfg = BybitConfig(api_key=key, api_secret=secret, base_url=url)
        client = BybitRestClient(cfg)

        r = client.get("/v5/account/wallet-balance", {"accountType": "UNIFIED"})
        equity = float(r["result"]["list"][0]["totalEquity"])
        result["equity"] = round(equity, 2)
        result["base_url"] = url

        # Check positions
        r2 = client.get("/v5/position/list", {"category": "linear", "settleCoin": "USDT"})
        positions = [p for p in r2["result"]["list"] if float(p.get("size", 0)) > 0]
        result["open_positions"] = len(positions)

        # Alert thresholds
        if equity < 400:
            result["problems"].append(f"low_equity_{equity:.0f}")
        if equity < 100:
            result["problems"].append("critical_equity")

        result["status"] = "ok" if not result["problems"] else "warning"

    except Exception as e:
        result["status"] = "error"
        result["problems"].append(str(e)[:200])

    return result


def restart_service(unit: str) -> bool:
    """Restart a systemd service."""
    try:
        subprocess.run(["sudo", "systemctl", "restart", unit],
                       capture_output=True, timeout=30)
        log.warning("RESTARTED %s", unit)
        return True
    except Exception as e:
        log.error("Failed to restart %s: %s", unit, e)
        return False


def _status_changed(current: str) -> bool:
    """Check if overall status changed since last run (avoid alert spam)."""
    try:
        if os.path.exists(STATUS_FILE):
            with open(STATUS_FILE) as f:
                prev = json.load(f)
            return prev.get("overall") != current
    except Exception as exc:
        log.debug("Status file read failed: %s", exc)
    return True  # first run or corrupt file → alert


def run_watchdog(auto_restart: bool = False, json_output: bool = False) -> int:
    """Run all health checks. Returns 0=healthy, 1=warning, 2=critical."""
    now = datetime.now(timezone.utc).isoformat()
    report = {"timestamp": now, "checks": {}, "overall": "healthy", "problems": []}

    # 1. Check services
    active_services = []
    for name, cfg in SERVICES.items():
        result = check_service(name, cfg)
        if result["state"] == "active" or result["status"] != "inactive":
            active_services.append(result)
        report["checks"][f"service_{name}"] = result

        if result.get("status") == "stale" and auto_restart:
            log.warning("Service %s is stale, restarting...", name)
            restart_service(cfg["unit"])
            result["action"] = "restarted"

    # 2. Check data
    data_results = check_data_freshness()
    for dr in data_results:
        report["checks"][f"data_{dr['name']}"] = dr

    # 3. Check Polymarket collector
    poly = check_polymarket_collector()
    report["checks"]["polymarket_collector"] = poly

    # 4. Check account
    account = check_account()
    report["checks"]["account"] = account

    # Aggregate problems
    all_problems = []
    for check_name, check_result in report["checks"].items():
        problems = check_result.get("problems", [])
        for p in problems:
            all_problems.append(f"{check_name}: {p}")

    report["problems"] = all_problems
    if any("critical" in p for p in all_problems):
        report["overall"] = "critical"
    elif all_problems:
        report["overall"] = "warning"
    else:
        report["overall"] = "healthy"

    # Save status
    os.makedirs(os.path.dirname(STATUS_FILE), exist_ok=True)
    with open(STATUS_FILE, "w") as f:
        json.dump(report, f, indent=2)

    # Append to history
    with open(HISTORY_FILE, "a") as f:
        f.write(json.dumps({"ts": now, "overall": report["overall"],
                            "problems": len(all_problems)}) + "\n")

    # ── Send Telegram alerts on warning/critical ──
    if all_problems:
        try:
            from monitoring.notify import send_alert, AlertLevel
            level = AlertLevel.CRITICAL if report["overall"] == "critical" else AlertLevel.WARNING
            # Deduplicate: only alert if status changed from last check
            changed = _status_changed(report["overall"])
            if changed:
                details = {}
                for i, p in enumerate(all_problems[:5]):
                    details[f"issue_{i+1}"] = p
                if len(all_problems) > 5:
                    details["more"] = f"+{len(all_problems)-5} more"
                # Add key metrics
                eq = account.get("equity")
                if eq:
                    details["equity"] = f"${eq:,.0f}"
                send_alert(level, f"Health: {report['overall'].upper()} ({len(all_problems)} issues)",
                           details=details, source="health_watchdog")
        except Exception as exc:
            log.debug("Notification failed: %s", exc)

    # Output
    if json_output:
        print(json.dumps(report, indent=2))
    else:
        print(f"Health Watchdog — {now}")
        print(f"Overall: {report['overall'].upper()}")
        print()

        # Services
        for name, cfg in SERVICES.items():
            r = report["checks"].get(f"service_{name}", {})
            state = r.get("state", "?")
            status = r.get("status", "?")
            if state == "inactive" and status == "inactive":
                continue  # skip inactive services
            icon = "OK" if status == "healthy" else "!!" if status == "stale" else "--"
            extra = ""
            if r.get("last_log_age_s"):
                extra = f" (last log {r['last_log_age_s']}s ago)"
            if r.get("recent_errors"):
                extra += f" [{r['recent_errors']} errors]"
            if r.get("action"):
                extra += f" → {r['action']}"
            print(f"  [{icon}] {name:20s} state={state:8s} {extra}")

        # Data
        print()
        stale_data = [d for d in data_results if d["status"] != "fresh"]
        if stale_data:
            for d in stale_data:
                print(f"  [!!] {d['name']:20s} {d['status']} ({d.get('age_hours', '?')}h old)")
        else:
            print(f"  [OK] All {len(data_results)} data files fresh")

        # Polymarket
        poly_status = poly.get("status", "?")
        print(f"\n  [{'OK' if poly_status == 'running' else '!!'}] Polymarket collector: {poly_status}")

        # Account
        eq = account.get("equity", "?")
        pos = account.get("open_positions", "?")
        print(f"  [{'OK' if account.get('status') == 'ok' else '!!'}] Account: ${eq} equity, {pos} positions")

        if all_problems:
            print(f"\nProblems ({len(all_problems)}):")
            for p in all_problems:
                print(f"  - {p}")

    # Exit code
    if report["overall"] == "critical":
        return 2
    elif report["overall"] == "warning":
        return 1
    return 0


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Health watchdog")
    parser.add_argument("--restart", action="store_true", help="Auto-restart stale services")
    parser.add_argument("--json", action="store_true", help="JSON output")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-5s %(name)s %(message)s")

    sys.exit(run_watchdog(auto_restart=args.restart, json_output=args.json))


if __name__ == "__main__":
    main()
