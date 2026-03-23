"""Live Validation Dashboard — 30-day demo trading verification.

Aggregates health, IC, track record, and log data into a unified
validation status with a 30-day go-live checklist.

Usage:
    python3 -m scripts.ops.live_validation_dashboard
    python3 -m scripts.ops.live_validation_dashboard --json
    python3 -m scripts.ops.live_validation_dashboard --markdown
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Data sources
HEALTH_STATUS = Path("data/runtime/health_status.json")
IC_HEALTH = Path("data/runtime/ic_health.json")
TRACK_RECORD = Path("data/live/track_record.json")
SIGNAL_RECONCILE = Path("data/runtime/signal_reconcile.json")
LOG_FILE = Path("logs/bybit_alpha.log")


def _load_json(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _parse_track_record(data: Dict) -> Dict[str, Any]:
    """Extract summary metrics from track_record.json."""
    daily = data.get("daily", {})
    if not daily:
        return {"days": 0, "total_pnl": 0, "total_trades": 0, "max_dd": 0}

    sorted_dates = sorted(daily.keys())
    n_days = len(sorted_dates)

    total_pnl = 0.0
    total_trades = 0
    max_dd = 0.0
    daily_pnls = []
    per_runner: Dict[str, Dict] = {}

    for date_str in sorted_dates:
        day = daily[date_str]
        day_pnl = day.get("total_pnl_usd", 0.0)
        day_trades = day.get("total_trades", 0)
        day_dd = day.get("max_drawdown", 0.0)

        total_pnl += day_pnl
        total_trades += day_trades
        max_dd = max(max_dd, day_dd)
        daily_pnls.append(day_pnl)

        # Per-runner stats
        for sym, sym_data in day.get("symbols", {}).items():
            if sym not in per_runner:
                per_runner[sym] = {"signals": 0, "trades": 0, "pnl": 0.0, "bars": 0}
            per_runner[sym]["signals"] += sum(sym_data.get("signals", {}).values())
            per_runner[sym]["trades"] += sym_data.get("trades", 0)
            per_runner[sym]["pnl"] += sym_data.get("pnl_usd", 0.0)
            per_runner[sym]["bars"] += sym_data.get("bars", 0)

    # Rolling 30-day Sharpe
    sharpe_30d = 0.0
    if len(daily_pnls) >= 7:
        arr = np.array(daily_pnls[-30:])
        if np.std(arr) > 0:
            sharpe_30d = float(np.mean(arr) / np.std(arr) * np.sqrt(365))

    first_date = sorted_dates[0] if sorted_dates else "N/A"
    last_date = sorted_dates[-1] if sorted_dates else "N/A"

    return {
        "days": n_days,
        "first_date": first_date,
        "last_date": last_date,
        "total_pnl": total_pnl,
        "total_trades": total_trades,
        "max_dd": max_dd,
        "sharpe_30d": sharpe_30d,
        "daily_pnls": daily_pnls,
        "per_runner": per_runner,
    }


def _check_ic_health(data: Optional[Dict]) -> Dict[str, str]:
    """Extract per-model IC status."""
    if not data:
        return {}
    models = data.get("models", [])
    result = {}
    if isinstance(models, list):
        for info in models:
            name = info.get("model", info.get("symbol", "unknown"))
            result[name] = info.get("overall_status", info.get("status", "UNKNOWN"))
    elif isinstance(models, dict):
        for model_name, info in models.items():
            if isinstance(info, dict):
                result[model_name] = info.get("status", info.get("level", "UNKNOWN"))
            else:
                result[model_name] = str(info)
    return result


def _check_health_status(data: Optional[Dict]) -> str:
    """Extract overall health from health_status.json."""
    if not data:
        return "UNKNOWN"
    checks = data.get("checks", {})
    problems = []
    for name, info in checks.items():
        if isinstance(info, dict) and info.get("problems"):
            problems.extend(info["problems"])
    return "HEALTHY" if not problems else f"ISSUES ({len(problems)})"


def _check_signal_reconcile(data: Optional[Dict]) -> float:
    """Extract signal match rate."""
    if not data:
        return -1.0
    return data.get("match_rate", data.get("overall_match_rate", -1.0))


def build_dashboard() -> Dict[str, Any]:
    """Build complete validation dashboard data."""
    # Load sources
    health_data = _load_json(HEALTH_STATUS)
    ic_data = _load_json(IC_HEALTH)
    track_data = _load_json(TRACK_RECORD)
    reconcile_data = _load_json(SIGNAL_RECONCILE)

    # Parse track record
    tr = _parse_track_record(track_data) if track_data else {
        "days": 0, "total_pnl": 0, "total_trades": 0, "max_dd": 0,
        "sharpe_30d": 0, "per_runner": {}, "daily_pnls": [],
    }

    # IC health
    ic_status = _check_ic_health(ic_data)

    # Health
    health = _check_health_status(health_data)

    # Signal reconcile
    match_rate = _check_signal_reconcile(reconcile_data)

    # 30-day checklist
    checklist = {
        "Sharpe > 1.0 (30d rolling)": tr["sharpe_30d"] > 1.0,
        "Signal match rate > 85%": match_rate > 85.0 if match_rate >= 0 else False,
        "IC not decayed (GREEN)": all(s == "GREEN" for s in ic_status.values()) if ic_status else False,
        "MaxDD < 20%": tr["max_dd"] < 20.0,
        "No unhandled crashes": health in ("HEALTHY", "UNKNOWN"),
        "At least 50 trades": tr["total_trades"] >= 50,
    }
    all_pass = all(checklist.values())

    now = datetime.now(timezone.utc)

    return {
        "timestamp": now.isoformat(),
        "days_running": tr["days"],
        "date_range": f"{tr.get('first_date', 'N/A')} → {tr.get('last_date', 'N/A')}",
        "total_pnl_usd": tr["total_pnl"],
        "total_trades": tr["total_trades"],
        "sharpe_30d": tr["sharpe_30d"],
        "max_drawdown_pct": tr["max_dd"],
        "health_status": health,
        "signal_match_rate": match_rate,
        "ic_status": ic_status,
        "per_runner": tr["per_runner"],
        "checklist": checklist,
        "all_pass": all_pass,
    }


def print_terminal(dashboard: Dict[str, Any]) -> None:
    """Print dashboard as terminal table."""
    print(f"\n{'='*70}")
    print("  LIVE VALIDATION DASHBOARD")
    print(f"{'='*70}")
    print(f"  Date:           {dashboard['timestamp'][:19]}")
    print(f"  Days running:   {dashboard['days_running']}")
    print(f"  Date range:     {dashboard['date_range']}")
    print(f"  Health:         {dashboard['health_status']}")
    print("\n  --- Performance ---")
    print(f"  Total PnL:      ${dashboard['total_pnl_usd']:+,.2f}")
    print(f"  Total trades:   {dashboard['total_trades']}")
    print(f"  Sharpe (30d):   {dashboard['sharpe_30d']:.2f}")
    print(f"  Max drawdown:   {dashboard['max_drawdown_pct']:.1f}%")
    mr = dashboard['signal_match_rate']
    print(f"  Signal match:   {mr:.1f}%" if mr >= 0 else "  Signal match:   N/A")

    if dashboard['per_runner']:
        print("\n  --- Per Runner ---")
        print(f"  {'Runner':<18} {'Bars':>6} {'Signals':>8} {'Trades':>7} {'PnL':>10}")
        for name, stats in sorted(dashboard['per_runner'].items()):
            print(f"  {name:<18} {stats['bars']:>6} {stats['signals']:>8} "
                  f"{stats['trades']:>7} ${stats['pnl']:>+9.2f}")

    if dashboard['ic_status']:
        print("\n  --- IC Health ---")
        for model, status in dashboard['ic_status'].items():
            print(f"  {model}: {status}")

    print("\n  --- 30-Day Validation Checklist ---")
    for check, passed in dashboard['checklist'].items():
        mark = "PASS" if passed else "FAIL"
        print(f"  [{mark}] {check}")

    verdict = "PASS" if dashboard['all_pass'] else "FAIL"
    print(f"\n  VERDICT: {verdict}")
    print(f"{'='*70}")


def print_markdown(dashboard: Dict[str, Any]) -> None:
    """Print dashboard as Markdown."""
    print("# Live Validation Dashboard\n")
    print(f"**Date:** {dashboard['timestamp'][:19]}  ")
    print(f"**Days running:** {dashboard['days_running']}  ")
    print(f"**Range:** {dashboard['date_range']}\n")

    print("## Performance\n")
    print("| Metric | Value |")
    print("|--------|-------|")
    print(f"| Total PnL | ${dashboard['total_pnl_usd']:+,.2f} |")
    print(f"| Total trades | {dashboard['total_trades']} |")
    print(f"| Sharpe (30d) | {dashboard['sharpe_30d']:.2f} |")
    print(f"| Max drawdown | {dashboard['max_drawdown_pct']:.1f}% |")
    mr = dashboard['signal_match_rate']
    print(f"| Signal match | {mr:.1f}% |" if mr >= 0 else "| Signal match | N/A |")
    print(f"| Health | {dashboard['health_status']} |")

    if dashboard['per_runner']:
        print("\n## Per Runner\n")
        print("| Runner | Bars | Signals | Trades | PnL |")
        print("|--------|------|---------|--------|-----|")
        for name, stats in sorted(dashboard['per_runner'].items()):
            print(f"| {name} | {stats['bars']} | {stats['signals']} | "
                  f"{stats['trades']} | ${stats['pnl']:+.2f} |")

    print("\n## 30-Day Checklist\n")
    for check, passed in dashboard['checklist'].items():
        mark = "x" if passed else " "
        print(f"- [{mark}] {check}")

    verdict = "PASS" if dashboard['all_pass'] else "FAIL"
    print(f"\n**Verdict: {verdict}**")


def main() -> None:
    parser = argparse.ArgumentParser(description="Live Validation Dashboard")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    parser.add_argument("--markdown", action="store_true", help="Output Markdown")
    args = parser.parse_args()

    dashboard = build_dashboard()

    if args.json:
        print(json.dumps(dashboard, indent=2, default=str))
    elif args.markdown:
        print_markdown(dashboard)
    else:
        print_terminal(dashboard)


if __name__ == "__main__":
    main()
