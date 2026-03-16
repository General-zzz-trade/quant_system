#!/usr/bin/env python3
"""Pre-live automated checklist -- verifies system readiness before going live.

Runs a series of automated checks (API keys, models, burn-in, safety limits)
and prints a clear pass/fail report. Intended to be run before switching
from testnet/demo to live trading.

Usage:
    python3 -m scripts.ops.pre_live_checklist
    python3 -m scripts.ops.pre_live_checklist --json
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


def check_api_keys() -> Tuple[bool, Any]:
    """Verify required API keys are set."""
    required = ["BYBIT_API_KEY", "BYBIT_API_SECRET"]
    missing = [k for k in required if not os.environ.get(k)]
    return len(missing) == 0, missing


def check_max_order_notional() -> Tuple[bool, Any]:
    """Verify MAX_ORDER_NOTIONAL is set to a safe value."""
    try:
        from scripts.ops.config import MAX_ORDER_NOTIONAL
    except ImportError:
        return False, "cannot import config"
    return MAX_ORDER_NOTIONAL <= 1000, MAX_ORDER_NOTIONAL


def check_burnin_passed(
    report_path: str = "data/live/burnin_report.json",
) -> Tuple[bool, Any]:
    """Verify all burn-in phases passed."""
    path = Path(report_path)
    if not path.exists():
        return False, "no burn-in report"
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        return False, f"read error: {exc}"

    phases: Dict[str, bool] = {}
    if isinstance(data, list):
        # List of phase reports (from burnin_report.py aggregate)
        for entry in data:
            phases[entry.get("phase", "?")] = entry.get("passed", False)
    elif isinstance(data, dict):
        # Dict with "phases" key (alternative format)
        if "phases" in data:
            for phase_key, phase_data in data["phases"].items():
                if isinstance(phase_data, dict):
                    phases[phase_key] = phase_data.get("status") == "passed"
                else:
                    phases[phase_key] = bool(phase_data)
        else:
            # Single phase report
            phases[data.get("phase", "?")] = data.get("passed", False)

    all_passed = all(phases.get(p, False) for p in ("A", "B", "C"))
    return all_passed, phases


def check_models_exist(
    models_dir: str = "models_v8",
) -> Tuple[bool, Any]:
    """Verify production models exist."""
    base = Path(models_dir)
    required = ["ETHUSDT_gate_v2", "BTCUSDT_gate_v2"]
    missing = [m for m in required if not (base / m).exists()]
    return len(missing) == 0, missing


def check_systemd_service() -> Tuple[bool, Any]:
    """Verify systemd service is configured."""
    try:
        result = subprocess.run(
            ["systemctl", "cat", "bybit-alpha.service"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0, ""
    except FileNotFoundError:
        return False, "systemctl not found"
    except Exception as exc:
        return False, str(exc)


def check_log_directory() -> Tuple[bool, Any]:
    """Verify log directory exists and is writable."""
    log_dir = Path("logs")
    if not log_dir.exists():
        return False, "logs/ directory missing"
    if not os.access(log_dir, os.W_OK):
        return False, "logs/ not writable"
    return True, str(log_dir.resolve())


def check_data_files() -> Tuple[bool, Any]:
    """Verify essential data files exist."""
    data_dir = Path("data_files")
    required = ["ETHUSDT_1h.csv", "BTCUSDT_1h.csv"]
    missing = [f for f in required if not (data_dir / f).exists()]
    return len(missing) == 0, missing


# All checks as (name, callable) pairs
ALL_CHECKS: List[Tuple[str, Any]] = [
    ("API keys present", check_api_keys),
    ("MAX_ORDER_NOTIONAL safe", check_max_order_notional),
    ("Burn-in phases passed", check_burnin_passed),
    ("Production models exist", check_models_exist),
    ("Systemd service configured", check_systemd_service),
    ("Log directory ready", check_log_directory),
    ("Data files present", check_data_files),
]


def run_checks(
    checks: List[Tuple[str, Any]] | None = None,
) -> List[Dict[str, Any]]:
    """Run all checks and return results."""
    if checks is None:
        checks = ALL_CHECKS
    results = []
    for name, check_fn in checks:
        try:
            ok, detail = check_fn()
            results.append({
                "name": name,
                "status": "PASS" if ok else "FAIL",
                "detail": detail if not ok else "ok",
            })
        except Exception as exc:
            results.append({
                "name": name,
                "status": "SKIP",
                "detail": str(exc),
            })
    return results


def format_checklist(results: List[Dict[str, Any]]) -> str:
    """Format checklist results for display."""
    lines = [
        "=" * 60,
        "  PRE-LIVE CHECKLIST",
        "=" * 60,
        "",
    ]

    all_pass = True
    for r in results:
        status = r["status"]
        name = r["name"]
        detail = r["detail"]
        lines.append(f"  [{status:4s}] {name}: {detail}")
        if status == "FAIL":
            all_pass = False

    lines.append("")
    lines.append("  Manual checks required:")
    lines.append("  [ ] Confirm initial capital deposited")
    lines.append("  [ ] Confirm risk parameters reviewed")
    lines.append("  [ ] Confirm monitoring alerts configured")
    lines.append("  [ ] Confirm BYBIT_BASE_URL points to production")
    lines.append("")

    overall = "PASS" if all_pass else "FAIL"
    lines.append(f"  AUTOMATED: {overall}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Pre-live automated checklist"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )
    args = parser.parse_args()

    results = run_checks()

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print(format_checklist(results))

    all_pass = all(r["status"] == "PASS" for r in results)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
