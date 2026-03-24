"""CLI and alerting functions for IC decay monitor.

Extracted from ic_decay_monitor.py: print_table, send_alerts, main.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def print_table(results: List[Dict[str, Any]]) -> None:
    """Print a formatted status table."""
    STATUS_COLORS = {"GREEN": "\033[92m", "YELLOW": "\033[93m", "RED": "\033[91m"}
    RESET = "\033[0m"

    print()
    print("=" * 88)
    print(f"{'IC Decay Monitor':^88}")
    print(f"{'(' + datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC') + ')':^88}")
    print("=" * 88)
    print(
        f"{'Model':<22} {'H':>3} {'Train IC':>10} "
        f"{'30d IC':>9} {'60d IC':>9} {'90d IC':>9} "
        f"{'Decay%':>8} {'Status':>8}"
    )
    print("-" * 88)

    for r in results:
        if "error" in r and "horizons" not in r:
            print(f"{r['model']:<22}  {'ERROR: ' + r['error']}")
            continue

        for hr in r.get("horizons", []):
            if "error" in hr:
                print(f"{r['model']:<22} {hr['horizon']:>3}  {hr['error']}")
                continue

            train_ic = hr.get("training_ic", 0)
            windows = hr.get("windows", {})
            ic_30 = windows.get("30d", {}).get("ic")
            ic_60 = windows.get("60d", {}).get("ic")
            ic_90 = windows.get("90d", {}).get("ic")
            decay = hr.get("decay_ratio")
            status = hr.get("primary_status", "UNKNOWN")

            ic_30_s = f"{ic_30:.4f}" if ic_30 is not None else "N/A"
            ic_60_s = f"{ic_60:.4f}" if ic_60 is not None else "N/A"
            ic_90_s = f"{ic_90:.4f}" if ic_90 is not None else "N/A"
            decay_s = f"{decay * 100:.0f}%" if decay is not None else "N/A"

            color = STATUS_COLORS.get(status, "")
            print(
                f"{r['model']:<22} {hr['horizon']:>3} {train_ic:>10.4f} "
                f"{ic_30_s:>9} {ic_60_s:>9} {ic_90_s:>9} "
                f"{decay_s:>8} {color}{status:>8}{RESET}"
            )

    print("=" * 88)

    # Summary
    statuses = [r.get("overall_status") for r in results if "overall_status" in r]
    n_green = statuses.count("GREEN")
    n_yellow = statuses.count("YELLOW")
    n_red = statuses.count("RED")
    print(
        f"\nSummary: {n_green} GREEN, {n_yellow} YELLOW, {n_red} RED "
        f"out of {len(statuses)} models"
    )
    if n_red > 0:
        red_models = [r["model"] for r in results if r.get("overall_status") == "RED"]
        print(f"  RETRAIN NEEDED: {', '.join(red_models)}")
    print()


def send_alerts(results: List[Dict[str, Any]]) -> None:
    """Send Telegram alerts for YELLOW/RED models."""
    try:
        from monitoring.notify import send_alert, AlertLevel
    except ImportError:
        logger.warning("monitoring.notify not available — skipping alerts")
        return

    for r in results:
        status = r.get("overall_status")
        if status not in ("YELLOW", "RED"):
            continue

        model = r.get("model", "unknown")
        symbol = r.get("symbol", "")

        details: Dict[str, str] = {"symbol": symbol, "model": model}
        for hr in r.get("horizons", []):
            h = hr.get("horizon", "?")
            pic = hr.get("primary_ic")
            tic = hr.get("training_ic", 0)
            details[f"h{h}_ic"] = f"{pic:.4f}" if pic is not None else "N/A"
            details[f"h{h}_train_ic"] = f"{tic:.4f}"
            dr = hr.get("decay_ratio")
            if dr is not None:
                details[f"h{h}_decay"] = f"{dr * 100:.0f}%"

        if status == "RED":
            send_alert(
                AlertLevel.CRITICAL,
                f"IC DECAY: {model} — retrain needed",
                details=details,
                source="ic_decay_monitor",
            )
        else:
            send_alert(
                AlertLevel.WARNING,
                f"IC DECAY: {model} — alpha weakening",
                details=details,
                source="ic_decay_monitor",
            )


def main() -> None:
    from monitoring.ic_decay_monitor import run_monitor, save_results, maybe_trigger_retrain

    parser = argparse.ArgumentParser(
        description="IC Decay Monitor — detect model alpha decay",
    )
    parser.add_argument(
        "--alert", action="store_true",
        help="Send Telegram alerts for YELLOW/RED models",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="JSON output only (no table)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    project_root = Path(__file__).resolve().parent.parent
    os.chdir(project_root)

    results = run_monitor()
    save_results(results)

    if args.json:
        print(json.dumps({"models": results}, indent=2, default=str))
    else:
        print_table(results)

    if args.alert:
        send_alerts(results)

    maybe_trigger_retrain(results)

    statuses = [r.get("overall_status") for r in results if "overall_status" in r]
    if "RED" in statuses:
        sys.exit(2)
    elif "YELLOW" in statuses:
        sys.exit(1)
