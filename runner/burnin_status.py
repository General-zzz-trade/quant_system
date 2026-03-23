#!/usr/bin/env python3
"""Quick burn-in status checker. Shows current phase and readiness."""
import json
from pathlib import Path

def main():
    report_path = Path("data/live/burnin_report.json")

    if not report_path.exists():
        print("Burn-in status: NOT STARTED")
        print("  Phase A (Paper, 7d): PENDING")
        print("  Phase B (Shadow, 7d): PENDING")
        print("  Phase C (Testnet, 3d): PENDING")
        print("\n  To start: shadow_mode=true in production.local.yaml, then run 7 days")
        return

    with open(report_path) as f:
        report = json.load(f)

    phases = {"A": "Paper (7d)", "B": "Shadow (7d)", "C": "Testnet (3d)"}

    print("Burn-in Status:")
    for phase_key, label in phases.items():
        phase = report.get(f"phase_{phase_key}", {})
        status = phase.get("status", "pending")
        days = phase.get("days_completed", 0)
        required = {"A": 7, "B": 7, "C": 3}[phase_key]

        if status == "passed":
            icon = "[PASS]"
        elif status == "running":
            icon = f"[{days}/{required}d]"
        else:
            icon = "[    ]"

        print(f"  {icon} Phase {phase_key}: {label} — {status}")

    # Overall readiness
    all_passed = all(
        report.get(f"phase_{k}", {}).get("status") == "passed"
        for k in "ABC"
    )
    print(f"\n  Live trading ready: {'YES' if all_passed else 'NO'}")

if __name__ == "__main__":
    main()
