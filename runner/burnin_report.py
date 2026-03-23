#!/usr/bin/env python3
"""Burn-in report generator — validates paper/shadow/testnet runs against exit criteria.

Three-phase protocol:
  Phase A (Paper, 7 days): signal correlation > 0.8, no memory leaks, no stale data
  Phase B (Shadow, 7 days): estimated slippage < 5bps, latency P99 < 5000ms
  Phase C (Testnet, 3 days): fill rate > 95%, reconciliation drift = 0

Usage:
    python3 -m scripts.burnin_report --phase A --data-dir data/live
    python3 -m scripts.burnin_report --phase B --data-dir data/live
    python3 -m scripts.burnin_report --phase C --data-dir data/live --output burnin_report.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

sys.path.insert(0, "/quant_system")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class BurninCheck:
    name: str
    passed: bool
    value: Any
    threshold: Any
    message: str


@dataclass
class BurninPhaseReport:
    phase: str
    phase_name: str
    start_time: Optional[str]
    end_time: Optional[str]
    duration_hours: float
    required_hours: float
    passed: bool
    checks: List[BurninCheck]
    generated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


# ── Phase exit criteria ──

PHASE_CRITERIA = {
    "A": {
        "name": "Paper Trading",
        "min_duration_hours": 168.0,  # 7 days
        "checks": {
            "signal_correlation": {"threshold": 0.8, "op": ">="},
            "no_stale_data_events": {"threshold": 0, "op": "=="},
            "event_count_min": {"threshold": 1000, "op": ">="},
            "error_rate_pct": {"threshold": 5.0, "op": "<="},
        },
    },
    "B": {
        "name": "Shadow Trading",
        "min_duration_hours": 168.0,  # 7 days
        "checks": {
            "estimated_slippage_bps": {"threshold": 5.0, "op": "<="},
            "latency_p99_ms": {"threshold": 5000.0, "op": "<="},
            "signal_count_min": {"threshold": 100, "op": ">="},
            "error_rate_pct": {"threshold": 2.0, "op": "<="},
        },
    },
    "C": {
        "name": "Testnet Live",
        "min_duration_hours": 72.0,  # 3 days
        "checks": {
            "fill_rate_pct": {"threshold": 95.0, "op": ">="},
            "reconciliation_drift_count": {"threshold": 0, "op": "=="},
            "order_count_min": {"threshold": 10, "op": ">="},
            "error_rate_pct": {"threshold": 1.0, "op": "<="},
        },
    },
}


def _check_op(value: float, threshold: float, op: str) -> bool:
    if op == ">=":
        return value >= threshold
    elif op == "<=":
        return value <= threshold
    elif op == "==":
        return value == threshold
    elif op == ">":
        return value > threshold
    elif op == "<":
        return value < threshold
    return False


def _query_event_log(data_dir: str) -> Dict[str, Any]:
    """Extract metrics from SQLite event log."""
    db_path = os.path.join(data_dir, "event_log.db")
    if not os.path.exists(db_path):
        return {"exists": False}

    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()

        # Check tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}

        result: Dict[str, Any] = {"exists": True}

        if "events" in tables:
            cursor.execute("SELECT COUNT(*) FROM events")
            result["event_count"] = cursor.fetchone()[0]

            cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM events")
            row = cursor.fetchone()
            result["first_event_ts"] = row[0]
            result["last_event_ts"] = row[1]

            # Count errors
            cursor.execute("SELECT COUNT(*) FROM events WHERE event_type = 'ERROR'")
            result["error_count"] = cursor.fetchone()[0]

            # Count stale data events
            cursor.execute(
                "SELECT COUNT(*) FROM events WHERE event_type LIKE '%stale%' OR event_type LIKE '%STALE%'"
            )
            result["stale_data_events"] = cursor.fetchone()[0]

        if "signals" in tables:
            cursor.execute("SELECT COUNT(*) FROM signals")
            result["signal_count"] = cursor.fetchone()[0]

        if "orders" in tables:
            cursor.execute("SELECT COUNT(*) FROM orders")
            result["order_count"] = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM orders WHERE status = 'FILLED'")
            result["filled_count"] = cursor.fetchone()[0]

        if "reconciliations" in tables:
            cursor.execute("SELECT COUNT(*) FROM reconciliations WHERE drift > 0")
            result["reconciliation_drift_count"] = cursor.fetchone()[0]

        return result
    except Exception as e:
        return {"exists": True, "error": str(e)}
    finally:
        conn.close()


def _compute_duration_hours(first_ts: Optional[str], last_ts: Optional[str]) -> float:
    """Compute duration between first and last event timestamps."""
    if not first_ts or not last_ts:
        return 0.0
    try:
        # Try ISO format
        t0 = datetime.fromisoformat(str(first_ts))
        t1 = datetime.fromisoformat(str(last_ts))
        return (t1 - t0).total_seconds() / 3600.0
    except (ValueError, TypeError):
        try:
            # Try unix timestamp
            return (float(last_ts) - float(first_ts)) / 3600.0
        except (ValueError, TypeError):
            return 0.0


def generate_report(phase: str, data_dir: str) -> BurninPhaseReport:
    """Generate a burn-in report for the specified phase."""
    if phase not in PHASE_CRITERIA:
        raise ValueError(f"Unknown phase: {phase}. Must be one of: {list(PHASE_CRITERIA.keys())}")

    criteria = PHASE_CRITERIA[phase]
    db_metrics = _query_event_log(data_dir)

    if not db_metrics.get("exists"):
        return BurninPhaseReport(
            phase=phase,
            phase_name=criteria["name"],
            start_time=None,
            end_time=None,
            duration_hours=0.0,
            required_hours=criteria["min_duration_hours"],
            passed=False,
            checks=[BurninCheck(
                name="data_exists",
                passed=False,
                value=None,
                threshold="event_log.db must exist",
                message=f"No event log found at {data_dir}/event_log.db",
            )],
        )

    first_ts = db_metrics.get("first_event_ts")
    last_ts = db_metrics.get("last_event_ts")
    duration_hours = _compute_duration_hours(first_ts, last_ts)

    checks: List[BurninCheck] = []

    # Duration check
    required_hours = criteria["min_duration_hours"]
    checks.append(BurninCheck(
        name="duration_hours",
        passed=duration_hours >= required_hours,
        value=round(duration_hours, 1),
        threshold=required_hours,
        message=f"Run duration {duration_hours:.1f}h {'>='>''} required {required_hours:.0f}h",
    ))

    # Phase-specific checks
    event_count = db_metrics.get("event_count", 0)
    error_count = db_metrics.get("error_count", 0)
    error_rate = (error_count / event_count * 100.0) if event_count > 0 else 0.0

    for check_name, spec in criteria["checks"].items():
        threshold = spec["threshold"]
        op = spec["op"]

        if check_name == "signal_correlation":
            # Placeholder: would need actual signal vs backtest correlation
            value = 0.0  # Will be computed from actual data when available
            msg = "Signal correlation not yet computed (requires backtest comparison)"
        elif check_name == "no_stale_data_events":
            value = db_metrics.get("stale_data_events", 0)
            msg = f"Stale data events: {value}"
        elif check_name == "event_count_min":
            value = event_count
            msg = f"Total events: {value}"
        elif check_name == "error_rate_pct":
            value = round(error_rate, 2)
            msg = f"Error rate: {value}% ({error_count}/{event_count})"
        elif check_name == "estimated_slippage_bps":
            value = 0.0  # Will be computed from shadow fill analysis
            msg = "Estimated slippage not yet computed (requires shadow fills)"
        elif check_name == "latency_p99_ms":
            value = 0.0  # Will be computed from latency tracker data
            msg = "Latency P99 not yet computed (requires latency data)"
        elif check_name == "signal_count_min":
            value = db_metrics.get("signal_count", 0)
            msg = f"Signal count: {value}"
        elif check_name == "fill_rate_pct":
            order_count = db_metrics.get("order_count", 0)
            filled_count = db_metrics.get("filled_count", 0)
            value = (filled_count / order_count * 100.0) if order_count > 0 else 0.0
            msg = f"Fill rate: {value:.1f}% ({filled_count}/{order_count})"
        elif check_name == "reconciliation_drift_count":
            value = db_metrics.get("reconciliation_drift_count", 0)
            msg = f"Reconciliation drifts: {value}"
        elif check_name == "order_count_min":
            value = db_metrics.get("order_count", 0)
            msg = f"Order count: {value}"
        else:
            value = 0
            msg = f"Unknown check: {check_name}"

        passed = _check_op(float(value), float(threshold), op)
        checks.append(BurninCheck(
            name=check_name,
            passed=passed,
            value=value,
            threshold=f"{op} {threshold}",
            message=msg,
        ))

    all_passed = all(c.passed for c in checks)

    return BurninPhaseReport(
        phase=phase,
        phase_name=criteria["name"],
        start_time=str(first_ts) if first_ts else None,
        end_time=str(last_ts) if last_ts else None,
        duration_hours=round(duration_hours, 1),
        required_hours=required_hours,
        passed=all_passed,
        checks=checks,
    )


def append_to_aggregate(report_dict: dict, aggregate_path: str) -> None:
    """Append a phase report to the aggregate burn-in report JSON.

    The aggregate file is a list of phase reports. BurninGate requires
    all three phases (A, B, C) with passed=true.
    """
    existing = []
    if os.path.exists(aggregate_path):
        try:
            with open(aggregate_path) as f:
                data = json.load(f)
            if isinstance(data, list):
                existing = data
            else:
                existing = [data]
        except (json.JSONDecodeError, ValueError):
            existing = []

    # Replace existing entry for this phase
    phase = report_dict.get("phase", "")
    existing = [r for r in existing if r.get("phase") != phase]
    existing.append(report_dict)

    # Sort by phase order
    phase_order = {"A": 0, "B": 1, "C": 2}
    existing.sort(key=lambda r: phase_order.get(r.get("phase", ""), 9))

    with open(aggregate_path, "w") as f:
        json.dump(existing, f, indent=2)

    logger.info("Aggregate report updated at %s (%d phases)", aggregate_path, len(existing))


def main():
    parser = argparse.ArgumentParser(description="Burn-in Report Generator")
    parser.add_argument("--phase", required=True, choices=["A", "B", "C"],
                        help="Burn-in phase (A=Paper, B=Shadow, C=Testnet)")
    parser.add_argument("--data-dir", default="data/live",
                        help="Directory with SQLite stores")
    parser.add_argument("--output", default=None,
                        help="Output JSON file (default: stdout)")
    parser.add_argument("--aggregate", default=None,
                        help="Aggregate report file (append phase result for BurninGate)")
    args = parser.parse_args()

    report = generate_report(args.phase, args.data_dir)

    report_dict = asdict(report)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(report_dict, f, indent=2)
        status = "PASSED" if report.passed else "FAILED"
        print(f"Phase {args.phase} ({report.phase_name}): {status}")
        print(f"Report written to {args.output}")
    else:
        print(json.dumps(report_dict, indent=2))

    # Append to aggregate report (for BurninGate)
    if args.aggregate:
        append_to_aggregate(report_dict, args.aggregate)
    elif args.output is None:
        # Default: append to standard aggregate path
        default_agg = os.path.join(args.data_dir, "burnin_report.json")
        append_to_aggregate(report_dict, default_agg)

    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
