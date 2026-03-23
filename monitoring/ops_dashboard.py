#!/usr/bin/env python3
"""Unified ops dashboard -- one command to see all system status.

Aggregates: service health, model versions, burn-in status, recent signals,
and data freshness into a single view.

Usage:
    python3 -m scripts.ops.ops_dashboard
    python3 -m scripts.ops.ops_dashboard --json
"""
from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List


ACTIVE_HOST_SERVICES: tuple[str, ...] = (
    "bybit-alpha.service",
    "bybit-mm.service",
)


def check_service_health(service_name: str) -> str:
    """Check if a specific systemd service is running."""
    try:
        result = subprocess.run(
            ["systemctl", "is-active", service_name],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip()
    except FileNotFoundError:
        return "systemctl_not_found"
    except Exception:
        return "unknown"


def check_active_host_services() -> Dict[str, str]:
    """Check current systemd state for both active host trading services."""
    return {
        service_name: check_service_health(service_name)
        for service_name in ACTIVE_HOST_SERVICES
    }


def check_system_health() -> str:
    """Backward-compatible alpha-only service health alias."""
    return check_service_health("bybit-alpha.service")


def check_burnin_status() -> Dict[str, Any]:
    """Check burn-in phase status from report file."""
    report_path = Path("data/live/burnin_report.json")
    if not report_path.exists():
        return {"status": "not_started", "phases": {}}
    try:
        data = json.loads(report_path.read_text())
        phases: Dict[str, Any] = {}
        if isinstance(data, list):
            for entry in data:
                phase = entry.get("phase", "?")
                phases[phase] = {
                    "passed": entry.get("passed", False),
                    "duration_hours": entry.get("duration_hours", 0),
                }
        all_passed = all(
            phases.get(p, {}).get("passed", False) for p in ("A", "B", "C")
        )
        return {
            "status": "passed" if all_passed else "in_progress",
            "phases": phases,
        }
    except (json.JSONDecodeError, OSError):
        return {"status": "error", "phases": {}}


def check_model_versions(
    models_path: str = "models_v8",
) -> List[Dict[str, Any]]:
    """Check current model versions and training dates."""
    models_dir = Path(models_path)
    if not models_dir.exists():
        return []
    results = []
    for model_dir in sorted(models_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        info: Dict[str, Any] = {"name": model_dir.name}
        config_path = model_dir / "config.json"
        if config_path.exists():
            try:
                cfg = json.loads(config_path.read_text())
                info["trained"] = cfg.get("trained_at", "unknown")
                info["horizon"] = cfg.get("horizon", "?")
                info["n_features"] = len(cfg.get("features", []))
            except (json.JSONDecodeError, OSError):
                info["trained"] = "error"
        else:
            info["trained"] = "no config"
        results.append(info)
    return results


def check_data_freshness() -> List[Dict[str, Any]]:
    """Check freshness of data files."""
    data_dir = Path("data_files")
    if not data_dir.exists():
        return []
    results = []
    now = datetime.now()
    for csv_file in sorted(data_dir.glob("*_1h.csv")):
        mtime = datetime.fromtimestamp(csv_file.stat().st_mtime)
        age_hours = (now - mtime).total_seconds() / 3600.0
        results.append({
            "file": csv_file.name,
            "modified": mtime.strftime("%Y-%m-%d %H:%M"),
            "age_hours": round(age_hours, 1),
        })
    return results


def check_recent_signals(
    log_path: str = "logs/bybit_alpha.log", hours: int = 24
) -> Dict[str, Any]:
    """Count recent signals from log."""
    path = Path(log_path)
    if not path.exists():
        return {"error": "log not found", "total_bars": 0}

    cutoff = datetime.now() - timedelta(hours=hours)
    ts_re = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
    sig_re = re.compile(r"sig=([+\-]?\d+)")

    total = 0
    nonzero = 0
    for line in open(path, encoding="utf-8", errors="replace"):
        m = ts_re.match(line)
        if not m:
            continue
        try:
            ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue
        if ts < cutoff:
            continue
        sig_m = sig_re.search(line)
        if sig_m:
            total += 1
            if int(sig_m.group(1)) != 0:
                nonzero += 1

    return {
        "hours": hours,
        "total_bars": total,
        "nonzero_signals": nonzero,
        "signal_rate": round(nonzero / total, 3) if total > 0 else 0.0,
    }


def check_disk_usage() -> Dict[str, Any]:
    """Check disk usage for key directories."""
    results: Dict[str, Any] = {}
    for name, path in [
        ("logs", "logs"),
        ("data_files", "data_files"),
        ("models", "models_v8"),
    ]:
        p = Path(path)
        if not p.exists():
            results[name] = "N/A"
            continue
        total_bytes = sum(
            f.stat().st_size for f in p.rglob("*") if f.is_file()
        )
        results[name] = f"{total_bytes / (1024 * 1024):.1f} MB"
    return results


def format_dashboard(data: Dict[str, Any]) -> str:
    """Format all dashboard sections for display."""
    lines = [
        "=" * 60,
        "  QUANT SYSTEM OPS DASHBOARD",
        f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60,
        "",
    ]

    # Service status
    service_statuses = data.get("service_statuses")
    if service_statuses:
        lines.append("  Active host services:")
        for service_name, state in service_statuses.items():
            lines.append(f"    {service_name:<20} {state}")
    else:
        svc = data.get("service_status", "unknown")
        lines.append(f"  Service: bybit-alpha.service = {svc}")
    lines.append("")

    # Burn-in
    burnin = data.get("burnin", {})
    lines.append(f"  Burn-in: {burnin.get('status', 'unknown')}")
    for phase_key in ("A", "B", "C"):
        phase = burnin.get("phases", {}).get(phase_key, {})
        passed = phase.get("passed", False)
        dur = phase.get("duration_hours", 0)
        icon = "PASS" if passed else "----"
        lines.append(f"    Phase {phase_key}: [{icon}] {dur:.0f}h")
    lines.append("")

    # Models
    lines.append("  Models:")
    for model in data.get("models", []):
        name = model.get("name", "?")
        trained = model.get("trained", "?")
        h = model.get("horizon", "?")
        nf = model.get("n_features", "?")
        lines.append(f"    {name:<22} h={h}  feats={nf}  trained={trained}")
    if not data.get("models"):
        lines.append("    (none found)")
    lines.append("")

    # Recent signals
    sig = data.get("signals", {})
    lines.append(
        f"  Signals (last {sig.get('hours', 24)}h): "
        f"{sig.get('nonzero_signals', 0)}/{sig.get('total_bars', 0)} bars "
        f"({sig.get('signal_rate', 0):.1%} active)"
    )
    lines.append("")

    # Data freshness
    lines.append("  Data freshness:")
    for df in data.get("data_freshness", []):
        age = df.get("age_hours", 0)
        stale = " [STALE]" if age > 48 else ""
        lines.append(
            f"    {df['file']:<25} {df['modified']}  "
            f"({age:.0f}h ago){stale}"
        )
    if not data.get("data_freshness"):
        lines.append("    (no data files)")
    lines.append("")

    # Disk usage
    disk = data.get("disk_usage", {})
    lines.append("  Disk usage:")
    for name, size in disk.items():
        lines.append(f"    {name:<15} {size}")
    lines.append("")

    return "\n".join(lines)


def collect_all(log_path: str = "logs/bybit_alpha.log") -> Dict[str, Any]:
    """Collect all dashboard data."""
    return {
        "timestamp": datetime.now().isoformat(),
        "service_status": check_system_health(),
        "service_statuses": check_active_host_services(),
        "burnin": check_burnin_status(),
        "models": check_model_versions(),
        "signals": check_recent_signals(log_path),
        "data_freshness": check_data_freshness(),
        "disk_usage": check_disk_usage(),
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Unified ops dashboard"
    )
    parser.add_argument(
        "--log-file",
        default="logs/bybit_alpha.log",
        help="Path to alpha log file",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )
    args = parser.parse_args()

    data = collect_all(log_path=args.log_file)

    if args.json:
        print(json.dumps(data, indent=2, default=str))
    else:
        print(format_dashboard(data))


if __name__ == "__main__":
    main()
