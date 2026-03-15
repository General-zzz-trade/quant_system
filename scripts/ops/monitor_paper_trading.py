#!/usr/bin/env python3
"""Monitor paper trading — parse logs, show signal/fill stats, detect anomalies.

Usage:
    python3 -m scripts.monitor_paper_trading
    python3 -m scripts.monitor_paper_trading --watch   # refresh every 60s
"""
from __future__ import annotations

import argparse
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

CONFIGS = {
    "BTC": {
        "pid_file": "logs/paper_trading_v2.pid",
        "log_file": "logs/paper_trading_v2.log",
        "symbol": "BTCUSDT",
    },
    "SOL": {
        "pid_file": "logs/sol_paper_trading.pid",
        "log_file": "logs/sol_paper_trading.log",
        "symbol": "SOLUSDT",
    },
    "ETH": {
        "pid_file": "logs/eth_paper_trading.pid",
        "log_file": "logs/eth_paper_trading.log",
        "symbol": "ETHUSDT",
    },
}

# Regex patterns for log parsing
RE_INFERENCE = re.compile(
    r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ .* Inference .*/(\w+): "
    r"side=(\w+) strength=([\d.]+) score=([\d.e+-]+) latency=([\d.]+)ms"
)
RE_STATUS = re.compile(
    r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ .* STATUS .* event_index=(\d+) +fills=(\d+)"
)
RE_POLLER = re.compile(
    r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ .* (FundingPoller|OIPoller|FGIPoller|BtcKlinePoller|DeribitIV)"
)
RE_ERROR = re.compile(
    r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ (ERROR|CRITICAL|WARNING) "
)


def _is_running(pid_file: str) -> tuple[bool, int]:
    p = Path(pid_file)
    if not p.exists():
        return False, 0
    pid = int(p.read_text().strip())
    try:
        os.kill(pid, 0)
        return True, pid
    except (OSError, ProcessLookupError):
        return False, pid


def _parse_log(log_file: str) -> dict:
    p = Path(log_file)
    if not p.exists():
        return {"error": "log file not found"}

    inferences = []
    last_status = None
    errors = []
    last_poller_ts = {}
    last_line_ts = None

    # Read last 2000 lines for efficiency
    lines = p.read_text().splitlines()
    recent = lines[-2000:] if len(lines) > 2000 else lines

    for line in recent:
        m = RE_INFERENCE.match(line)
        if m:
            ts_str, symbol, side, strength, score, latency = m.groups()
            inferences.append({
                "ts": ts_str,
                "side": side,
                "strength": float(strength),
                "score": float(score),
                "latency": float(latency),
            })
            last_line_ts = ts_str
            continue

        m = RE_STATUS.match(line)
        if m:
            ts_str, event_idx, fills = m.groups()
            last_status = {
                "ts": ts_str,
                "events": int(event_idx),
                "fills": int(fills),
            }
            last_line_ts = ts_str
            continue

        m = RE_POLLER.match(line)
        if m:
            ts_str, poller_name = m.groups()
            last_poller_ts[poller_name] = ts_str
            last_line_ts = ts_str
            continue

        m = RE_ERROR.match(line)
        if m:
            ts_str, level = m.groups()
            errors.append({"ts": ts_str, "level": level, "line": line[:120]})

    # Compute signal stats from inferences
    n_inf = len(inferences)
    side_counts = {"long": 0, "short": 0, "flat": 0}
    total_strength = 0.0
    total_latency = 0.0
    for inf in inferences:
        side_counts[inf["side"]] = side_counts.get(inf["side"], 0) + 1
        total_strength += inf["strength"]
        total_latency += inf["latency"]

    return {
        "n_inferences": n_inf,
        "side_counts": side_counts,
        "avg_strength": total_strength / n_inf if n_inf > 0 else 0,
        "avg_latency_ms": total_latency / n_inf if n_inf > 0 else 0,
        "last_inference": inferences[-1] if inferences else None,
        "last_status": last_status,
        "pollers": last_poller_ts,
        "errors_recent": errors[-5:],
        "last_log_ts": last_line_ts,
        "total_lines": len(lines),
    }


def _format_report(name: str, cfg: dict, stats: dict) -> str:
    running, pid = _is_running(cfg["pid_file"])
    status = f"\033[92mRUNNING\033[0m (PID {pid})" if running else f"\033[91mDEAD\033[0m (PID {pid})"

    lines = [
        f"\n{'='*60}",
        f"  {name} ({cfg['symbol']})  —  {status}",
        f"{'='*60}",
    ]

    if "error" in stats:
        lines.append(f"  {stats['error']}")
        return "\n".join(lines)

    lines.append(f"  Log: {stats['total_lines']} lines, last: {stats['last_log_ts'] or 'N/A'}")

    # Status
    st = stats.get("last_status")
    if st:
        lines.append(f"  Events: {st['events']}, Fills: {st['fills']}, at {st['ts']}")

    # Inference summary
    n = stats["n_inferences"]
    sc = stats["side_counts"]
    lines.append(f"\n  Signals ({n} inferences in window):")
    lines.append(f"    Long: {sc.get('long',0)}, Short: {sc.get('short',0)}, Flat: {sc.get('flat',0)}")
    lines.append(f"    Avg strength: {stats['avg_strength']:.4f}")
    lines.append(f"    Avg latency: {stats['avg_latency_ms']:.1f} ms")

    last = stats.get("last_inference")
    if last:
        lines.append(f"    Latest: side={last['side']}, str={last['strength']:.4f}, score={last['score']:.4f} @ {last['ts']}")

    # Pollers
    if stats["pollers"]:
        lines.append("\n  Pollers (last seen):")
        for name_p, ts in sorted(stats["pollers"].items()):
            lines.append(f"    {name_p}: {ts}")

    # Errors
    errs = stats.get("errors_recent", [])
    if errs:
        lines.append(f"\n  Recent warnings/errors ({len(errs)}):")
        for e in errs[-3:]:
            lines.append(f"    [{e['level']}] {e['line'][:100]}")

    return "\n".join(lines)


def monitor_once():
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"\n  Paper Trading Monitor — {now}")

    for name, cfg in CONFIGS.items():
        stats = _parse_log(cfg["log_file"])
        print(_format_report(name, cfg, stats))

    print()


def main():
    parser = argparse.ArgumentParser(description="Monitor paper trading")
    parser.add_argument("--watch", action="store_true", help="Refresh every 60s")
    args = parser.parse_args()

    if args.watch:
        try:
            while True:
                os.system("clear")
                monitor_once()
                time.sleep(60)
        except KeyboardInterrupt:
            print("\nStopped.")
    else:
        monitor_once()


if __name__ == "__main__":
    main()
