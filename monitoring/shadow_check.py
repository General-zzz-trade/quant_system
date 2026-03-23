#!/usr/bin/env python3
"""Shadow mode health check -- analyzes recent log data for shadow trading status.

Parses bybit_alpha.log to extract signal, regime, and trade statistics
over a configurable time window. Useful during burn-in Phase B (Shadow, 7d).

Usage:
    python3 -m scripts.ops.shadow_mode_check --log-file logs/bybit_alpha.log --hours 24
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class ShadowReport:
    """Aggregated shadow mode statistics."""

    window_hours: int = 0
    log_lines_parsed: int = 0
    log_lines_in_window: int = 0
    # Signals
    total_bars: int = 0
    signals_long: int = 0
    signals_short: int = 0
    signals_flat: int = 0
    # Trades (attempted)
    trade_attempts: int = 0
    trade_errors: int = 0
    combo_signals: int = 0
    # Regime distribution
    regimes: Dict[str, int] = field(default_factory=dict)
    # Per-symbol bar counts
    symbol_bars: Dict[str, int] = field(default_factory=dict)
    # Heartbeats
    heartbeats: int = 0
    # Errors
    error_lines: int = 0
    # Time range
    first_ts: Optional[str] = None
    last_ts: Optional[str] = None

    @property
    def signal_rate(self) -> float:
        """Fraction of bars with a non-zero signal."""
        if self.total_bars == 0:
            return 0.0
        return (self.signals_long + self.signals_short) / self.total_bars

    @property
    def trade_success_rate(self) -> float:
        """Fraction of trade attempts that did not error."""
        if self.trade_attempts == 0:
            return 0.0
        return (self.trade_attempts - self.trade_errors) / self.trade_attempts


# Regex patterns for log parsing
_TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
_BAR_RE = re.compile(
    r"(?:WS )?"
    r"(\w+) bar \d+: "
    r"\$[\d.]+ "
    r"z=[+\-]?[\d.]+ "
    r"sig=([+\-]?\d+)"
)
_REGIME_RE = re.compile(r"regime=(\w+)")
_TRADE_RE = re.compile(r"TRADE=\{")
_TRADE_ERR_RE = re.compile(r"'status': 'error'")
_COMBO_RE = re.compile(r"COMBO=\{")
_HEARTBEAT_RE = re.compile(r"HEARTBEAT")
_ERROR_RE = re.compile(r"\b(ERROR|CRITICAL)\b")


def _parse_ts(line: str) -> Optional[datetime]:
    """Extract timestamp from log line."""
    m = _TS_RE.match(line)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def parse_log_entries(log_path: str, hours: int = 24) -> ShadowReport:
    """Parse recent log entries and extract signal/regime/trade stats."""
    report = ShadowReport(window_hours=hours)
    path = Path(log_path)
    if not path.exists():
        return report

    cutoff = datetime.now() - timedelta(hours=hours)
    regimes: Counter[str] = Counter()
    symbol_bars: Counter[str] = Counter()

    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            report.log_lines_parsed += 1
            ts = _parse_ts(line)
            if ts is None:
                continue
            if ts < cutoff:
                continue

            report.log_lines_in_window += 1
            ts_str = ts.strftime("%Y-%m-%d %H:%M:%S")
            if report.first_ts is None:
                report.first_ts = ts_str
            report.last_ts = ts_str

            # Bar lines (signal info)
            bar_m = _BAR_RE.search(line)
            if bar_m:
                symbol = bar_m.group(1)
                sig = int(bar_m.group(2))
                report.total_bars += 1
                symbol_bars[symbol] += 1
                if sig > 0:
                    report.signals_long += 1
                elif sig < 0:
                    report.signals_short += 1
                else:
                    report.signals_flat += 1

                regime_m = _REGIME_RE.search(line)
                if regime_m:
                    regimes[regime_m.group(1)] += 1

            # Trade attempts
            if _TRADE_RE.search(line):
                report.trade_attempts += 1
                if _TRADE_ERR_RE.search(line):
                    report.trade_errors += 1

            # Combo signals
            if _COMBO_RE.search(line):
                report.combo_signals += 1

            # Heartbeats
            if _HEARTBEAT_RE.search(line):
                report.heartbeats += 1

            # Errors
            if _ERROR_RE.search(line):
                report.error_lines += 1

    report.regimes = dict(regimes)
    report.symbol_bars = dict(symbol_bars)
    return report


def format_report(report: ShadowReport) -> str:
    """Format the shadow report for display."""
    lines = [
        "=" * 60,
        "  SHADOW MODE HEALTH CHECK",
        "=" * 60,
        "",
        f"  Window:     last {report.window_hours}h",
        f"  Log lines:  {report.log_lines_in_window} in window"
        f" (of {report.log_lines_parsed} total)",
        f"  Time range: {report.first_ts or 'N/A'}"
        f" -> {report.last_ts or 'N/A'}",
        "",
        "  --- Signals ---",
        f"  Total bars:   {report.total_bars}",
        f"  Long (+1):    {report.signals_long}",
        f"  Short (-1):   {report.signals_short}",
        f"  Flat (0):     {report.signals_flat}",
        f"  Signal rate:  {report.signal_rate:.1%}",
        "",
        "  --- Trades ---",
        f"  Attempts:     {report.trade_attempts}",
        f"  Errors:       {report.trade_errors}",
        f"  Success rate: {report.trade_success_rate:.1%}"
        if report.trade_attempts > 0
        else "  Success rate: N/A",
        f"  COMBO signals: {report.combo_signals}",
        "",
        "  --- Regimes ---",
    ]
    if report.regimes:
        for regime, count in sorted(
            report.regimes.items(), key=lambda x: -x[1]
        ):
            lines.append(f"  {regime:<15} {count:>5} bars")
    else:
        lines.append("  (no regime data)")

    lines.append("")
    lines.append("  --- Per-Symbol Bars ---")
    if report.symbol_bars:
        for sym, count in sorted(
            report.symbol_bars.items(), key=lambda x: -x[1]
        ):
            lines.append(f"  {sym:<15} {count:>5} bars")
    else:
        lines.append("  (no bar data)")

    lines.append("")
    lines.append(f"  Heartbeats:   {report.heartbeats}")
    lines.append(f"  Error lines:  {report.error_lines}")
    lines.append("")

    # Health verdict
    issues: List[str] = []
    if report.total_bars == 0:
        issues.append("No bars processed in window")
    if report.heartbeats == 0:
        issues.append("No heartbeats — service may be down")
    if report.trade_attempts > 0 and report.trade_success_rate < 0.5:
        issues.append(
            f"Low trade success rate: {report.trade_success_rate:.0%}"
        )
    if report.error_lines > 10:
        issues.append(f"High error count: {report.error_lines}")

    if issues:
        lines.append("  ISSUES:")
        for issue in issues:
            lines.append(f"    [!] {issue}")
    else:
        lines.append("  STATUS: OK")

    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Shadow mode health check"
    )
    parser.add_argument(
        "--log-file",
        default="logs/bybit_alpha.log",
        help="Path to log file",
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=24,
        help="Hours of history to analyze (default: 24)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON instead of formatted text",
    )
    args = parser.parse_args()

    report = parse_log_entries(args.log_file, args.hours)

    if args.json:
        import json
        import dataclasses

        print(json.dumps(dataclasses.asdict(report), indent=2))
    else:
        print(format_report(report))

    # Exit 1 if no data at all
    if report.total_bars == 0 and report.heartbeats == 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
