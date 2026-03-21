"""Weekly performance report generator.

Usage:
    python3 -m scripts.ops.weekly_report [--week YYYY_WW] [--out reports/weekly/]
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.ops.demo_tracker import TRACK_RECORD_PATH, load_track_record, compute_rolling_sharpe

# ---------------------------------------------------------------------------
# Walk-forward baselines (hardcoded from CLAUDE.md)
# ---------------------------------------------------------------------------

WF_BASELINES: dict[str, dict[str, float]] = {
    "BTCUSDT": {"sharpe": 4.37, "returns_pct": 142},
    "ETHUSDT": {"sharpe": 4.67, "returns_pct": 705},
    "ETHUSDT_15m": {"sharpe": 1.04, "returns_pct": 121},
}

DEFAULT_OUT_DIR = Path("reports/weekly")

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def get_week_data(record: dict[str, Any], year: int, week: int) -> dict[str, Any]:
    """Return daily entries whose ISO week matches *year* / *week*.

    Returns a dict: {date_str: day_dict} filtered to the requested week.
    """
    result: dict[str, Any] = {}
    for date_str, day in record.get("daily", {}).items():
        try:
            d = datetime.strptime(date_str, "%Y-%m-%d")
            iso = d.isocalendar()
            if iso.year == year and iso.week == week:
                result[date_str] = day
        except ValueError:
            continue
    return result


def _week_symbol_stats(week_data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Aggregate per-symbol stats across all days in *week_data*."""
    sym_stats: dict[str, dict[str, Any]] = {}
    for day in week_data.values():
        for sym, sym_day in day.get("symbols", {}).items():
            if sym not in sym_stats:
                sym_stats[sym] = {
                    "pnl_usd": 0.0,
                    "trades": 0,
                    "wins": 0,
                    "losses": 0,
                    "bars": 0,
                }
            s = sym_stats[sym]
            s["pnl_usd"] += sym_day.get("pnl_usd", 0.0)
            s["trades"] += sym_day.get("trades", 0)
            s["wins"] += sym_day.get("wins", 0)
            s["losses"] += sym_day.get("losses", 0)
            s["bars"] += sym_day.get("bars", 0)
    return sym_stats


def _week_pnl_series(week_data: dict[str, Any]) -> list[float]:
    """Daily P&L values sorted by date for the week."""
    items = sorted(week_data.items())
    return [day.get("total_pnl_usd", 0.0) for _, day in items]


def _week_regime_dist(week_data: dict[str, Any]) -> dict[str, int]:
    """Count regime labels appearing in bar signals (not tracked directly — placeholder)."""
    # Regime distribution is not stored per-day in current schema; return empty.
    return {}


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def format_symbol_table(week_data: dict[str, Any]) -> str:
    """Return a markdown table of per-symbol breakdown for the week."""
    sym_stats = _week_symbol_stats(week_data)
    if not sym_stats:
        return "_No symbol data for this week._\n"

    header = (
        "| Symbol | PnL (USD) | Trades | Win Rate | WF Sharpe | Deviation |\n"
        "|--------|-----------|--------|----------|-----------|-----------|\n"
    )
    rows: list[str] = []
    for sym in sorted(sym_stats):
        s = sym_stats[sym]
        total = s["wins"] + s["losses"]
        win_rate_str = f"{s['wins'] / total:.0%}" if total > 0 else "N/A"
        wf = WF_BASELINES.get(sym)
        wf_sharpe_str = f"{wf['sharpe']:.2f}" if wf else "N/A"
        # Deviation: not enough weekly data to compute a meaningful Sharpe diff; show placeholder
        deviation_str = "—"
        rows.append(
            f"| {sym} | ${s['pnl_usd']:+.2f} | {s['trades']} | {win_rate_str} "
            f"| {wf_sharpe_str} | {deviation_str} |"
        )
    return header + "\n".join(rows) + "\n"


def _format_wf_table(sym_stats: dict[str, dict[str, Any]], week_pnl: list[float]) -> str:
    """Walk-forward comparison table."""
    week_sharpe = compute_rolling_sharpe(week_pnl, window=7)
    week_sharpe_str = f"{week_sharpe:.2f}" if week_sharpe is not None else "N/A"

    header = (
        "| Symbol | WF Sharpe | Live 7d Sharpe | Status |\n"
        "|--------|-----------|----------------|--------|\n"
    )
    rows: list[str] = []

    # Portfolio-level row
    rows.append(f"| Portfolio | — | {week_sharpe_str} | — |")

    for sym in sorted(WF_BASELINES):
        wf = WF_BASELINES[sym]
        s = sym_stats.get(sym)
        if s is None:
            rows.append(f"| {sym} | {wf['sharpe']:.2f} | N/A | NO DATA |")
            continue
        # Per-symbol Sharpe from week data is not directly computable without per-symbol daily;
        # use portfolio-level as proxy and flag it
        status = "PASS" if (week_sharpe is not None and week_sharpe >= 0.5 * wf["sharpe"]) else "WATCH"
        rows.append(f"| {sym} | {wf['sharpe']:.2f} | {week_sharpe_str}* | {status} |")

    rows.append("\n_\\* Portfolio-level 7d Sharpe used as proxy for per-symbol comparison._")
    return header + "\n".join(rows) + "\n"


# ---------------------------------------------------------------------------
# Report generator
# ---------------------------------------------------------------------------


def generate_report(record: dict[str, Any], year: int, week: int) -> str:
    """Generate a full markdown weekly performance report."""
    week_data = get_week_data(record, year, week)
    sym_stats = _week_symbol_stats(week_data)
    week_pnl = _week_pnl_series(week_data)

    total_pnl = sum(d.get("total_pnl_usd", 0.0) for d in week_data.values())
    total_trades = sum(d.get("total_trades", 0) for d in week_data.values())
    total_wins = sum(s.get("wins", 0) for s in sym_stats.values())
    total_losses = sum(s.get("losses", 0) for s in sym_stats.values())
    total_wl = total_wins + total_losses
    win_rate = total_wins / total_wl if total_wl > 0 else 0.0
    sharpe_7d = compute_rolling_sharpe(week_pnl, window=7)
    sharpe_7d_str = f"{sharpe_7d:.2f}" if sharpe_7d is not None else "N/A"

    # Max drawdown across week
    cum = 0.0
    peak = 0.0
    max_dd = 0.0
    for pnl in week_pnl:
        cum += pnl
        if cum > peak:
            peak = cum
        dd = cum - peak
        if dd < max_dd:
            max_dd = dd

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines: list[str] = [
        f"# Weekly Performance Report: Week {week:02d}, {year}",
        "",
        f"_Generated: {generated_at}_",
        "",
        "## Portfolio Summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Total PnL | ${total_pnl:+.2f} |",
        f"| Trades | {total_trades} |",
        f"| Win Rate | {win_rate:.0%} |",
        f"| 7d Sharpe | {sharpe_7d_str} |",
        f"| Max Drawdown | ${max_dd:.2f} |",
        f"| Days Tracked | {len(week_data)} |",
        "",
        "## Per-Symbol Breakdown",
        "",
        format_symbol_table(week_data),
        "## Regime Distribution",
        "",
        "_Regime distribution is not stored in current daily schema. "
        "See live logs for regime labels per bar._",
        "",
        "## Walk-Forward Comparison",
        "",
        _format_wf_table(sym_stats, week_pnl),
    ]

    if not week_data:
        lines.append("\n> **Warning**: No data found for this week.")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate weekly performance report from data/live/track_record.json"
    )
    now = datetime.now(timezone.utc)
    iso = now.isocalendar()
    default_week = f"{iso.year}_{iso.week:02d}"

    parser.add_argument(
        "--week",
        default=default_week,
        help="Week to report in YYYY_WW format (default: current week)",
    )
    parser.add_argument(
        "--record",
        default=str(TRACK_RECORD_PATH),
        help=f"Path to track_record.json (default: {TRACK_RECORD_PATH})",
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUT_DIR),
        help=f"Output directory (default: {DEFAULT_OUT_DIR})",
    )
    args = parser.parse_args()

    # Parse week
    try:
        year_str, week_str = args.week.split("_")
        year = int(year_str)
        week = int(week_str)
    except (ValueError, AttributeError):
        parser.error(f"Invalid --week format '{args.week}', expected YYYY_WW")

    record = load_track_record(Path(args.record))
    report_md = generate_report(record, year, week)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"week_{year}_{week:02d}.md"
    out_file.write_text(report_md, encoding="utf-8")
    print(f"weekly_report: wrote {out_file}")


if __name__ == "__main__":
    main()
