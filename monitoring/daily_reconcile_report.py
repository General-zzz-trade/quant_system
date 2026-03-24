# monitoring/daily_reconcile_report.py
"""Report formatting and CLI entry point for daily reconciliation.

Extracted from daily_reconcile.py to reduce file size.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from monitoring.daily_reconcile import (
    RUNTIME_DIR,
    OUTPUT_PREFIX,
    parse_log,
    reconcile_symbol,
)

log = logging.getLogger("daily_reconciliation")


def print_report(report: dict) -> None:
    """Print a human-readable reconciliation report to stdout."""
    print("=" * 72)
    print(f"  DAILY RECONCILIATION: {report['symbol']}")
    print("=" * 72)

    if report.get("status") != "ok":
        print(f"  Status: {report['status']}")
        if "error" in report:
            print(f"  Error: {report['error']}")
        print()
        return

    # Config
    cfg = report["config"]
    print(f"  Model: {report['model']}")
    print(f"  Config: dz={cfg['deadzone']} (effective: {cfg['effective_dz']}), "
          f"hold={cfg['min_hold']}-{cfg['max_hold']}")
    p = report["period"]
    print(f"  Period: {p['start']} -> {p['end']} ({p['n_bars']} bars, {p['n_days']} days)")
    print()

    # Key metrics
    m = report["metrics"]
    print("  KEY METRICS:")
    match_pct = m["signal_match_rate"] * 100
    status_icon = "PASS" if match_pct >= 95 else ("WARN" if match_pct >= 80 else "FAIL")
    print(f"    Signal match rate:     {match_pct:.1f}%  [{status_icon}]")

    slip = m["avg_fill_slippage_bps"]
    slip_status = "PASS" if abs(slip) < 5 else ("WARN" if abs(slip) < 20 else "FAIL")
    print(f"    Avg fill slippage:     {slip:+.2f} bps  [{slip_status}]")

    delay = m["timing_delay_bars"]
    delay_status = "PASS" if delay < 1 else ("WARN" if delay < 3 else "FAIL")
    print(f"    Timing delay:          {delay:.2f} bars  [{delay_status}]")

    if m["pnl_gap_pct"] is not None:
        gap = m["pnl_gap_pct"]
        gap_status = "PASS" if abs(gap) < 20 else ("WARN" if abs(gap) < 50 else "FAIL")
        print(f"    PnL gap:               {gap:+.1f}%  [{gap_status}]")
    else:
        print("    PnL gap:               N/A (insufficient data)")
    print()

    # Signal distribution
    sd = report["signal_distribution"]
    print("  SIGNAL DISTRIBUTION:")
    print(f"    Live:     short={sd['live'].get(-1, sd['live'].get('-1', 0)):>4}  "
          f"flat={sd['live'].get(0, sd['live'].get('0', 0)):>4}  "
          f"long={sd['live'].get(1, sd['live'].get('1', 0)):>4}")
    print(f"    Backtest: short={sd['backtest'].get(-1, sd['backtest'].get('-1', 0)):>4}  "
          f"flat={sd['backtest'].get(0, sd['backtest'].get('0', 0)):>4}  "
          f"long={sd['backtest'].get(1, sd['backtest'].get('1', 0)):>4}")
    sc = report["signal_changes"]
    print(f"    Signal changes: live={sc['live']}  backtest={sc['backtest']}")
    print()

    # Fills
    fi = report["fills"]
    print("  TRADES:")
    print(f"    Opens: {fi['open_count']}  Closes: {fi['close_count']}  "
          f"BT trades: {fi['bt_trade_count']}")
    print(f"    Live PnL:    ${fi['live_total_pnl_usd']:.4f} ({fi['live_total_pnl_pct']:.2f}%)")
    print(f"    Backtest PnL: {fi['bt_total_pnl_pct']:.2f}% (signal-on-close)")
    print()

    # Divergence causes
    dc = report["divergence_causes"]
    total_divs = sum(dc.values())
    if total_divs > 0:
        print(f"  DIVERGENCE CAUSES ({total_divs} total):")
        for cause, count in sorted(dc.items(), key=lambda x: -x[1]):
            if count > 0:
                print(f"    {cause:20s}  {count:>4}  ({count / total_divs * 100:.0f}%)")
        print()

    # Sample disagreements
    disags = report.get("disagreements_sample", [])
    if disags:
        n_show = min(10, len(disags))
        print(f"  DISAGREEMENTS (showing {n_show}/{len(disags)}):")
        for d in disags[:n_show]:
            print(f"    bar {d['bar_num']:>4} @ {d['timestamp']}: "
                  f"close=${d['close']:.2f} z={d['z']:+.3f} "
                  f"live={d['live_signal']:+d} bt={d['bt_signal']:+d} "
                  f"regime={d['regime']} dz={d['dz']:.3f}")
    else:
        print("  No signal disagreements found.")
    print()
    print("=" * 72)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Daily live vs backtest reconciliation report",
    )
    parser.add_argument(
        "--log-file", required=True, type=Path,
        help="Path to Bybit alpha log file (e.g. logs/bybit_alpha.log)",
    )
    parser.add_argument(
        "--days", type=int, default=7,
        help="Number of days to look back (default: 7)",
    )
    parser.add_argument(
        "--symbol", default=None,
        help="Filter to specific symbol (e.g. ETHUSDT)",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output as JSON instead of text",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=RUNTIME_DIR,
        help=f"Directory for JSON output (default: {RUNTIME_DIR})",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Validate log file
    if not args.log_file.exists():
        log.error("Log file not found: %s", args.log_file)
        print(f"Error: log file not found: {args.log_file}", file=sys.stderr)
        return 1

    # Compute lookback window
    since = datetime.now() - timedelta(days=args.days)
    log.info("Parsing %s (last %d days, since %s)", args.log_file, args.days, since)

    # Parse log
    sessions = parse_log(
        args.log_file,
        symbol_filter=args.symbol,
        since=since,
    )

    if not sessions:
        msg = "No trading data found in log file for the specified period."
        log.warning(msg)
        print(msg, file=sys.stderr)
        return 1

    # Run reconciliation per symbol
    reports = []
    for sym in sorted(sessions):
        session = sessions[sym]
        if not session.bars:
            log.info("Skipping %s: no bars", sym)
            continue
        log.info("Reconciling %s: %d bars, %d fills",
                 sym, len(session.bars), len(session.fills))
        report = reconcile_symbol(session)
        reports.append(report)

    if not reports:
        print("No symbols with bar data found.", file=sys.stderr)
        return 1

    # Summary report across all symbols
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "log_file": str(args.log_file),
        "lookback_days": args.days,
        "symbols": [r["symbol"] for r in reports],
        "reports": reports,
    }

    # Aggregate metrics
    ok_reports = [r for r in reports if r.get("status") == "ok"]
    if ok_reports:
        summary["aggregate"] = {
            "signal_match_rate": round(
                float(np.mean([r["metrics"]["signal_match_rate"] for r in ok_reports])), 4
            ),
            "avg_fill_slippage_bps": round(
                float(np.mean([r["metrics"]["avg_fill_slippage_bps"] for r in ok_reports])), 2
            ),
            "timing_delay_bars": round(
                float(np.mean([r["metrics"]["timing_delay_bars"] for r in ok_reports])), 2
            ),
            "total_bars": sum(r["period"]["n_bars"] for r in ok_reports),
            "total_disagreements": sum(
                len(r.get("disagreements_sample", [])) for r in ok_reports
            ),
        }

    # Output
    if args.json:
        print(json.dumps(summary, indent=2, default=str))
    else:
        # Header
        print()
        total_bars = sum(len(sessions[s].bars) for s in sessions)
        print(f"Reconciliation: {len(reports)} symbol(s), {total_bars} total bars, "
              f"last {args.days} days")
        print()
        for report in reports:
            print_report(report)

        # Aggregate summary
        if "aggregate" in summary:
            agg = summary["aggregate"]
            print()
            print("AGGREGATE SUMMARY:")
            print(f"  Signal match rate:  {agg['signal_match_rate'] * 100:.1f}%")
            print(f"  Avg slippage:       {agg['avg_fill_slippage_bps']:+.2f} bps")
            print(f"  Timing delay:       {agg['timing_delay_bars']:.2f} bars")
            print(f"  Total bars:         {agg['total_bars']}")
            print(f"  Disagreements:      {agg['total_disagreements']}")
            print()

    # Save JSON output
    args.output_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")
    output_path = args.output_dir / f"{OUTPUT_PREFIX}_{date_str}.json"
    try:
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        log.info("Report saved to %s", output_path)
        if not args.json:
            print(f"Report saved to {output_path}")
    except OSError as e:
        log.warning("Failed to save report: %s", e)

    return 0


if __name__ == "__main__":
    sys.exit(main())
