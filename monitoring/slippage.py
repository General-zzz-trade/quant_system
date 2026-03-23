"""Slippage Analyzer — compare actual fills vs backtest assumptions.

Parses bybit_alpha.log for fill events and computes:
- Actual fill price vs bar close
- Slippage in bps
- Maker order fill rate (PostOnly 45s)
- Fee comparison vs backtest assumptions

Usage:
    python3 -m scripts.ops.slippage_analyzer --hours 168
    python3 -m scripts.ops.slippage_analyzer --hours 24 --json
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict

import numpy as np

logger = logging.getLogger(__name__)

LOG_FILE = Path("logs/bybit_alpha.log")

# Regex patterns for log parsing
_FILL_PATTERN = re.compile(
    r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*'
    r'FILL.*symbol=(\w+).*side=(\w+).*price=([\d.]+).*qty=([\d.]+)'
)
_ORDER_PATTERN = re.compile(
    r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*'
    r'(?:ORDER_PLACED|LIMIT_ORDER).*symbol=(\w+).*side=(\w+).*price=([\d.]+)'
)
_CLOSE_PATTERN = re.compile(
    r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*'
    r'BAR.*symbol=(\w+).*close=([\d.]+)'
)
_MAKER_FILL = re.compile(
    r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*LIMIT_FILLED.*symbol=(\w+)'
)
_MARKET_FALLBACK = re.compile(
    r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*MARKET_FALLBACK.*symbol=(\w+)'
)


def parse_log(log_path: Path, hours: int) -> Dict[str, Any]:
    """Parse log file for fill events within the given time window."""
    if not log_path.exists():
        return {"error": f"Log file not found: {log_path}"}

    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    fills = []
    maker_fills = 0
    market_fallbacks = 0
    last_closes: Dict[str, float] = {}

    with open(log_path) as f:
        for line in f:
            # Parse timestamp
            ts_match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
            if not ts_match:
                continue
            try:
                ts = datetime.strptime(ts_match.group(1), "%Y-%m-%d %H:%M:%S")
                ts = ts.replace(tzinfo=timezone.utc)
            except ValueError:
                continue

            if ts < cutoff:
                continue

            # Track bar closes
            close_m = _CLOSE_PATTERN.match(line)
            if close_m:
                sym = close_m.group(2)
                last_closes[sym] = float(close_m.group(3))

            # Track fills
            fill_m = _FILL_PATTERN.match(line)
            if fill_m:
                sym = fill_m.group(2)
                side = fill_m.group(3)
                price = float(fill_m.group(4))
                qty = float(fill_m.group(5))
                bar_close = last_closes.get(sym, price)
                slippage_bps = (price - bar_close) / bar_close * 10000
                if side.lower() in ("sell", "short"):
                    slippage_bps = -slippage_bps  # sell wants higher price

                fills.append({
                    "timestamp": ts.isoformat(),
                    "symbol": sym,
                    "side": side,
                    "fill_price": price,
                    "bar_close": bar_close,
                    "slippage_bps": slippage_bps,
                    "qty": qty,
                })

            # Track maker/market
            if _MAKER_FILL.match(line):
                maker_fills += 1
            if _MARKET_FALLBACK.match(line):
                market_fallbacks += 1

    return {
        "fills": fills,
        "maker_fills": maker_fills,
        "market_fallbacks": market_fallbacks,
    }


def compute_stats(parsed: Dict[str, Any]) -> Dict[str, Any]:
    """Compute slippage statistics from parsed fills."""
    fills = parsed.get("fills", [])
    maker_fills = parsed.get("maker_fills", 0)
    market_fallbacks = parsed.get("market_fallbacks", 0)

    if not fills:
        return {
            "n_fills": 0,
            "avg_slippage_bps": 0,
            "median_slippage_bps": 0,
            "p95_slippage_bps": 0,
            "maker_fill_rate": 0,
            "maker_fills": maker_fills,
            "market_fallbacks": market_fallbacks,
            "per_symbol": {},
        }

    slippages = [f["slippage_bps"] for f in fills]
    arr = np.array(slippages)

    # Per-symbol breakdown
    per_symbol: Dict[str, Dict] = {}
    for f in fills:
        sym = f["symbol"]
        if sym not in per_symbol:
            per_symbol[sym] = {"fills": 0, "slippages": []}
        per_symbol[sym]["fills"] += 1
        per_symbol[sym]["slippages"].append(f["slippage_bps"])

    per_symbol_stats = {}
    for sym, data in per_symbol.items():
        s = np.array(data["slippages"])
        per_symbol_stats[sym] = {
            "n_fills": data["fills"],
            "avg_slippage_bps": float(np.mean(s)),
            "median_slippage_bps": float(np.median(s)),
        }

    total_orders = maker_fills + market_fallbacks
    maker_rate = maker_fills / total_orders * 100 if total_orders > 0 else 0

    # Backtest assumption comparison
    backtest_cost_bps = 4.0  # round-trip assumption
    actual_avg = float(np.mean(np.abs(arr)))

    return {
        "n_fills": len(fills),
        "avg_slippage_bps": float(np.mean(arr)),
        "median_slippage_bps": float(np.median(arr)),
        "p95_slippage_bps": float(np.percentile(np.abs(arr), 95)),
        "std_slippage_bps": float(np.std(arr)),
        "maker_fill_rate": maker_rate,
        "maker_fills": maker_fills,
        "market_fallbacks": market_fallbacks,
        "per_symbol": per_symbol_stats,
        "backtest_cost_assumption_bps": backtest_cost_bps,
        "actual_avg_abs_slippage_bps": actual_avg,
        "cost_vs_assumption": "BETTER" if actual_avg < backtest_cost_bps else "WORSE",
    }


def print_report(stats: Dict[str, Any], hours: int) -> None:
    """Print slippage analysis report."""
    print(f"\n{'='*60}")
    print(f"  SLIPPAGE ANALYSIS ({hours}h window)")
    print(f"{'='*60}")
    print(f"  Total fills:       {stats['n_fills']}")
    print(f"  Avg slippage:      {stats['avg_slippage_bps']:+.2f} bps")
    print(f"  Median slippage:   {stats['median_slippage_bps']:+.2f} bps")
    print(f"  P95 |slippage|:    {stats['p95_slippage_bps']:.2f} bps")
    print(f"  Maker fill rate:   {stats['maker_fill_rate']:.1f}%")
    print(f"    Maker fills:     {stats['maker_fills']}")
    print(f"    Market fallback: {stats['market_fallbacks']}")
    print("\n  --- vs Backtest ---")
    print(f"  Backtest assumes:  {stats['backtest_cost_assumption_bps']:.1f} bps RT")
    print(f"  Actual avg |slip|: {stats['actual_avg_abs_slippage_bps']:.2f} bps")
    print(f"  Verdict:           {stats['cost_vs_assumption']}")

    if stats['per_symbol']:
        print("\n  --- Per Symbol ---")
        print(f"  {'Symbol':<18} {'Fills':>6} {'Avg bps':>10} {'Med bps':>10}")
        for sym, data in sorted(stats['per_symbol'].items()):
            print(f"  {sym:<18} {data['n_fills']:>6} {data['avg_slippage_bps']:>+10.2f} "
                  f"{data['median_slippage_bps']:>+10.2f}")
    print(f"{'='*60}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Slippage Analyzer")
    parser.add_argument("--hours", type=int, default=168, help="Analysis window in hours")
    parser.add_argument("--log-file", type=str, default=str(LOG_FILE))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    log_path = Path(args.log_file)
    parsed = parse_log(log_path, args.hours)

    if "error" in parsed:
        print(f"  ERROR: {parsed['error']}")
        return

    stats = compute_stats(parsed)

    if args.json:
        print(json.dumps(stats, indent=2, default=str))
    else:
        print_report(stats, args.hours)


if __name__ == "__main__":
    main()
