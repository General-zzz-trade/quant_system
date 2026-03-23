#!/usr/bin/env python3
"""Compare live trading signals/trades vs backtest on the same period.

Parses the Bybit alpha log to extract bar signals, trade events, and PnL,
then replays the same price data through the backtest engine and compares.

Usage:
    python3 -m scripts.ops.compare_live_backtest --log-file logs/bybit_alpha.log
    python3 -m scripts.ops.compare_live_backtest --log-file logs/bybit_alpha.log --symbol ETHUSDT
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from execution.sim.live_comparison import BacktestLiveComparison

logger = logging.getLogger(__name__)

# ── Log line patterns ────────────────────────────────────────────────

# Bar lines (both polling and WS):
#   "ETHUSDT bar 201: $2100.8 z=-0.223 sig=0 hold=1"
#   "WS ETHUSDT bar 201: $2106.9 z=-0.076 sig=0 hold=1 regime=active dz=0.200"
BAR_RE = re.compile(
    r"(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ INFO __main__: "
    r"(?:WS )?(?P<symbol>\w+) bar (?P<bar>\d+): "
    r"\$(?P<close>[\d.]+) "
    r"z=(?P<z>[+-]?[\d.]+) "
    r"sig=(?P<sig>[+-]?\d+) "
    r"hold=(?P<hold>\d+)"
    r"(?: regime=(?P<regime>\w+))?"
    r"(?: dz=(?P<dz>[\d.]+))?"
    r"(?: TRADE=(?P<trade>\{.*\}))?"
)

# Close/P&L lines:
#   "ETHUSDT CLOSE short: pnl=$-0.1234 (−0.59%) total=$1.23 wins=3/5"
CLOSE_RE = re.compile(
    r"(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ INFO __main__: "
    r"(?P<symbol>\w+) CLOSE (?P<side>long|short): "
    r"pnl=\$(?P<pnl>[+-]?[\d.]+) "
    r"\((?P<pct>[+-]?[\d.]+)%\) "
    r"total=\$(?P<total>[+-]?[\d.]+) "
    r"wins=(?P<wins>\d+)/(?P<trades>\d+)"
)

# Stop close lines:
#   "ETHUSDT STOP CLOSED: pnl=$-0.0123 total=$0.45 trades=2/4"
STOP_RE = re.compile(
    r"(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ INFO __main__: "
    r"(?P<symbol>\w+) STOP CLOSED: "
    r"pnl=\$(?P<pnl>[+-]?[\d.]+) "
    r"total=\$(?P<total>[+-]?[\d.]+) "
    r"trades=(?P<wins>\d+)/(?P<trades>\d+)"
)

# Warmup config lines:
#   "ETHUSDT: model vv11, 14 features, dz=0.3, hold=12-60, size=0.0100"
CONFIG_RE = re.compile(
    r"(?P<symbol>\w+): model (?P<model>\w+), (?P<nfeat>\d+) features, "
    r"dz=(?P<dz>[\d.]+), hold=(?P<min_hold>\d+)-(?P<max_hold>\d+), "
    r"size=(?P<size>[\d.]+)"
)

# Sizing lines:
#   "ETHUSDT SIZING: equity=$101 lev=10x → 0.48 ETH ($1011 notional)"
SIZING_RE = re.compile(
    r"(?P<symbol>\w+) SIZING: equity=\$(?P<equity>[\d.]+) "
    r"lev=(?P<lev>[\d.]+)x"
)

# Open trade lines:
#   "Opened sell 0.4800 @ ~$2107.0: {..."
OPEN_RE = re.compile(
    r"Opened (?P<side>buy|sell) (?P<qty>[\d.]+) @ ~\$(?P<price>[\d.]+): "
    r"(?P<result>\{.*\})"
)


@dataclass
class LiveBar:
    """A single bar observation from the live log."""
    timestamp: datetime
    symbol: str
    bar_num: int
    close: float
    z: float
    signal: int
    hold: int
    regime: str = "active"
    dz: float = 0.0
    trade: dict | None = None


@dataclass
class LiveTrade:
    """A trade event extracted from the live log."""
    timestamp: datetime
    symbol: str
    side: str          # "long" or "short"
    pnl: float = 0.0
    pnl_pct: float = 0.0
    exit_type: str = "signal"  # "signal" or "stop"
    entry_price: float = 0.0
    exit_price: float = 0.0


@dataclass
class LiveSession:
    """Data from a single live trading session (one startup)."""
    symbol: str
    model: str = ""
    deadzone: float = 0.0
    min_hold: int = 0
    max_hold: int = 0
    bars: list[LiveBar] = field(default_factory=list)
    trades: list[LiveTrade] = field(default_factory=list)


def parse_log(log_path: Path, symbol_filter: str | None = None) -> dict[str, list[LiveSession]]:
    """Parse Bybit alpha log file into per-symbol sessions.

    Returns dict mapping symbol -> list of LiveSession.
    """
    sessions: dict[str, list[LiveSession]] = {}
    current_configs: dict[str, dict] = {}
    active_sessions: dict[str, LiveSession] = {}

    with open(log_path) as f:
        for line in f:
            line = line.rstrip()

            # Config line → start new session
            m = CONFIG_RE.search(line)
            if m:
                sym = m.group("symbol")
                if symbol_filter and sym != symbol_filter:
                    continue
                cfg = {
                    "model": m.group("model"),
                    "dz": float(m.group("dz")),
                    "min_hold": int(m.group("min_hold")),
                    "max_hold": int(m.group("max_hold")),
                    "size": float(m.group("size")),
                }
                # If config changed or new session, save old and start new
                if sym in active_sessions and current_configs.get(sym) != cfg:
                    sessions.setdefault(sym, []).append(active_sessions[sym])
                if sym not in active_sessions or current_configs.get(sym) != cfg:
                    active_sessions[sym] = LiveSession(
                        symbol=sym, model=cfg["model"],
                        deadzone=cfg["dz"], min_hold=cfg["min_hold"],
                        max_hold=cfg["max_hold"],
                    )
                    current_configs[sym] = cfg
                continue

            # Bar line
            m = BAR_RE.match(line)
            if m:
                sym = m.group("symbol")
                if symbol_filter and sym != symbol_filter:
                    continue
                ts = datetime.strptime(m.group("ts"), "%Y-%m-%d %H:%M:%S")
                trade_data = None
                if m.group("trade"):
                    try:
                        trade_data = _parse_trade_dict(m.group("trade"))
                    except Exception:
                        trade_data = {"raw": m.group("trade")}
                bar = LiveBar(
                    timestamp=ts, symbol=sym,
                    bar_num=int(m.group("bar")),
                    close=float(m.group("close")),
                    z=float(m.group("z")),
                    signal=int(m.group("sig")),
                    hold=int(m.group("hold")),
                    regime=m.group("regime") or "active",
                    dz=float(m.group("dz")) if m.group("dz") else 0.0,
                    trade=trade_data,
                )
                if sym in active_sessions:
                    active_sessions[sym].bars.append(bar)
                continue

            # Close trade line
            m = CLOSE_RE.match(line)
            if m:
                sym = m.group("symbol")
                if symbol_filter and sym != symbol_filter:
                    continue
                ts = datetime.strptime(m.group("ts"), "%Y-%m-%d %H:%M:%S")
                trade = LiveTrade(
                    timestamp=ts, symbol=sym,
                    side=m.group("side"),
                    pnl=float(m.group("pnl")),
                    pnl_pct=float(m.group("pct")),
                    exit_type="signal",
                )
                if sym in active_sessions:
                    active_sessions[sym].trades.append(trade)
                continue

            # Stop close line
            m = STOP_RE.match(line)
            if m:
                sym = m.group("symbol")
                if symbol_filter and sym != symbol_filter:
                    continue
                ts = datetime.strptime(m.group("ts"), "%Y-%m-%d %H:%M:%S")
                trade = LiveTrade(
                    timestamp=ts, symbol=sym,
                    side="unknown",
                    pnl=float(m.group("pnl")),
                    exit_type="stop",
                )
                if sym in active_sessions:
                    active_sessions[sym].trades.append(trade)

    # Finalize remaining sessions
    for sym, session in active_sessions.items():
        sessions.setdefault(sym, []).append(session)

    return sessions


def _parse_trade_dict(raw: str) -> dict:
    """Parse TRADE={...} from log line. Handles Python-style dict strings."""
    # Replace single quotes with double quotes for JSON parsing
    s = raw.replace("'", '"')
    # Handle True/False/None
    s = s.replace("True", "true").replace("False", "false").replace("None", "null")
    return json.loads(s)


def run_backtest_signals(bars: list[LiveBar], deadzone: float,
                         min_hold: int, max_hold: int,
                         symbol: str) -> list[dict]:
    """Replay bars through the same signal logic used by AlphaRunner.

    This uses the z-scores and regime info already logged (since we cannot
    reproduce the exact feature engine state from logs alone). The comparison
    is therefore focused on the signal generation logic given the same z-scores.
    """
    results = []
    current_signal = 0
    hold_count = 0

    for bar in bars:
        z = bar.z
        regime_ok = bar.regime == "active"
        prev_signal = current_signal

        # Determine desired signal
        if not regime_ok:
            desired = 0
        elif z > deadzone:
            desired = 1
        elif z < -deadzone:
            desired = -1
        else:
            desired = 0

        # Min-hold / max-hold logic (matches AlphaRunner.process_bar)
        if hold_count < min_hold and prev_signal != 0:
            new_signal = prev_signal
            hold_count += 1
        elif desired != prev_signal:
            new_signal = desired
            hold_count = 1
        else:
            new_signal = desired
            hold_count += 1

        if hold_count >= max_hold and new_signal != 0:
            new_signal = 0
            hold_count = 1

        current_signal = new_signal

        results.append({
            "bar_num": bar.bar_num,
            "timestamp": bar.timestamp,
            "close": bar.close,
            "z": z,
            "signal": new_signal,
            "hold": hold_count,
            "regime": bar.regime,
        })

    return results


def compare_session(session: LiveSession) -> dict:
    """Compare a single live session against backtest replay.

    Returns a comparison report dict.
    """
    bars = session.bars
    if not bars:
        return {"symbol": session.symbol, "error": "no bars"}

    # Get the effective deadzone: use dynamic dz from bars if available,
    # otherwise fall back to session config
    dz_values = [b.dz for b in bars if b.dz > 0]
    avg_dz = float(np.mean(dz_values)) if dz_values else session.deadzone

    # Run backtest replay with the z-scores from the log
    bt_results = run_backtest_signals(
        bars, deadzone=avg_dz,
        min_hold=session.min_hold, max_hold=session.max_hold,
        symbol=session.symbol,
    )

    # Compare signal agreement
    n_bars = len(bars)
    agreements = 0
    disagreements = []
    live_signals = []
    bt_signals = []

    for i, (bar, bt) in enumerate(zip(bars, bt_results)):
        live_sig = bar.signal
        bt_sig = bt["signal"]
        live_signals.append(live_sig)
        bt_signals.append(bt_sig)

        if live_sig == bt_sig:
            agreements += 1
        else:
            disagreements.append({
                "bar_num": bar.bar_num,
                "timestamp": str(bar.timestamp),
                "close": bar.close,
                "z": bar.z,
                "live_signal": live_sig,
                "bt_signal": bt_sig,
                "live_hold": bar.hold,
                "bt_hold": bt["hold"],
                "regime": bar.regime,
                "dz": bar.dz,
            })

    agreement_rate = agreements / n_bars if n_bars > 0 else 0.0

    # Signal distribution
    live_counts = {-1: 0, 0: 0, 1: 0}
    bt_counts = {-1: 0, 0: 0, 1: 0}
    for s in live_signals:
        live_counts[s] = live_counts.get(s, 0) + 1
    for s in bt_signals:
        bt_counts[s] = bt_counts.get(s, 0) + 1

    # Signal changes (entries/exits)
    live_changes = sum(1 for i in range(1, len(live_signals))
                       if live_signals[i] != live_signals[i - 1])
    bt_changes = sum(1 for i in range(1, len(bt_signals))
                     if bt_signals[i] != bt_signals[i - 1])

    # Compute backtest PnL from signal changes on bar closes
    bt_trades = _extract_trades_from_signals(bt_results)

    report = {
        "symbol": session.symbol,
        "model": session.model,
        "config": {
            "deadzone": session.deadzone,
            "avg_dynamic_dz": round(avg_dz, 4),
            "min_hold": session.min_hold,
            "max_hold": session.max_hold,
        },
        "period": {
            "start": str(bars[0].timestamp),
            "end": str(bars[-1].timestamp),
            "n_bars": n_bars,
        },
        "signal_agreement": {
            "rate": round(agreement_rate, 4),
            "agreed": agreements,
            "disagreed": len(disagreements),
        },
        "signal_distribution": {
            "live": live_counts,
            "backtest": bt_counts,
        },
        "signal_changes": {
            "live": live_changes,
            "backtest": bt_changes,
        },
        "live_trades": len(session.trades),
        "backtest_trades": len(bt_trades),
        "live_total_pnl": round(sum(t.pnl for t in session.trades), 4),
        "backtest_total_pnl": round(sum(t["pnl"] for t in bt_trades), 4),
        "disagreements": disagreements[:20],  # first 20 only
    }

    # Use BacktestLiveComparison for matched-trade analysis if we have trades
    if session.trades and bt_trades:
        comp = BacktestLiveComparison()
        for t in bt_trades:
            comp.add_backtest_trade(
                bar=t["entry_bar"], side=t["side"],
                entry=t["entry_price"], exit_price=t["exit_price"],
                pnl=t["pnl"],
            )
        for i, t in enumerate(session.trades):
            side = 1 if t.side == "long" else -1
            comp.add_live_trade(
                bar=i, side=side,
                entry=t.entry_price, exit_price=t.exit_price,
                pnl=t.pnl,
            )
        cr = comp.compute_report()
        report["matched_analysis"] = {
            "n_matched": cr.n_matched,
            "pnl_gap": round(cr.pnl_gap, 4),
            "pnl_gap_pct": round(cr.pnl_gap_pct, 2),
            "reality_discount": round(cr.reality_discount, 4),
            "backtest_win_rate": round(cr.backtest_win_rate, 4),
            "live_win_rate": round(cr.live_win_rate, 4),
        }

    return report


def _extract_trades_from_signals(bt_results: list[dict]) -> list[dict]:
    """Extract trades from backtest signal sequence.

    A trade opens when signal changes from 0 to +1/-1 (or flips),
    and closes when signal changes to 0 or flips.
    """
    trades = []
    entry_bar = None
    entry_price = 0.0
    entry_side = 0

    for r in bt_results:
        sig = r["signal"]
        if entry_side == 0 and sig != 0:
            # Open trade
            entry_bar = r["bar_num"]
            entry_price = r["close"]
            entry_side = sig
        elif entry_side != 0 and sig != entry_side:
            # Close trade (flat or flip)
            exit_price = r["close"]
            if entry_side == 1:
                pnl_pct = (exit_price - entry_price) / entry_price * 100
            else:
                pnl_pct = (entry_price - exit_price) / entry_price * 100
            trades.append({
                "entry_bar": entry_bar,
                "exit_bar": r["bar_num"],
                "side": entry_side,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl_pct": round(pnl_pct, 4),
                "pnl": round(pnl_pct / 100 * entry_price * 0.01, 6),  # approx
            })
            # If flipping, open new trade
            if sig != 0:
                entry_bar = r["bar_num"]
                entry_price = r["close"]
                entry_side = sig
            else:
                entry_bar = None
                entry_price = 0.0
                entry_side = 0

    return trades


def print_report(report: dict) -> None:
    """Print a human-readable comparison report."""
    print("=" * 70)
    print(f"  LIVE vs BACKTEST COMPARISON: {report['symbol']}")
    print("=" * 70)

    if "error" in report:
        print(f"  Error: {report['error']}")
        return

    print(f"  Model: {report['model']}")
    cfg = report["config"]
    print(f"  Config: dz={cfg['deadzone']} (avg dynamic: {cfg['avg_dynamic_dz']}), "
          f"hold={cfg['min_hold']}-{cfg['max_hold']}")
    p = report["period"]
    print(f"  Period: {p['start']} -> {p['end']} ({p['n_bars']} bars)")
    print()

    # Signal agreement
    sa = report["signal_agreement"]
    print(f"  Signal Agreement: {sa['rate']:.1%} ({sa['agreed']}/{sa['agreed'] + sa['disagreed']})")

    sd = report["signal_distribution"]
    print("  Signal Distribution:")
    live_d, bt_d = sd["live"], sd["backtest"]
    print(f"    Live:     short={live_d.get(-1, 0):>4}  "
          f"flat={live_d.get(0, 0):>4}  long={live_d.get(1, 0):>4}")
    print(f"    Backtest: short={bt_d.get(-1, 0):>4}  "
          f"flat={bt_d.get(0, 0):>4}  long={bt_d.get(1, 0):>4}")

    sc = report["signal_changes"]
    print(f"  Signal Changes: live={sc['live']}  backtest={sc['backtest']}")
    print()

    # Trades
    print(f"  Trades: live={report['live_trades']}  backtest={report['backtest_trades']}")
    print(f"  Total PnL: live=${report['live_total_pnl']:.4f}  backtest=${report['backtest_total_pnl']:.4f}")

    if "matched_analysis" in report:
        ma = report["matched_analysis"]
        print()
        print("  Matched Trade Analysis:")
        print(f"    Matched trades: {ma['n_matched']}")
        print(f"    PnL gap: ${ma['pnl_gap']:.4f} ({ma['pnl_gap_pct']:.1f}%)")
        print(f"    Reality discount: {ma['reality_discount']:.2f}x")
        print(f"    Win rates: live={ma['live_win_rate']:.1%}  backtest={ma['backtest_win_rate']:.1%}")

    # Disagreements
    disags = report.get("disagreements", [])
    if disags:
        print()
        print(f"  Signal Disagreements (first {len(disags)}):")
        for d in disags:
            print(f"    bar {d['bar_num']} @ {d['timestamp']}: "
                  f"close=${d['close']:.2f} z={d['z']:+.3f} "
                  f"live_sig={d['live_signal']} bt_sig={d['bt_signal']} "
                  f"hold={d['live_hold']}/{d['bt_hold']} "
                  f"regime={d['regime']} dz={d['dz']:.3f}")
    else:
        print()
        print("  No signal disagreements found.")

    print()
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Compare live trading signals vs backtest replay",
    )
    parser.add_argument("--log-file", required=True, type=Path,
                        help="Path to Bybit alpha log file")
    parser.add_argument("--symbol", default=None,
                        help="Filter to specific symbol (e.g. ETHUSDT)")
    parser.add_argument("--session", type=int, default=-1,
                        help="Session index to analyze (-1 = last, 0 = first)")
    parser.add_argument("--all-sessions", action="store_true",
                        help="Analyze all sessions")
    parser.add_argument("--json", action="store_true",
                        help="Output as JSON instead of text")
    args = parser.parse_args()

    if not args.log_file.exists():
        print(f"Error: log file not found: {args.log_file}", file=sys.stderr)
        sys.exit(1)

    logging.basicConfig(level=logging.WARNING)

    # Parse log
    sessions = parse_log(args.log_file, symbol_filter=args.symbol)

    if not sessions:
        print("No sessions found in log file.", file=sys.stderr)
        sys.exit(1)

    # Collect reports
    reports = []
    for symbol, sym_sessions in sorted(sessions.items()):
        # Filter sessions with actual bars
        valid = [s for s in sym_sessions if s.bars]
        if not valid:
            continue

        if args.all_sessions:
            for s in valid:
                reports.append(compare_session(s))
        else:
            idx = args.session if args.session >= 0 else len(valid) + args.session
            idx = max(0, min(idx, len(valid) - 1))
            reports.append(compare_session(valid[idx]))

    if args.json:
        # Convert datetimes for JSON serialization
        print(json.dumps(reports, indent=2, default=str))
    else:
        # Summary header
        total_bars = sum(r.get("period", {}).get("n_bars", 0) for r in reports)
        total_symbols = len(reports)
        print()
        print(f"Parsed {total_bars} bars across {total_symbols} symbol(s)")
        print()
        for report in reports:
            print_report(report)


if __name__ == "__main__":
    main()
