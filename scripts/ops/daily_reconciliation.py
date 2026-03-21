#!/usr/bin/env python3
"""Daily live vs backtest reconciliation report.

Parses live trading logs to extract bar-level signals, fills, and PnL,
then replays the same predictions through the constraint pipeline to
detect signal divergences, execution slippage, and timing gaps.

Metrics computed:
- signal_match_rate: fraction of bars where live signal == backtest signal
- avg_fill_slippage_bps: mean (fill_price - bar_close) in basis points
- timing_delay_bars: how many bars late live trades are vs backtest signals
- pnl_gap_pct: (live_pnl - backtest_pnl) / abs(backtest_pnl) * 100

Usage:
    python3 -m scripts.ops.daily_reconciliation --log-file logs/bybit_alpha.log --days 7
    python3 -m scripts.ops.daily_reconciliation --log-file logs/bybit_alpha.log --symbol ETHUSDT
    python3 -m scripts.ops.daily_reconciliation --log-file logs/bybit_alpha.log --json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


log = logging.getLogger("daily_reconciliation")

# ── Output paths ─────────────────────────────────────────────────────

RUNTIME_DIR = Path("data/runtime")
OUTPUT_PREFIX = "reconciliation"

# ── Log line patterns ────────────────────────────────────────────────
# Bar log format (from run_bybit_alpha.py):
#   "2026-03-20 14:00:01,123 INFO __main__: WS ETHUSDT bar 201: $2100.8 z=+0.223 sig=1 hold=3 regime=active dz=0.300"
#   "... ETHUSDT bar 201: $2100.8 z=-0.223 sig=0 hold=1 regime=active dz=0.200 TRADE={...}"
BAR_RE = re.compile(
    r"(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ INFO \S+: "
    r"(?:WS )?(?P<symbol>\w+) bar (?P<bar>\d+): "
    r"\$(?P<close>[\d.]+) "
    r"z=(?P<z>[+-]?[\d.]+) "
    r"sig=(?P<sig>[+-]?\d+) "
    r"hold=(?P<hold>\d+)"
    r"(?: regime=(?P<regime>\w+))?"
    r"(?: dz=(?P<dz>[\d.]+))?"
    r"(?: zs=(?P<zs>[\d.]+))?"
    r"(?: TRADE=(?P<trade>\{.*\}))?"
)

# Alternate bar format from process_bar result dict:
#   "ETHUSDT bar=123 pred=0.001234 z=1.2345 signal=1 prev_signal=0 close=2500.0 regime=active"
BAR_ALT_RE = re.compile(
    r"(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ .*"
    r"(?P<symbol>\w+) bar=(?P<bar>\d+) "
    r"pred=(?P<pred>[+-]?[\d.eE-]+) "
    r"z=(?P<z>[+-]?[\d.eE-]+) "
    r"signal=(?P<sig>[+-]?\d+) "
    r"prev_signal=(?P<prev_sig>[+-]?\d+) "
    r"close=(?P<close>[\d.]+)"
    r"(?: regime=(?P<regime>\w+))?"
)

# Open trade:
#   "2026-03-20 14:00:02,456 INFO __main__: ETHUSDT OPEN LONG size=0.1 price=2500.50"
#   "Opened buy 0.48 @ ~$2107.0: {...}"
OPEN_RE = re.compile(
    r"(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ .*"
    r"(?P<symbol>\w+) OPEN (?P<side>LONG|SHORT) size=(?P<size>[\d.]+) price=(?P<price>[\d.]+)"
)
OPEN_ALT_RE = re.compile(
    r"(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ .*"
    r"Opened (?P<side>buy|sell) (?P<qty>[\d.]+) @ ~\$(?P<price>[\d.]+)"
)

# Close trade:
#   "ETHUSDT CLOSE long: pnl=$0.9500 (0.38%) total=$1.23 wins=3/5"
CLOSE_RE = re.compile(
    r"(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ .*"
    r"(?P<symbol>\w+) CLOSE (?P<side>long|short): "
    r"pnl=\$(?P<pnl>[+-]?[\d.]+) "
    r"\((?P<pct>[+-]?[\d.]+)%\) "
    r"total=\$(?P<total>[+-]?[\d.]+) "
    r"wins=(?P<wins>\d+)/(?P<trades>\d+)"
)

# Stop close:
#   "ETHUSDT STOP CLOSED: pnl=$-0.0123 total=$0.45 trades=2/4"
STOP_RE = re.compile(
    r"(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ .*"
    r"(?P<symbol>\w+) STOP CLOSED: "
    r"pnl=\$(?P<pnl>[+-]?[\d.]+) "
    r"total=\$(?P<total>[+-]?[\d.]+) "
    r"trades=(?P<wins>\d+)/(?P<trades>\d+)"
)

# Config line: "ETHUSDT: model vv11, 14 features, dz=0.3, hold=12-60, size=0.0100"
CONFIG_RE = re.compile(
    r"(?P<symbol>\w+): model (?P<model>\w+), (?P<nfeat>\d+) features, "
    r"dz=(?P<dz>[\d.]+), hold=(?P<min_hold>\d+)-(?P<max_hold>\d+), "
    r"size=(?P<size>[\d.]+)"
)


# ── Data classes ─────────────────────────────────────────────────────

@dataclass
class BarEntry:
    """A single bar observation parsed from the live log."""
    timestamp: datetime
    symbol: str
    bar_num: int
    close: float
    z: float
    signal: int
    hold: int
    regime: str = "active"
    dz: float = 0.0
    pred: Optional[float] = None
    prev_signal: Optional[int] = None
    trade: Optional[dict] = None


@dataclass
class FillEntry:
    """A fill event (open or close) parsed from the live log."""
    timestamp: datetime
    symbol: str
    side: str       # "long" / "short" / "buy" / "sell"
    price: float
    size: float = 0.0
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    fill_type: str = "open"  # "open" or "close"
    exit_type: str = "signal"  # "signal" or "stop"


@dataclass
class SymbolSession:
    """Accumulated data for one symbol across the analysis window."""
    symbol: str
    model: str = ""
    deadzone: float = 0.3
    min_hold: int = 18
    max_hold: int = 120
    bars: list[BarEntry] = field(default_factory=list)
    fills: list[FillEntry] = field(default_factory=list)


# ── Log parsing ──────────────────────────────────────────────────────

def parse_log(
    log_path: Path,
    *,
    symbol_filter: Optional[str] = None,
    since: Optional[datetime] = None,
) -> dict[str, SymbolSession]:
    """Parse Bybit alpha log into per-symbol sessions.

    Returns dict mapping symbol -> SymbolSession.
    """
    sessions: dict[str, SymbolSession] = {}

    if not log_path.exists():
        log.warning("Log file not found: %s", log_path)
        return sessions

    with open(log_path) as f:
        for line in f:
            line = line.rstrip()
            if not line:
                continue

            # Config line — update session params
            m = CONFIG_RE.search(line)
            if m:
                sym = m.group("symbol")
                if symbol_filter and sym != symbol_filter:
                    continue
                sess = sessions.setdefault(sym, SymbolSession(symbol=sym))
                sess.model = m.group("model")
                sess.deadzone = float(m.group("dz"))
                sess.min_hold = int(m.group("min_hold"))
                sess.max_hold = int(m.group("max_hold"))
                continue

            # Bar line (primary format)
            m = BAR_RE.match(line)
            if m:
                sym = m.group("symbol")
                if symbol_filter and sym != symbol_filter:
                    continue
                ts = _parse_ts(m.group("ts"))
                if since and ts < since:
                    continue
                trade_data = None
                if m.group("trade"):
                    trade_data = _safe_parse_dict(m.group("trade"))
                bar = BarEntry(
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
                sessions.setdefault(sym, SymbolSession(symbol=sym)).bars.append(bar)
                continue

            # Bar line (alternate format)
            m = BAR_ALT_RE.match(line)
            if m:
                sym = m.group("symbol")
                if symbol_filter and sym != symbol_filter:
                    continue
                ts = _parse_ts(m.group("ts"))
                if since and ts < since:
                    continue
                bar = BarEntry(
                    timestamp=ts, symbol=sym,
                    bar_num=int(m.group("bar")),
                    close=float(m.group("close")),
                    z=float(m.group("z")),
                    signal=int(m.group("sig")),
                    hold=int(m.group("hold")) if m.groupdict().get("hold") else 0,
                    regime=m.group("regime") or "active",
                    pred=float(m.group("pred")),
                    prev_signal=int(m.group("prev_sig")),
                )
                sessions.setdefault(sym, SymbolSession(symbol=sym)).bars.append(bar)
                continue

            # Open trade
            m = OPEN_RE.search(line)
            if m:
                sym = m.group("symbol")
                if symbol_filter and sym != symbol_filter:
                    continue
                ts = _parse_ts(m.group("ts"))
                if since and ts < since:
                    continue
                fill = FillEntry(
                    timestamp=ts, symbol=sym,
                    side=m.group("side").lower(),
                    price=float(m.group("price")),
                    size=float(m.group("size")),
                    fill_type="open",
                )
                sessions.setdefault(sym, SymbolSession(symbol=sym)).fills.append(fill)
                continue

            m = OPEN_ALT_RE.search(line)
            if m:
                ts = _parse_ts(m.group("ts"))
                if since and ts < since:
                    continue
                side_raw = m.group("side")
                side = "long" if side_raw == "buy" else "short"
                # Try to infer symbol from recent bar context
                fill = FillEntry(
                    timestamp=ts, symbol="UNKNOWN",
                    side=side,
                    price=float(m.group("price")),
                    size=float(m.group("qty")),
                    fill_type="open",
                )
                # Attach to first matching session by timestamp proximity
                _attach_fill_to_session(sessions, fill)
                continue

            # Close trade
            m = CLOSE_RE.search(line)
            if m:
                sym = m.group("symbol")
                if symbol_filter and sym != symbol_filter:
                    continue
                ts = _parse_ts(m.group("ts"))
                if since and ts < since:
                    continue
                fill = FillEntry(
                    timestamp=ts, symbol=sym,
                    side=m.group("side"),
                    price=0.0,  # not logged in close line
                    pnl=float(m.group("pnl")),
                    pnl_pct=float(m.group("pct")),
                    fill_type="close",
                    exit_type="signal",
                )
                sessions.setdefault(sym, SymbolSession(symbol=sym)).fills.append(fill)
                continue

            # Stop close
            m = STOP_RE.search(line)
            if m:
                sym = m.group("symbol")
                if symbol_filter and sym != symbol_filter:
                    continue
                ts = _parse_ts(m.group("ts"))
                if since and ts < since:
                    continue
                fill = FillEntry(
                    timestamp=ts, symbol=sym,
                    side="unknown",
                    price=0.0,
                    pnl=float(m.group("pnl")),
                    fill_type="close",
                    exit_type="stop",
                )
                sessions.setdefault(sym, SymbolSession(symbol=sym)).fills.append(fill)
                continue

    return sessions


def _parse_ts(ts_str: str) -> datetime:
    """Parse timestamp from log line."""
    return datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")


def _safe_parse_dict(raw: str) -> Optional[dict]:
    """Parse a Python-style dict string from log output."""
    try:
        s = raw.replace("'", '"')
        s = s.replace("True", "true").replace("False", "false").replace("None", "null")
        return json.loads(s)
    except Exception:
        return {"raw": raw}


def _attach_fill_to_session(sessions: dict[str, SymbolSession], fill: FillEntry) -> None:
    """Attach an UNKNOWN-symbol fill to the most likely session."""
    if not sessions:
        return
    # Use the first session (most common case: single symbol)
    sym = next(iter(sessions))
    fill.symbol = sym
    sessions[sym].fills.append(fill)


# ── Backtest replay ──────────────────────────────────────────────────

def replay_signals(
    bars: list[BarEntry],
    deadzone: float,
    min_hold: int,
    max_hold: int,
) -> list[dict]:
    """Replay bar z-scores through the constraint pipeline.

    Uses the same min-hold / max-hold / deadzone logic as AlphaRunner
    (via apply_constraints), but applied to the logged z-scores.
    This shows what the signal *should* have been given the same z values.
    """
    if not bars:
        return []

    z_values = np.array([b.z for b in bars])

    # Discretize: z > deadzone -> +1, z < -deadzone -> -1, else 0
    raw = np.where(z_values > deadzone, 1.0,
                   np.where(z_values < -deadzone, -1.0, 0.0))

    # Apply min-hold / max-hold (single-pass, matching Rust semantics)
    signal = np.zeros(len(raw))
    hold_count = 1
    signal[0] = raw[0]

    for i in range(1, len(raw)):
        prev = signal[i - 1]
        desired = raw[i]

        # Min-hold lockout
        if hold_count < min_hold and prev != 0:
            signal[i] = prev
            hold_count += 1
            continue

        # Max-hold forced exit
        if hold_count >= max_hold and prev != 0:
            signal[i] = 0.0
            hold_count = 1
            continue

        # Allow change
        if desired != prev:
            signal[i] = desired
            hold_count = 1
        else:
            signal[i] = desired
            hold_count += 1

    results = []
    for i, bar in enumerate(bars):
        results.append({
            "bar_num": bar.bar_num,
            "timestamp": bar.timestamp,
            "close": bar.close,
            "z": bar.z,
            "signal": int(signal[i]),
            "regime": bar.regime,
        })
    return results


def _extract_bt_trades(bt_results: list[dict]) -> list[dict]:
    """Extract round-trip trades from a signal sequence."""
    trades = []
    entry_bar = None
    entry_price = 0.0
    entry_side = 0

    for r in bt_results:
        sig = r["signal"]
        if entry_side == 0 and sig != 0:
            entry_bar = r["bar_num"]
            entry_price = r["close"]
            entry_side = sig
        elif entry_side != 0 and sig != entry_side:
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
            })
            if sig != 0:
                entry_bar = r["bar_num"]
                entry_price = r["close"]
                entry_side = sig
            else:
                entry_bar = None
                entry_price = 0.0
                entry_side = 0

    return trades


# ── Reconciliation logic ─────────────────────────────────────────────

def reconcile_symbol(session: SymbolSession) -> dict:
    """Run full reconciliation for one symbol.

    Returns a structured report dict with all metrics.
    """
    bars = session.bars
    fills = session.fills

    if not bars:
        return {
            "symbol": session.symbol,
            "status": "no_data",
            "error": "No bar entries found in log",
        }

    # Effective deadzone: prefer per-bar dynamic dz if available
    dz_values = [b.dz for b in bars if b.dz > 0]
    effective_dz = float(np.mean(dz_values)) if dz_values else session.deadzone
    if effective_dz <= 0:
        effective_dz = 0.3  # safe default

    # Replay signals through constraint pipeline
    bt_results = replay_signals(
        bars,
        deadzone=effective_dz,
        min_hold=session.min_hold,
        max_hold=session.max_hold,
    )

    # ── 1. Signal agreement ──────────────────────────────────────
    n_bars = len(bars)
    agreements = 0
    disagreements = []
    live_signals = []
    bt_signals = []

    for bar, bt in zip(bars, bt_results):
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
                "regime": bar.regime,
                "dz": bar.dz,
            })

    signal_match_rate = agreements / n_bars if n_bars > 0 else 0.0

    # Signal distribution
    live_dist = {-1: 0, 0: 0, 1: 0}
    bt_dist = {-1: 0, 0: 0, 1: 0}
    for s in live_signals:
        live_dist[s] = live_dist.get(s, 0) + 1
    for s in bt_signals:
        bt_dist[s] = bt_dist.get(s, 0) + 1

    # Signal transitions
    live_changes = sum(1 for i in range(1, len(live_signals))
                       if live_signals[i] != live_signals[i - 1])
    bt_changes = sum(1 for i in range(1, len(bt_signals))
                     if bt_signals[i] != bt_signals[i - 1])

    # ── 2. Execution slippage ────────────────────────────────────
    # Match open fills to the closest bar to compute fill vs close slippage
    open_fills = [f for f in fills if f.fill_type == "open" and f.price > 0]
    slippage_bps_list = []

    for fill in open_fills:
        # Find closest bar by timestamp
        closest_bar = _find_closest_bar(bars, fill.timestamp)
        if closest_bar and closest_bar.close > 0:
            slip_bps = (fill.price - closest_bar.close) / closest_bar.close * 10000
            # For short entries, slippage sign is inverted
            if fill.side in ("short", "sell"):
                slip_bps = -slip_bps
            slippage_bps_list.append(slip_bps)

    avg_fill_slippage_bps = (
        float(np.mean(slippage_bps_list)) if slippage_bps_list else 0.0
    )

    # ── 3. Timing delay ──────────────────────────────────────────
    # Detect bars where backtest signal changed but live signal was delayed
    timing_delays = []
    for i in range(1, min(len(live_signals), len(bt_signals))):
        bt_changed = bt_signals[i] != bt_signals[i - 1]
        live_changed = live_signals[i] != live_signals[i - 1]

        if bt_changed and not live_changed:
            # Backtest changed but live didn't — look ahead for live catch-up
            for j in range(i + 1, min(i + 10, len(live_signals))):
                if live_signals[j] == bt_signals[i]:
                    timing_delays.append(j - i)
                    break

    timing_delay_bars = float(np.mean(timing_delays)) if timing_delays else 0.0

    # ── 4. PnL comparison ────────────────────────────────────────
    close_fills = [f for f in fills if f.fill_type == "close" and f.pnl is not None]
    live_total_pnl = sum(f.pnl for f in close_fills)

    bt_trades = _extract_bt_trades(bt_results)
    bt_total_pnl_pct = sum(t["pnl_pct"] for t in bt_trades)

    # PnL gap: percentage difference
    if abs(bt_total_pnl_pct) > 0.001:
        # Approximate: compare live dollar PnL vs backtest pct-based PnL
        # They use different units, so we report both and the gap
        pnl_gap_pct = None  # Not directly comparable without position sizing
    else:
        pnl_gap_pct = None

    # Compute comparable PnL gap if we have both in same units
    live_pnl_pct_list = [f.pnl_pct for f in close_fills if f.pnl_pct is not None]
    live_total_pnl_pct = sum(live_pnl_pct_list)
    if abs(bt_total_pnl_pct) > 0.001:
        pnl_gap_pct = round(
            (live_total_pnl_pct - bt_total_pnl_pct) / abs(bt_total_pnl_pct) * 100, 2
        )

    # ── 5. Disagreement analysis ─────────────────────────────────
    # Categorize why signals diverged
    divergence_causes = {
        "regime_filter": 0,
        "hold_timing": 0,
        "z_reversal_exit": 0,
        "stop_loss": 0,
        "unknown": 0,
    }
    for d in disagreements:
        if d["regime"] != "active":
            divergence_causes["regime_filter"] += 1
        elif d["live_signal"] == 0 and d["bt_signal"] != 0:
            # Live went flat but backtest still in position — likely forced exit
            stop_near = any(
                f.exit_type == "stop"
                and abs((f.timestamp - _parse_ts(d["timestamp"])).total_seconds()) < 7200
                for f in fills
                if f.fill_type == "close"
            )
            if stop_near:
                divergence_causes["stop_loss"] += 1
            else:
                divergence_causes["z_reversal_exit"] += 1
        elif abs(d.get("live_hold", 0)) > 0:
            divergence_causes["hold_timing"] += 1
        else:
            divergence_causes["unknown"] += 1

    # ── Build report ─────────────────────────────────────────────
    report = {
        "symbol": session.symbol,
        "status": "ok",
        "model": session.model,
        "config": {
            "deadzone": session.deadzone,
            "effective_dz": round(effective_dz, 4),
            "min_hold": session.min_hold,
            "max_hold": session.max_hold,
        },
        "period": {
            "start": str(bars[0].timestamp),
            "end": str(bars[-1].timestamp),
            "n_bars": n_bars,
            "n_days": round((bars[-1].timestamp - bars[0].timestamp).total_seconds() / 86400, 1),
        },
        "metrics": {
            "signal_match_rate": round(signal_match_rate, 4),
            "avg_fill_slippage_bps": round(avg_fill_slippage_bps, 2),
            "timing_delay_bars": round(timing_delay_bars, 2),
            "pnl_gap_pct": pnl_gap_pct,
        },
        "signal_distribution": {
            "live": live_dist,
            "backtest": bt_dist,
        },
        "signal_changes": {
            "live": live_changes,
            "backtest": bt_changes,
        },
        "fills": {
            "open_count": len(open_fills),
            "close_count": len(close_fills),
            "live_total_pnl_usd": round(live_total_pnl, 4),
            "live_total_pnl_pct": round(live_total_pnl_pct, 4),
            "bt_total_pnl_pct": round(bt_total_pnl_pct, 4),
            "bt_trade_count": len(bt_trades),
        },
        "slippage_samples": len(slippage_bps_list),
        "timing_delay_samples": len(timing_delays),
        "divergence_causes": divergence_causes,
        "disagreements_sample": disagreements[:20],
    }

    return report


def _find_closest_bar(bars: list[BarEntry], ts: datetime) -> Optional[BarEntry]:
    """Find the bar closest in time to a given timestamp."""
    if not bars:
        return None
    best = bars[0]
    best_delta = abs((bars[0].timestamp - ts).total_seconds())
    for b in bars[1:]:
        delta = abs((b.timestamp - ts).total_seconds())
        if delta < best_delta:
            best = b
            best_delta = delta
    return best


# ── Report formatting ────────────────────────────────────────────────

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


# ── Main ─────────────────────────────────────────────────────────────

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
