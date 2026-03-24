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

import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
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
    """Run full reconciliation for one symbol. Delegates to engine module."""
    from monitoring.daily_reconcile_engine import reconcile_symbol as _reconcile
    return _reconcile(session)


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


# Re-export report functions for backward compatibility
from monitoring.daily_reconcile_report import print_report, main  # noqa: F401, E402


if __name__ == "__main__":
    sys.exit(main())
