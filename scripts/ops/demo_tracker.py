"""Demo performance tracker — parses logs/bybit_alpha.log → data/live/track_record.json.

Usage:
    python3 -m scripts.ops.demo_tracker [--log logs/bybit_alpha.log] [--reset]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Regex patterns for log line parsing
# ---------------------------------------------------------------------------

# WS ETHUSDT bar 123: $3500.00 z=1.23 sig=1 hold=5 regime=trending dz=0.5
_RE_BAR = re.compile(
    r"WS\s+(?P<symbol>\S+)\s+bar\s+(?P<bar_n>\d+):\s+\$(?P<price>[\d.]+)"
    r"\s+z=(?P<z>[+-]?[\d.]+)"
    r"\s+sig=(?P<sig>[+-]?\d+)"
    r"\s+hold=(?P<hold>\d+)"
    r"\s+regime=(?P<regime>\S+)"
    r"\s+dz=(?P<dz>[\d.]+)"
)

# ETHUSDT CLOSE LONG: pnl=$12.50 (0.25%) total=$150.00 wins=3/5
_RE_CLOSE = re.compile(
    r"(?P<symbol>\S+)\s+CLOSE\s+(?P<side>LONG|SHORT):\s+"
    r"pnl=\$(?P<pnl>[+-]?[\d.]+)\s+"
    r"\((?P<pct>[+-]?[\d.]+)%\)\s+"
    r"total=\$(?P<total>[+-]?[\d.]+)\s+"
    r"wins=(?P<wins>\d+)/(?P<total_trades>\d+)"
)

# Opened LONG 0.01 @ ~$3500.00 stop=$3430.00
_RE_OPEN = re.compile(
    r"Opened\s+(?P<side>LONG|SHORT)\s+(?P<qty>[\d.]+)\s+"
    r"@\s+~\$(?P<price>[\d.]+)"
    r"(?:\s+stop=\$(?P<stop>[\d.]+))?"
)

# ---------------------------------------------------------------------------
# Track record template
# ---------------------------------------------------------------------------

TRACK_RECORD_PATH = Path("data/live/track_record.json")


def _empty_record() -> dict[str, Any]:
    return {
        "last_parsed_offset": 0,
        "last_updated": "",
        "daily": {},
        "summary": {
            "total_pnl_usd": 0.0,
            "total_trades": 0,
            "total_wins": 0,
            "total_losses": 0,
            "win_rate": 0.0,
            "sharpe_7d": None,
            "sharpe_30d": None,
            "symbols": [],
        },
    }


def _empty_day() -> dict[str, Any]:
    return {
        "symbols": {},
        "total_pnl_usd": 0.0,
        "total_trades": 0,
        "max_drawdown": 0.0,
    }


def _empty_symbol_day() -> dict[str, Any]:
    return {
        "bars": 0,
        "signals": {"long": 0, "short": 0, "flat": 0},
        "trades": 0,
        "pnl_usd": 0.0,
        "wins": 0,
        "losses": 0,
    }


# ---------------------------------------------------------------------------
# Line parsers
# ---------------------------------------------------------------------------


def parse_bar_line(line: str) -> dict[str, Any] | None:
    """Parse a WS bar log line.

    Returns a dict with keys: symbol, bar_n, price, z, sig, hold, regime, dz
    or None if the line does not match.
    """
    m = _RE_BAR.search(line)
    if not m:
        return None
    try:
        return {
            "symbol": m.group("symbol"),
            "bar_n": int(m.group("bar_n")),
            "price": float(m.group("price")),
            "z": float(m.group("z")),
            "sig": int(m.group("sig")),
            "hold": int(m.group("hold")),
            "regime": m.group("regime"),
            "dz": float(m.group("dz")),
        }
    except (ValueError, TypeError):
        return None


def parse_close_line(line: str) -> dict[str, Any] | None:
    """Parse a CLOSE trade log line.

    Returns a dict with keys: symbol, side, pnl_usd, pct, total_usd, wins, total_trades
    or None if the line does not match.
    """
    m = _RE_CLOSE.search(line)
    if not m:
        return None
    try:
        return {
            "symbol": m.group("symbol"),
            "side": m.group("side"),
            "pnl_usd": float(m.group("pnl")),
            "pct": float(m.group("pct")),
            "total_usd": float(m.group("total")),
            "wins": int(m.group("wins")),
            "total_trades": int(m.group("total_trades")),
        }
    except (ValueError, TypeError):
        return None


def parse_open_line(line: str) -> dict[str, Any] | None:
    """Parse an Opened trade log line.

    Returns a dict with keys: side, qty, price, stop (may be None)
    or None if the line does not match.
    """
    m = _RE_OPEN.search(line)
    if not m:
        return None
    try:
        stop_str = m.group("stop")
        return {
            "side": m.group("side"),
            "qty": float(m.group("qty")),
            "price": float(m.group("price")),
            "stop": float(stop_str) if stop_str else None,
        }
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def load_track_record(path: Path) -> dict[str, Any]:
    """Load track record JSON from *path*, or return an empty template if missing/corrupt."""
    if not path.exists():
        return _empty_record()
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        # Ensure required top-level keys exist (graceful upgrade)
        template = _empty_record()
        for key, default in template.items():
            data.setdefault(key, default)
        return data
    except (json.JSONDecodeError, OSError) as exc:
        warnings.warn(f"demo_tracker: corrupt track record at {path}: {exc}; resetting")
        return _empty_record()


def save_track_record(record: dict[str, Any], path: Path) -> None:
    """Atomically save *record* to *path* (write .tmp then rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as fh:
            json.dump(record, fh, indent=2, default=str)
        os.replace(tmp_path, path)
    except OSError as exc:
        # Clean up tmp on failure
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise exc


# ---------------------------------------------------------------------------
# Incremental parser
# ---------------------------------------------------------------------------


def parse_incremental(log_path: Path, record: dict[str, Any]) -> int:
    """Seek to last_parsed_offset in *log_path*, parse new lines, update *record* in-place.

    Returns number of lines parsed (not bytes).
    """
    if not log_path.exists():
        return 0

    offset: int = record.get("last_parsed_offset", 0)
    lines_parsed = 0

    with log_path.open("r", encoding="utf-8", errors="replace") as fh:
        # If the file shrank (log rotation), restart from 0
        fh.seek(0, 2)
        file_size = fh.tell()
        if offset > file_size:
            offset = 0
        fh.seek(offset)

        for raw_line in fh:
            line = raw_line.rstrip("\n")
            lines_parsed += 1
            _process_line(line, record)

        record["last_parsed_offset"] = fh.tell()

    record["last_updated"] = datetime.now(timezone.utc).isoformat()
    _recompute_summary(record)
    return lines_parsed


def _today_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _ensure_day(record: dict[str, Any], date_str: str) -> dict[str, Any]:
    daily: dict[str, Any] = record.setdefault("daily", {})
    if date_str not in daily:
        daily[date_str] = _empty_day()
    return daily[date_str]


def _ensure_sym_day(day: dict[str, Any], symbol: str) -> dict[str, Any]:
    syms: dict[str, Any] = day.setdefault("symbols", {})
    if symbol not in syms:
        syms[symbol] = _empty_symbol_day()
    return syms[symbol]


def _process_line(line: str, record: dict[str, Any]) -> None:
    """Dispatch a single log line to the appropriate handler."""
    # Bar line
    bar = parse_bar_line(line)
    if bar is not None:
        date_str = _today_str()
        day = _ensure_day(record, date_str)
        sym = _ensure_sym_day(day, bar["symbol"])
        sym["bars"] += 1
        sig = bar["sig"]
        if sig > 0:
            sym["signals"]["long"] += 1
        elif sig < 0:
            sym["signals"]["short"] += 1
        else:
            sym["signals"]["flat"] += 1
        return

    # Close line
    close = parse_close_line(line)
    if close is not None:
        date_str = _today_str()
        day = _ensure_day(record, date_str)
        sym = _ensure_sym_day(day, close["symbol"])
        sym["trades"] += 1
        sym["pnl_usd"] += close["pnl_usd"]
        if close["pnl_usd"] >= 0:
            sym["wins"] += 1
        else:
            sym["losses"] += 1
        day["total_pnl_usd"] += close["pnl_usd"]
        day["total_trades"] += 1
        # Update max_drawdown (track cumulative minimum daily pnl)
        if day["total_pnl_usd"] < day["max_drawdown"]:
            day["max_drawdown"] = day["total_pnl_usd"]
        return

    # Open line — not aggregated to daily stats, but could be extended


def _recompute_summary(record: dict[str, Any]) -> None:
    """Recompute summary fields from daily data."""
    daily: dict[str, Any] = record.get("daily", {})
    total_pnl = 0.0
    total_trades = 0
    total_wins = 0
    total_losses = 0
    all_symbols: set[str] = set()
    daily_pnls_sorted: list[tuple[str, float]] = []

    for date_str, day in daily.items():
        total_pnl += day.get("total_pnl_usd", 0.0)
        total_trades += day.get("total_trades", 0)
        for sym, sym_data in day.get("symbols", {}).items():
            all_symbols.add(sym)
            total_wins += sym_data.get("wins", 0)
            total_losses += sym_data.get("losses", 0)
        daily_pnls_sorted.append((date_str, day.get("total_pnl_usd", 0.0)))

    daily_pnls_sorted.sort(key=lambda x: x[0])
    pnl_series = [v for _, v in daily_pnls_sorted]

    win_rate = total_wins / (total_wins + total_losses) if (total_wins + total_losses) > 0 else 0.0

    summary = record.setdefault("summary", {})
    summary["total_pnl_usd"] = round(total_pnl, 4)
    summary["total_trades"] = total_trades
    summary["total_wins"] = total_wins
    summary["total_losses"] = total_losses
    summary["win_rate"] = round(win_rate, 4)
    summary["sharpe_7d"] = compute_rolling_sharpe(pnl_series, window=7)
    summary["sharpe_30d"] = compute_rolling_sharpe(pnl_series, window=30)
    summary["symbols"] = sorted(all_symbols)


# ---------------------------------------------------------------------------
# Sharpe computation
# ---------------------------------------------------------------------------


def compute_rolling_sharpe(daily_pnl: list[float], window: int) -> float | None:
    """Compute annualised Sharpe over the last *window* days of *daily_pnl*.

    Uses daily P&L std; returns None if insufficient data or zero std.
    Annualisation factor: sqrt(252).
    """
    if not daily_pnl:
        return None
    subset = daily_pnl[-window:]
    n = len(subset)
    if n < 2:
        return None
    mean = sum(subset) / n
    variance = sum((x - mean) ** 2 for x in subset) / (n - 1)
    std = math.sqrt(variance)
    if std == 0.0:
        return None
    sharpe = (mean / std) * math.sqrt(252)
    return round(sharpe, 4)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse bybit_alpha.log incrementally into data/live/track_record.json"
    )
    parser.add_argument(
        "--log",
        default="logs/bybit_alpha.log",
        help="Path to log file (default: logs/bybit_alpha.log)",
    )
    parser.add_argument(
        "--out",
        default=str(TRACK_RECORD_PATH),
        help=f"Output JSON path (default: {TRACK_RECORD_PATH})",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Clear existing track record and re-parse from beginning",
    )
    args = parser.parse_args()

    log_path = Path(args.log)
    out_path = Path(args.out)

    if args.reset:
        record: dict[str, Any] = _empty_record()
        print(f"demo_tracker: reset track record at {out_path}")
    else:
        record = load_track_record(out_path)

    lines = parse_incremental(log_path, record)
    save_track_record(record, out_path)

    summary = record.get("summary", {})
    print(
        f"demo_tracker: parsed {lines} new lines | "
        f"trades={summary.get('total_trades', 0)} "
        f"pnl=${summary.get('total_pnl_usd', 0.0):.2f} "
        f"win_rate={summary.get('win_rate', 0.0):.1%} "
        f"sharpe_7d={summary.get('sharpe_7d')}"
    )


if __name__ == "__main__":
    main()
