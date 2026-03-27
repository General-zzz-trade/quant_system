#!/usr/bin/env python3
"""Simple terminal dashboard for quant system monitoring.

Displays account equity, positions, recent trades, model z-scores,
system health, and rolling Sharpe. Refreshes every 30 seconds.

Usage:
    python3 scripts/dashboard.py           # Live refresh every 30s
    python3 scripts/dashboard.py --once    # Single snapshot and exit
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is on path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
os.chdir(_PROJECT_ROOT)

from execution.adapters.bybit.adapter import BybitAdapter  # noqa: E402
from execution.adapters.bybit.config import BybitConfig  # noqa: E402

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("dashboard")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RUNTIME_DIR = Path("data/runtime")
AUDIT_FILE = RUNTIME_DIR / "decision_audit.jsonl"
ZSCORE_DIR = RUNTIME_DIR / "zscore_checkpoints"
HEALTH_FILE = RUNTIME_DIR / "health_status.json"
IC_FILE = RUNTIME_DIR / "ic_health.json"
LOG_FILE = Path("logs/bybit_alpha.log")

# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
_RESET = "\033[0m"
_BOLD = "\033[1m"
_RED = "\033[91m"
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_CYAN = "\033[96m"
_DIM = "\033[2m"

WIDTH = 72


def _color_pnl(val: float) -> str:
    """Color a PnL value green/red."""
    color = _GREEN if val >= 0 else _RED
    return f"{color}{val:+.2f}{_RESET}"


def _color_signal(sig: int) -> str:
    if sig > 0:
        return f"{_GREEN}LONG{_RESET}"
    elif sig < 0:
        return f"{_RED}SHORT{_RESET}"
    return f"{_DIM}FLAT{_RESET}"


def _color_status(status: str) -> str:
    s = status.lower()
    if s in ("healthy", "active", "fresh", "ok", "green"):
        return f"{_GREEN}{status}{_RESET}"
    elif s in ("warning", "yellow", "stale"):
        return f"{_YELLOW}{status}{_RESET}"
    else:
        return f"{_RED}{status}{_RESET}"


def _header(title: str) -> str:
    return f"\n{_BOLD}{_CYAN}{'=' * WIDTH}{_RESET}\n{_BOLD}  {title}{_RESET}\n{_BOLD}{_CYAN}{'=' * WIDTH}{_RESET}"


def _sep() -> str:
    return f"{_DIM}{'-' * WIDTH}{_RESET}"


def _clear_screen() -> None:
    """Clear terminal screen using ANSI escape."""
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Data sources
# ---------------------------------------------------------------------------

def _create_adapter() -> Optional[BybitAdapter]:
    """Create Bybit adapter from env vars."""
    api_key = os.environ.get("BYBIT_API_KEY", "")
    api_secret = os.environ.get("BYBIT_API_SECRET", "")
    base_url = os.environ.get("BYBIT_BASE_URL", "https://api-demo.bybit.com")
    if not api_key or not api_secret:
        return None
    config = BybitConfig(
        api_key=api_key,
        api_secret=api_secret,
        base_url=base_url,
    )
    adapter = BybitAdapter(config)
    try:
        adapter.connect()
    except Exception as e:
        logger.warning("Bybit connection failed: %s", e)
        return None
    return adapter


def _get_equity(adapter: Optional[BybitAdapter]) -> Tuple[Optional[float], Optional[float]]:
    """Return (total_equity, available_balance) or (None, None)."""
    if adapter is None:
        return None, None
    try:
        snap = adapter.get_balances()
        for b in snap.balances:
            if b.asset == "USDT":
                return float(b.total), float(b.free)
        return None, None
    except Exception:
        return None, None


def _get_positions(adapter: Optional[BybitAdapter]) -> list:
    """Return list of VenuePosition with non-zero qty."""
    if adapter is None:
        return []
    try:
        return list(adapter.get_positions())
    except Exception:
        return []


def _get_recent_audit(n: int = 5) -> List[Dict[str, Any]]:
    """Read last n signal entries from decision_audit.jsonl."""
    if not AUDIT_FILE.exists():
        return []
    try:
        lines: deque = deque(maxlen=200)
        with open(AUDIT_FILE, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    lines.append(line)
        results = []
        for line in reversed(lines):
            try:
                rec = json.loads(line)
                results.append(rec)
                if len(results) >= n:
                    break
            except json.JSONDecodeError:
                continue
        return results
    except Exception:
        return []


def _get_recent_trades(n: int = 5) -> List[Dict[str, Any]]:
    """Get recent trade entries/exits from decision audit."""
    if not AUDIT_FILE.exists():
        return []
    try:
        lines: deque = deque(maxlen=500)
        with open(AUDIT_FILE, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    lines.append(line)
        results = []
        for line in reversed(lines):
            try:
                rec = json.loads(line)
                if rec.get("type") in ("entry", "exit", "force_exit"):
                    results.append(rec)
                    if len(results) >= n:
                        break
            except json.JSONDecodeError:
                continue
        return results
    except Exception:
        return []


def _get_zscore_status() -> Dict[str, Dict[str, Any]]:
    """Load z-score checkpoint data per runner."""
    result = {}
    if not ZSCORE_DIR.exists():
        return result
    for fp in ZSCORE_DIR.glob("*.json"):
        runner = fp.stem
        try:
            data = json.loads(fp.read_text())
            # zscore_buf / position / hold_counter can be dicts keyed by symbol
            z_buf_raw = data.get("zscore_buf", {})
            pos_raw = data.get("position", {})
            hold_raw = data.get("hold_counter", {})

            # Handle both dict-keyed and flat formats
            if isinstance(z_buf_raw, dict):
                for sym, buf in z_buf_raw.items():
                    latest_z = buf[-1] if isinstance(buf, list) and buf else None
                    pos = pos_raw.get(sym, 0) if isinstance(pos_raw, dict) else pos_raw
                    hold = hold_raw.get(sym, 0) if isinstance(hold_raw, dict) else hold_raw
                    key = runner if len(z_buf_raw) == 1 else f"{runner}/{sym}"
                    result[key] = {
                        "z_score": latest_z,
                        "position": int(pos) if pos else 0,
                        "hold_bars": int(hold) if hold else 0,
                        "buf_len": len(buf) if isinstance(buf, list) else 0,
                    }
            elif isinstance(z_buf_raw, list):
                latest_z = z_buf_raw[-1] if z_buf_raw else None
                pos = pos_raw if not isinstance(pos_raw, dict) else 0
                hold = hold_raw if not isinstance(hold_raw, dict) else 0
                result[runner] = {
                    "z_score": latest_z,
                    "position": int(pos) if pos else 0,
                    "hold_bars": int(hold) if hold else 0,
                    "buf_len": len(z_buf_raw),
                }
        except Exception:
            continue
    return result


def _get_health_status() -> Dict[str, Any]:
    """Load health_status.json."""
    if not HEALTH_FILE.exists():
        return {}
    try:
        return json.loads(HEALTH_FILE.read_text())
    except Exception:
        return {}


def _get_ic_health() -> Dict[str, Any]:
    """Load IC health data."""
    if not IC_FILE.exists():
        return {}
    try:
        return json.loads(IC_FILE.read_text())
    except Exception:
        return {}


def _get_service_status(unit: str) -> str:
    """Check systemd service status."""
    try:
        result = subprocess.run(
            ["systemctl", "is-active", unit],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def _get_monitor_lines(n: int = 4) -> List[str]:
    """Get last n MONITOR lines from log file."""
    if not LOG_FILE.exists():
        return []
    try:
        lines: deque = deque(maxlen=n)
        with open(LOG_FILE, "r") as f:
            for line in f:
                if "MONITOR" in line:
                    lines.append(line.strip())
        return list(lines)
    except Exception:
        return []


def _compute_daily_pnl(adapter: Optional[BybitAdapter]) -> Optional[float]:
    """Estimate daily PnL from closed PnL API (Bybit V5)."""
    if adapter is None:
        return None
    try:
        now_ms = int(time.time() * 1000)
        start_of_day = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0,
        )
        start_ms = int(start_of_day.timestamp() * 1000)
        data = adapter._client.get("/v5/position/closed-pnl", {
            "category": "linear",
            "startTime": str(start_ms),
            "endTime": str(now_ms),
            "limit": "50",
        })
        if data.get("retCode") != 0:
            return None
        items = data.get("result", {}).get("list", [])
        total_pnl = sum(float(item.get("closedPnl", "0")) for item in items)
        return total_pnl
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def render_dashboard(adapter: Optional[BybitAdapter]) -> str:
    """Build the full dashboard string."""
    lines: List[str] = []
    now = datetime.now(timezone.utc)

    lines.append(f"\n{_BOLD}{_CYAN}  QUANT SYSTEM DASHBOARD{_RESET}")
    lines.append(f"  {now.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    base_url = os.environ.get("BYBIT_BASE_URL", "")
    mode = "LIVE" if "api.bybit.com" == base_url.replace("https://", "").split("/")[0] else "DEMO"
    mode_color = _RED if mode == "LIVE" else _GREEN
    lines.append(f"  Mode: {mode_color}{_BOLD}{mode}{_RESET}")

    # --- Account ---
    lines.append(_header("ACCOUNT"))
    equity, available = _get_equity(adapter)
    daily_pnl = _compute_daily_pnl(adapter)
    if equity is not None:
        lines.append(f"  Equity:     ${equity:,.2f} USDT")
        lines.append(f"  Available:  ${available:,.2f} USDT")
        if daily_pnl is not None:
            pct = (daily_pnl / equity * 100) if equity > 0 else 0
            lines.append(f"  Daily PnL:  {_color_pnl(daily_pnl)} USDT ({pct:+.2f}%)")
    else:
        lines.append(f"  {_YELLOW}No API connection{_RESET}")

    # --- Positions ---
    lines.append(_header("POSITIONS"))
    positions = _get_positions(adapter)
    if positions:
        lines.append(f"  {'Symbol':<12} {'Side':<6} {'Qty':>10} {'Entry':>10} {'Mark':>10} {'uPnL':>10}")
        lines.append(f"  {_sep()}")
        for p in positions:
            side = "LONG" if p.is_long else "SHORT"
            side_color = _GREEN if p.is_long else _RED
            entry = f"{float(p.entry_price):,.2f}" if p.entry_price else "N/A"
            mark = f"{float(p.mark_price):,.2f}" if p.mark_price else "N/A"
            upnl = float(p.unrealized_pnl)
            lines.append(
                f"  {p.symbol:<12} {side_color}{side:<6}{_RESET} {float(p.qty):>10.4f} "
                f"{entry:>10} {mark:>10} {_color_pnl(upnl):>10}"
            )
    else:
        lines.append(f"  {_DIM}No open positions{_RESET}")

    # --- Z-Score Status ---
    lines.append(_header("MODEL Z-SCORES"))
    z_status = _get_zscore_status()
    if z_status:
        lines.append(f"  {'Runner':<16} {'Z-Score':>8} {'Position':>10} {'Hold':>6} {'Buffer':>8}")
        lines.append(f"  {_sep()}")
        for runner, info in sorted(z_status.items()):
            z = info.get("z_score")
            z_str = f"{z:+.3f}" if z is not None else "N/A"
            pos = info.get("position", 0)
            pos_str = _color_signal(pos)
            hold = info.get("hold_bars", 0)
            buf = info.get("buf_len", 0)
            lines.append(
                f"  {runner:<16} {z_str:>8} {pos_str:>10} {hold:>6} {buf:>8}"
            )
    else:
        lines.append(f"  {_DIM}No z-score checkpoints found{_RESET}")

    # --- Recent Signals ---
    lines.append(_header("LATEST SIGNALS (from audit)"))
    recent = _get_recent_audit(n=5)
    if recent:
        for rec in recent:
            ts = rec.get("ts", 0)
            dt = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%m-%d %H:%M")
            sym = rec.get("symbol", "?")
            rtype = rec.get("type", "?")
            z = rec.get("z_score")
            sig = rec.get("signal", 0)
            z_str = f"z={z:+.2f}" if z is not None else ""
            lines.append(f"  {dt}  {sym:<12} {rtype:<10} {_color_signal(sig)}  {z_str}")
    else:
        lines.append(f"  {_DIM}No audit data{_RESET}")

    # --- Recent Trades ---
    trades = _get_recent_trades(n=5)
    if trades:
        lines.append(_header("RECENT TRADES"))
        for rec in trades:
            ts = rec.get("ts", 0)
            dt = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%m-%d %H:%M")
            sym = rec.get("symbol", "?")
            rtype = rec.get("type", "?")
            sig = rec.get("signal", 0)
            reason = rec.get("force_exit") or ""
            lines.append(f"  {dt}  {sym:<12} {rtype:<10} {_color_signal(sig)}  {reason}")

    # --- System Health ---
    lines.append(_header("SYSTEM HEALTH"))

    # Service status
    service_status = _get_service_status("bybit-alpha.service")
    lines.append(f"  bybit-alpha:     {_color_status(service_status)}")

    # WS connection check from health file
    health = _get_health_status()
    if health:
        ts = health.get("timestamp", "")
        if ts:
            lines.append(f"  Last health check: {ts[:19]}")
        checks = health.get("checks", {})
        svc = checks.get("service_bybit-alpha", {})
        log_age = svc.get("last_log_age_s")
        if log_age is not None:
            age_str = f"{log_age:.0f}s ago"
            color = _GREEN if log_age < 120 else (_YELLOW if log_age < 300 else _RED)
            lines.append(f"  Last log activity: {color}{age_str}{_RESET}")

    # IC health
    ic = _get_ic_health()
    if ic:
        lines.append(f"  {_sep()}")
        lines.append(f"  {'Model':<20} {'IC Status':<10}")
        for model, info in ic.items():
            if isinstance(info, dict):
                status = info.get("status", "unknown")
                lines.append(f"  {model:<20} {_color_status(status)}")

    # --- Monitor lines ---
    monitor_lines = _get_monitor_lines(n=4)
    if monitor_lines:
        lines.append(_header("LIVE MONITOR (latest)"))
        for ml in monitor_lines:
            # Trim timestamp prefix for cleaner display
            parts = ml.split("MONITOR", 1)
            if len(parts) == 2:
                lines.append(f"  MONITOR{parts[1]}")
            else:
                lines.append(f"  {ml[:WIDTH]}")

    # Footer
    lines.append(f"\n{_DIM}  Refresh: 30s | Press Ctrl+C to exit{_RESET}\n")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Quant System Terminal Dashboard")
    parser.add_argument("--once", action="store_true", help="Single snapshot and exit")
    parser.add_argument("--interval", type=int, default=30, help="Refresh interval in seconds")
    args = parser.parse_args()

    # Create adapter once
    adapter = _create_adapter()

    if args.once:
        print(render_dashboard(adapter))
        return

    try:
        while True:
            _clear_screen()
            print(render_dashboard(adapter))
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print(f"\n{_DIM}Dashboard stopped.{_RESET}")


if __name__ == "__main__":
    main()
