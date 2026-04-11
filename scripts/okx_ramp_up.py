#!/usr/bin/env python3
"""OKX notional-cap gradual ramp-up.

When the OKX live runner is first deployed the per-order notional cap is
intentionally tiny ($50) so that the first few fills can be inspected for
qty-conversion or pricing bugs.  This script reads the real fill history
and automatically raises the cap as trust is established.

Ramp schedule (conservative, only raises — never lowers)::

    Level 0   $50     (initial)  → start here
    Level 1   $150    requires:  ≥ 5 fills in last 3 days, 0 rejects
    Level 2   $300    requires:  ≥ 15 fills total at ≥ L1, cum PnL > -5%
    Level 3   $800    requires:  ≥ 30 fills total,        cum PnL > 0
    Level 4   unlocked (writes OKX_MAX_ORDER_NOTIONAL=0 = no cap) —
                         requires ≥ 100 fills + 14-day PnL > +2%

State is persisted in ``data/runtime/okx_ramp_state.json`` so the script
is idempotent.  Call the binance-alpha runner's SIGHUP after updating
.env so the live process picks up the new cap without a restart.

Intended to run hourly (or every 6h) via systemd timer; safe to invoke
manually::

    python3 -m scripts.okx_ramp_up            # check + maybe promote
    python3 -m scripts.okx_ramp_up --dry-run  # report only
    python3 -m scripts.okx_ramp_up --force    # bypass gates (manual)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, "/quant_system")

from monitoring.decision_audit import audit_path_for

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────
STATE_PATH = Path("/quant_system/data/runtime/okx_ramp_state.json")
ENV_PATH = Path("/quant_system/.env")
ENV_VAR = "OKX_MAX_ORDER_NOTIONAL"

LOOKBACK_HOURS = 72  # 3 days


@dataclass(frozen=True, slots=True)
class RampLevel:
    level: int
    cap_usd: float
    min_fills: int            # cumulative required fills (queried from OKX)
    recent_window_hours: int  # how far back to look for fills
    min_cum_pnl_pct: float    # min cumulative PnL as fraction of equity
    description: str


RAMP_SCHEDULE: list[RampLevel] = [
    RampLevel(level=0, cap_usd=50.0,  min_fills=0,   recent_window_hours=0,    min_cum_pnl_pct=-1.0, description="initial (tiny cap for bug inspection)"),
    RampLevel(level=1, cap_usd=150.0, min_fills=5,   recent_window_hours=72,   min_cum_pnl_pct=-1.0, description="5 fills in 3 days, 0 rejects"),
    RampLevel(level=2, cap_usd=300.0, min_fills=15,  recent_window_hours=168,  min_cum_pnl_pct=-0.05, description="15 fills in 1 week, PnL > -5%"),
    RampLevel(level=3, cap_usd=800.0, min_fills=30,  recent_window_hours=168,  min_cum_pnl_pct=0.0,   description="30 fills in 1 week, PnL > 0"),
    RampLevel(level=4, cap_usd=0.0,   min_fills=100, recent_window_hours=336, min_cum_pnl_pct=0.02,  description="100 fills in 2 weeks, PnL > +2% → cap removed"),
]


# ── State management ──────────────────────────────────────────────
def _load_state() -> dict[str, Any]:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text())
        except Exception as e:
            logger.warning("ramp state read failed: %s", e)
    return {
        "current_level": 0,
        "current_cap": 50.0,
        "history": [],
        "last_check_ts": 0.0,
    }


def _save_state(state: dict[str, Any]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(STATE_PATH)


# ── .env atomic writer ────────────────────────────────────────────
def _update_env_cap(new_cap: float) -> None:
    """Update OKX_MAX_ORDER_NOTIONAL in /quant_system/.env atomically.

    Preserves every other line.  Chmod 600 after write.
    """
    import stat
    import re

    if not ENV_PATH.exists():
        raise RuntimeError(f"{ENV_PATH} missing")

    lines = ENV_PATH.read_text(encoding="utf-8").splitlines()
    out: list[str] = []
    found = False
    pattern = re.compile(rf"^\s*{re.escape(ENV_VAR)}\s*=")
    for line in lines:
        if pattern.match(line):
            out.append(f"{ENV_VAR}={int(new_cap)}")
            found = True
        else:
            out.append(line)
    if not found:
        out.append(f"{ENV_VAR}={int(new_cap)}")

    tmp = ENV_PATH.with_suffix(".env.tmp")
    tmp.write_text("\n".join(out) + "\n", encoding="utf-8")
    os.chmod(tmp, stat.S_IRUSR | stat.S_IWUSR)  # 0600 before rename
    tmp.replace(ENV_PATH)
    os.chmod(ENV_PATH, stat.S_IRUSR | stat.S_IWUSR)


def _sighup_okx_runner() -> bool:
    """Ask systemd for the okx-alpha PID and send SIGHUP for hot-reload."""
    try:
        res = subprocess.run(
            ["systemctl", "show", "-p", "MainPID", "okx-alpha.service"],
            capture_output=True, text=True, timeout=5,
        )
        if res.returncode != 0:
            return False
        line = res.stdout.strip()
        if "=" not in line:
            return False
        pid_str = line.split("=", 1)[1].strip()
        if not pid_str or pid_str == "0":
            return False
        pid = int(pid_str)
    except Exception as e:
        logger.warning("systemctl lookup failed: %s", e)
        return False
    try:
        r = subprocess.run(
            ["sudo", "-n", "kill", "-HUP", str(pid)],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0:
            logger.info("SIGHUP sent to okx-alpha (pid=%d)", pid)
            return True
    except Exception as e:
        logger.warning("sudo SIGHUP failed: %s", e)
    return False


# ── Fill/PnL statistics ───────────────────────────────────────────
def _read_audit_events(venue: str, since_ts: float) -> list[dict]:
    path = audit_path_for(venue)
    if not path.exists():
        return []
    out: list[dict] = []
    try:
        for line in path.read_text().splitlines():
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("ts", 0) >= since_ts:
                out.append(e)
    except Exception as e:
        logger.warning("audit read failed for %s: %s", path, e)
    return out


def _compute_fill_stats(level: RampLevel) -> dict[str, Any]:
    """Query OKX live fills + local audit to compute ramp decision inputs."""
    # Fetch fills directly from OKX (source of truth, even if audit log missing)
    from execution.adapters.okx.adapter import OkxAdapter
    from execution.adapters.okx.config import OkxConfig

    window_s = max(level.recent_window_hours, 72) * 3600
    since_ts = time.time() - window_s

    fills_count = 0
    rejects_count = 0
    cum_pnl = 0.0
    equity_usd = 0.0

    try:
        cfg = OkxConfig.from_env()
        adapter = OkxAdapter(cfg)
        if adapter.connect():
            # get_recent_fills returns latest ~10 by default; we need more for
            # ramp decisions, so call with a higher limit.
            fills = adapter.get_recent_fills(limit=100)
            for f in fills:
                if f.ts_ms / 1000 >= since_ts:
                    fills_count += 1
            # Current equity (for PnL% calculation)
            bal = adapter.get_balances()
            usdt = bal.get("USDT")
            if usdt:
                equity_usd = float(usdt.total)
    except Exception as e:
        logger.warning("OKX query failed: %s", e)

    # Pull trade outcomes from audit log (reconstructs pairs → per-trade PnL)
    try:
        from monitoring.daily_pnl_alert import _reconstruct_trades
        events = _read_audit_events("okx", since_ts)
        trades = _reconstruct_trades(events)
        cum_pnl = sum(t["pnl"] for t in trades)
        # Count rejected signals (log_gate with allowed=False) as red flags
        rejects_count = sum(
            1 for e in events
            if e.get("type") == "gate" and not e.get("allowed", True)
        )
    except Exception as e:
        logger.debug("audit reconstruction failed: %s", e)

    cum_pnl_pct = (cum_pnl / equity_usd) if equity_usd > 0 else 0.0

    return {
        "fills_count": fills_count,
        "rejects_count": rejects_count,
        "cum_pnl_usd": round(cum_pnl, 2),
        "cum_pnl_pct": round(cum_pnl_pct, 4),
        "equity_usd": round(equity_usd, 2),
        "since_ts": since_ts,
    }


def _should_promote(current: RampLevel, stats: dict[str, Any]) -> tuple[bool, str]:
    """Decide whether the current level can be promoted to the next."""
    next_level_idx = current.level + 1
    if next_level_idx >= len(RAMP_SCHEDULE):
        return False, "already at max level"
    nxt = RAMP_SCHEDULE[next_level_idx]

    if stats["fills_count"] < nxt.min_fills:
        return False, f"{stats['fills_count']} fills < required {nxt.min_fills}"
    if stats["rejects_count"] > 0:
        return False, f"{stats['rejects_count']} rejects blocks promotion"
    if stats["cum_pnl_pct"] < nxt.min_cum_pnl_pct:
        return False, (
            f"cum_pnl {stats['cum_pnl_pct']*100:.2f}% "
            f"< min {nxt.min_cum_pnl_pct*100:.2f}%"
        )
    return True, f"ready to promote to L{nxt.level} (${nxt.cap_usd:.0f})"


# ── Main ──────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Report only, no .env change")
    parser.add_argument("--force", action="store_true",
                        help="Force promote one level (manual override)")
    parser.add_argument("--show", action="store_true", help="Print state + exit")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    state = _load_state()
    current_idx = int(state.get("current_level", 0))
    current_level = RAMP_SCHEDULE[min(current_idx, len(RAMP_SCHEDULE) - 1)]

    if args.show:
        print(json.dumps(state, indent=2))
        return 0

    print(f"Current: L{current_level.level} (${current_level.cap_usd:.0f}) — {current_level.description}")

    stats = _compute_fill_stats(current_level)
    print(f"Stats (last {max(current_level.recent_window_hours, 72)}h):")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    if args.force:
        if current_idx + 1 >= len(RAMP_SCHEDULE):
            print("Already at max level, cannot force.")
            return 1
        nxt = RAMP_SCHEDULE[current_idx + 1]
        print(f"--force: overriding gates, promoting to L{nxt.level}")
        ok = True
        reason = "manual force"
    else:
        ok, reason = _should_promote(current_level, stats)
        if current_idx + 1 < len(RAMP_SCHEDULE):
            nxt = RAMP_SCHEDULE[current_idx + 1]
        else:
            nxt = None

    print(f"Decision: {'PROMOTE' if ok else 'HOLD'} — {reason}")

    if not ok or nxt is None:
        state["last_check_ts"] = time.time()
        _save_state(state)
        return 0

    if args.dry_run:
        print(f"[dry-run] would promote {current_level.cap_usd:.0f} → {nxt.cap_usd:.0f}")
        return 0

    # Promote
    _update_env_cap(nxt.cap_usd)
    state["current_level"] = nxt.level
    state["current_cap"] = nxt.cap_usd
    state.setdefault("history", []).append({
        "ts": time.time(),
        "from_cap": current_level.cap_usd,
        "to_cap": nxt.cap_usd,
        "reason": reason,
        "stats": stats,
    })
    state["last_check_ts"] = time.time()
    _save_state(state)

    sighup_ok = _sighup_okx_runner()
    print(f"PROMOTED to L{nxt.level} (${nxt.cap_usd:.0f})."
          f" SIGHUP {'sent' if sighup_ok else 'FAILED — manual restart required'}")

    # Telegram alert (best effort)
    try:
        from monitoring.notify import send_alert, AlertLevel
        send_alert(
            AlertLevel.INFO,
            f"OKX notional cap raised: ${current_level.cap_usd:.0f} → ${nxt.cap_usd:.0f}",
            details={
                "level": f"L{current_level.level} → L{nxt.level}",
                "reason": reason,
                "fills": str(stats["fills_count"]),
                "cum_pnl_pct": f"{stats['cum_pnl_pct']*100:.2f}%",
            },
            source="okx_ramp_up",
        )
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
