"""Live IC kill switch — auto-pause runners when live IC goes bad.

Reads the decision audit (live z-score signals) and realized bar returns,
computes rolling live IC per runner, and writes a pause state file that
``runner.alpha_main`` checks before emitting each bar.

Trigger logic (per runner_key):
  - PAUSE   if rolling 30-day Spearman IC < -0.10 for 3 consecutive daily checks
  - RESUME  if (already paused) rolling IC > +0.05 for 3 consecutive daily checks
  - HOLD    otherwise (current state preserved)

State file: data/runtime/symbol_pause_state.json
  {
    "BTCUSDT":    {"paused": false, "rolling_ic_30d": 0.18, "last_check": "..."},
    "ETHUSDT":    {"paused": true,  "rolling_ic_30d": -0.15,
                   "reason": "3 days IC < -0.10", "since": "..."},
    ...
  }

Usage:
  python3 -m monitoring.live_ic_killswitch              # print status
  python3 -m monitoring.live_ic_killswitch --update     # write state
  python3 -m monitoring.live_ic_killswitch --alert      # + Telegram on state change
  python3 -m monitoring.live_ic_killswitch --force-resume BTCUSDT  # manual override

Intended to be run hourly via systemd timer live-ic-killswitch.timer.
The runner reads the state file at bar-emit time (nearly free), so no
realtime coupling needed.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


AUDIT_PATHS = [
    Path("data/runtime/decision_audit_okx.jsonl"),
    Path("data/runtime/decision_audit_okx.jsonl.1"),
]
STATE_PATH = Path("data/runtime/symbol_pause_state.json")
HISTORY_PATH = Path("data/runtime/live_ic_history.json")
KLINE_DIR = Path("data_files")

# Per-runner config:
#   forward_bars: bars ahead to measure realized return for IC
#   active: if False, runner is monitored but never paused. Used for
#     signal-only runners like the 4h direction filter — pausing them
#     would silently break the 1h gating logic.
RUNNER_CONFIG = {
    "BTCUSDT":     {"forward_bars": 1, "active": True},
    "ETHUSDT":     {"forward_bars": 1, "active": True},
    "BTCUSDT_4h":  {"forward_bars": 1, "active": False},  # signal-only
    "ETHUSDT_4h":  {"forward_bars": 1, "active": False},  # signal-only
    "BTCUSDT_1d":  {"forward_bars": 1, "active": True},
    "ETHUSDT_1d":  {"forward_bars": 1, "active": True},
}

# IC trigger thresholds
PAUSE_THRESHOLD = -0.10
RESUME_THRESHOLD = 0.05
CONSECUTIVE_DAYS_REQUIRED = 3
ROLLING_WINDOW_DAYS = 30


def _iter_audit_records():
    for p in AUDIT_PATHS:
        if not p.exists():
            continue
        with p.open() as f:
            for line in f:
                try:
                    yield json.loads(line)
                except Exception:
                    continue


def _load_live_signals() -> dict[str, list[tuple[float, float, float]]]:
    """Return {runner_key: [(ts, z_score, close)...]} from audit."""
    by_runner: dict[str, list[tuple[float, float, float]]] = {}
    for r in _iter_audit_records():
        if r.get("type") != "signal":
            continue
        rk = r.get("runner_key")
        if rk not in RUNNER_CONFIG:
            continue
        ts = r.get("ts")
        z = r.get("z_score")
        close = (r.get("top_features") or {}).get("close")
        if ts is None or z is None or close is None:
            continue
        if not isinstance(close, (int, float)) or close <= 0:
            continue
        by_runner.setdefault(rk, []).append((float(ts), float(z), float(close)))
    # sort by ts
    for rk in by_runner:
        by_runner[rk].sort(key=lambda t: t[0])
    return by_runner


def _spearman_ic(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 10:
        return float("nan")
    from scipy.stats import spearmanr

    ic, _ = spearmanr(x[mask], y[mask])
    return float(ic) if np.isfinite(ic) else float("nan")


def _compute_ic_windows(
    signals: list[tuple[float, float, float]], forward_bars: int,
) -> dict:
    """Compute rolling 30d IC + per-day series for consecutive-day check."""
    if len(signals) < 30:
        return {"ic_30d": None, "ic_per_day": {}, "n_points": len(signals)}

    ts = np.array([s[0] for s in signals])
    z = np.array([s[1] for s in signals])
    close = np.array([s[2] for s in signals])

    # Forward return: log(close[i+H] / close[i])
    fwd = np.full(len(close), np.nan)
    if len(close) > forward_bars:
        fwd[:-forward_bars] = np.log(close[forward_bars:] / close[:-forward_bars])

    # Spearman IC over the last 30 days of signals
    cutoff_30d = ts[-1] - ROLLING_WINDOW_DAYS * 86400
    mask_30d = ts >= cutoff_30d
    ic_30d = _spearman_ic(z[mask_30d], fwd[mask_30d])

    # Daily IC: group signals by UTC date, compute IC per day
    ic_per_day: dict[str, float] = {}
    dates = [datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%d") for t in ts]
    for day in sorted(set(dates)):
        day_mask = np.array([d == day for d in dates])
        if day_mask.sum() >= 6:
            ic_day = _spearman_ic(z[day_mask], fwd[day_mask])
            if np.isfinite(ic_day):
                ic_per_day[day] = float(ic_day)

    return {
        "ic_30d": float(ic_30d) if np.isfinite(ic_30d) else None,
        "ic_per_day": ic_per_day,
        "n_points": int(mask_30d.sum()),
    }


def _decide_action(prev_paused: bool, ic_per_day: dict[str, float]) -> tuple[str, str]:
    """Return (action, reason). action ∈ {'PAUSE','RESUME','HOLD'}."""
    if not ic_per_day:
        return "HOLD", "not enough daily IC data"

    # Check last N days (sorted ascending dates)
    days = sorted(ic_per_day.keys())
    last_n = days[-CONSECUTIVE_DAYS_REQUIRED:]
    if len(last_n) < CONSECUTIVE_DAYS_REQUIRED:
        return "HOLD", f"only {len(last_n)} daily IC points"

    last_ics = [ic_per_day[d] for d in last_n]

    if not prev_paused and all(ic < PAUSE_THRESHOLD for ic in last_ics):
        return "PAUSE", (
            f"IC < {PAUSE_THRESHOLD} for {CONSECUTIVE_DAYS_REQUIRED} days: "
            + ", ".join(f"{d}={ic:.3f}" for d, ic in zip(last_n, last_ics))
        )

    if prev_paused and all(ic > RESUME_THRESHOLD for ic in last_ics):
        return "RESUME", (
            f"IC > {RESUME_THRESHOLD} for {CONSECUTIVE_DAYS_REQUIRED} days: "
            + ", ".join(f"{d}={ic:.3f}" for d, ic in zip(last_n, last_ics))
        )

    return "HOLD", ""


def _load_state() -> dict:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text())
        except Exception:
            return {}
    return {}


def _save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2, sort_keys=True))


def _load_history() -> dict:
    if HISTORY_PATH.exists():
        try:
            return json.loads(HISTORY_PATH.read_text())
        except Exception:
            return {}
    return {}


def _save_history(hist: dict) -> None:
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    HISTORY_PATH.write_text(json.dumps(hist, indent=2, sort_keys=True))


def is_runner_paused(runner_key: str) -> bool:
    """Used by runner.alpha_main to check state. Returns False if no state."""
    try:
        if not STATE_PATH.exists():
            return False
        state = json.loads(STATE_PATH.read_text())
        return bool(state.get(runner_key, {}).get("paused", False))
    except Exception:
        return False


def run_check(update: bool = False, alert: bool = False) -> dict:
    by_runner = _load_live_signals()
    prev_state = _load_state()
    hist = _load_history()
    now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")

    new_state: dict[str, dict[str, Any]] = {}
    state_changes: list[str] = []

    for rk, signals in sorted(by_runner.items()):
        rc = RUNNER_CONFIG.get(rk, {"forward_bars": 1, "active": True})
        forward_bars = rc["forward_bars"]
        active = rc["active"]
        windows = _compute_ic_windows(signals, forward_bars)
        ic_30d = windows["ic_30d"]
        n_points = windows["n_points"]
        ic_per_day = windows["ic_per_day"]

        prev = prev_state.get(rk, {"paused": False})
        if active:
            action, reason = _decide_action(prev.get("paused", False), ic_per_day)
        else:
            action, reason = "MONITOR_ONLY", "signal-only runner"

        entry: dict[str, Any] = {
            "paused": prev.get("paused", False),
            "rolling_ic_30d": ic_30d,
            "n_signals_30d": n_points,
            "active": active,
            "last_check": now_iso,
        }

        if action == "PAUSE":
            entry["paused"] = True
            entry["reason"] = reason
            entry["since"] = now_iso
            state_changes.append(f"PAUSE {rk}: {reason}")
        elif action == "RESUME":
            entry["paused"] = False
            entry["resumed_at"] = now_iso
            entry["reason"] = reason
            entry.pop("since", None)
            state_changes.append(f"RESUME {rk}: {reason}")
        else:
            # preserve prior metadata
            for k in ("reason", "since", "resumed_at"):
                if k in prev:
                    entry[k] = prev[k]

        new_state[rk] = entry

        # Append to history
        hist.setdefault(rk, []).append({
            "ts": now_iso,
            "ic_30d": ic_30d,
            "n_points": n_points,
            "paused": entry["paused"],
            "action": action,
        })
        # keep last 60 history rows per runner
        hist[rk] = hist[rk][-60:]

    if update:
        _save_state(new_state)
        _save_history(hist)

    # Print summary
    print("=" * 72)
    print(f"Live IC kill-switch check @ {now_iso}")
    print("=" * 72)
    print(f"{'runner_key':<14s} {'mode':>10s} {'paused':>7s} "
          f"{'IC_30d':>8s} {'n_30d':>6s}  days_ic")
    print("-" * 72)
    for rk, e in new_state.items():
        ic = e.get("rolling_ic_30d")
        ic_str = f"{ic:+.3f}" if ic is not None else "  n/a"
        mode = "active" if e.get("active", True) else "monitor"
        daily_ic = hist.get(rk, [])
        recent = [f"{h['ic_30d']:+.2f}" if h.get("ic_30d") is not None else "n/a"
                  for h in daily_ic[-5:]]
        print(f"{rk:<14s} {mode:>10s} {'Y' if e['paused'] else 'N':>7s} "
              f"{ic_str:>8s} {e.get('n_signals_30d', 0):>6d}  {', '.join(recent)}")

    if state_changes:
        print()
        print("STATE CHANGES:")
        for c in state_changes:
            print(f"  {c}")

    if alert and state_changes:
        try:
            from monitoring.telegram_alert import send_alert, AlertLevel
            for c in state_changes:
                level = AlertLevel.CRITICAL if c.startswith("PAUSE") else AlertLevel.INFO
                title = c.split(":", 1)[0]
                details = {"reason": c.split(":", 1)[1].strip() if ":" in c else ""}
                send_alert(level, title, details=details, source="ic_killswitch")
        except Exception as e:
            logger.warning("Telegram alert failed: %s", e)

    return {"state": new_state, "changes": state_changes}


def force_override(runner_key: str, paused: bool, reason: str = "manual") -> None:
    state = _load_state()
    state.setdefault(runner_key, {})
    state[runner_key]["paused"] = paused
    state[runner_key]["reason"] = reason
    state[runner_key]["last_check"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if paused:
        state[runner_key]["since"] = state[runner_key]["last_check"]
    else:
        state[runner_key]["resumed_at"] = state[runner_key]["last_check"]
    _save_state(state)
    print(f"{'PAUSED' if paused else 'RESUMED'} {runner_key}: {reason}")


def main():
    parser = argparse.ArgumentParser(description="Live IC kill switch")
    parser.add_argument("--update", action="store_true",
                        help="Write updated state to disk")
    parser.add_argument("--alert", action="store_true",
                        help="Send Telegram alert on state changes")
    parser.add_argument("--force-pause", metavar="RUNNER",
                        help="Manually pause a runner")
    parser.add_argument("--force-resume", metavar="RUNNER",
                        help="Manually resume a runner")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    if args.force_pause:
        force_override(args.force_pause, True, "manual force-pause")
        return
    if args.force_resume:
        force_override(args.force_resume, False, "manual force-resume")
        return

    result = run_check(update=args.update, alert=args.alert)
    sys.exit(0 if not result["changes"] else 2)


if __name__ == "__main__":
    main()
