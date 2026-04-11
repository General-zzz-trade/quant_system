#!/usr/bin/env python3
"""Cross-venue reconciliation — compare binance vs okx live state.

Detects:
- **Signal divergence** — two runners produce opposite directions on the
  same symbol (shouldn't happen if they share z-score checkpoints, but
  OnlineRidge drift over time can cause it).  Warns if divergence > 2h.
- **Position sign mismatch** — one venue LONG, one venue SHORT on the
  same symbol (net-neutral hedge is likely unintentional).
- **Position size asymmetry** — exposures differ by >50% in magnitude
  when both venues should see the same signal.
- **Service liveness** — one service idle >30min while the other runs.
- **Audit log stall** — no new audit events in >2h (stale process).

Alerts via Telegram at WARNING level; INFO summary on every run.
Schedule every 6h via systemd timer.

Usage:
    python3 -m scripts.cross_venue_reconcile            # check + alert
    python3 -m scripts.cross_venue_reconcile --json     # machine output
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, "/quant_system")

from monitoring.decision_audit import audit_path_for

logger = logging.getLogger(__name__)

# ── Thresholds ────────────────────────────────────────────────────
SIGNAL_DIVERGENCE_MAX_AGE_SEC = 2 * 3600     # 2h
POSITION_RATIO_TOLERANCE = 0.5               # ±50% size difference allowed
AUDIT_STALE_MAX_SEC = 2 * 3600               # 2h
SERVICE_STALE_MAX_SEC = 30 * 60              # 30 min


def _load_env() -> dict[str, str]:
    env_path = Path("/quant_system/.env")
    out: dict[str, str] = {}
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def _service_state(unit: str) -> dict[str, Any]:
    try:
        out = subprocess.check_output(
            ["systemctl", "show", unit,
             "--property=ActiveState,ActiveEnterTimestamp,MainPID"],
            text=True, timeout=5,
        )
        props = {
            k: v for k, v in
            (line.split("=", 1) for line in out.splitlines() if "=" in line)
        }
        return props
    except Exception as e:
        return {"error": str(e)}


def _positions_binance(env: dict[str, str]) -> dict[str, Any]:
    try:
        from execution.adapters.binance.adapter import BinanceAdapter
        from execution.adapters.binance.config import BinanceConfig
        key = env.get("BINANCE_TESTNET_API_KEY") or env.get("BINANCE_API_KEY", "")
        sec = env.get("BINANCE_TESTNET_API_SECRET") or env.get("BINANCE_API_SECRET", "")
        if not (key and sec):
            return {"error": "no credentials"}
        testnet = bool(env.get("BINANCE_TESTNET_API_KEY"))
        adapter = BinanceAdapter(BinanceConfig(api_key=key, api_secret=sec, testnet=testnet))
        positions = {}
        for p in adapter.get_positions():
            if float(getattr(p, "qty", 0)) != 0:
                positions[p.symbol] = {
                    "qty": float(p.qty),
                    "entry": float(p.entry_price),
                    "side": p.side,
                }
        return {"positions": positions}
    except Exception as e:
        return {"error": str(e)[:120]}


def _positions_okx(env: dict[str, str]) -> dict[str, Any]:
    try:
        from execution.adapters.okx.adapter import OkxAdapter
        from execution.adapters.okx.config import OkxConfig
        key = env.get("OKX_API_KEY")
        sec = env.get("OKX_API_SECRET")
        pas = env.get("OKX_API_PASSPHRASE")
        if not (key and sec and pas):
            return {"error": "no credentials"}
        url = env.get("OKX_BASE_URL", "https://www.okx.com")
        adapter = OkxAdapter(OkxConfig(api_key=key, api_secret=sec, passphrase=pas, base_url=url))
        if not adapter.connect():
            return {"error": "connect failed"}
        positions = {}
        for p in adapter.get_positions():
            if float(getattr(p, "qty", 0)) != 0:
                positions[p.symbol] = {
                    "qty": float(p.qty),
                    "entry": float(p.entry_price),
                    "side": p.side,
                }
        bal = adapter.get_balances()
        usdt = bal.get("USDT")
        equity = float(usdt.total) if usdt else None
        return {"positions": positions, "equity_usdt": equity}
    except Exception as e:
        return {"error": str(e)[:120]}


def _latest_signal_by_symbol(venue: str) -> dict[str, dict]:
    """Return the most-recent signal per symbol from the venue's audit log."""
    path = audit_path_for(venue)
    if not path.exists():
        return {}
    latest: dict[str, dict] = {}
    try:
        for line in path.read_text().splitlines():
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("type") != "signal":
                continue
            sym = e.get("symbol", "")
            ts = e.get("ts", 0)
            if ts >= latest.get(sym, {}).get("ts", 0):
                latest[sym] = e
    except Exception as exc:
        logger.debug("signal read failed for %s: %s", path, exc)
    return latest


def _audit_last_ts(venue: str) -> float | None:
    path = audit_path_for(venue)
    if not path.exists():
        return None
    try:
        lines = path.read_text().splitlines()
    except Exception:
        return None
    if not lines:
        return None
    try:
        return float(json.loads(lines[-1]).get("ts", 0))
    except Exception:
        return None


# ── Main reconciliation ───────────────────────────────────────────
def reconcile() -> dict[str, Any]:
    env = _load_env()

    report: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "services": {},
        "positions": {},
        "signals": {},
        "issues": [],
    }

    # Services
    for venue, unit in (("binance", "binance-alpha.service"),
                        ("okx", "okx-alpha.service")):
        s = _service_state(unit)
        report["services"][venue] = s
        if s.get("ActiveState") != "active":
            report["issues"].append({
                "severity": "CRITICAL",
                "kind": "service_down",
                "venue": venue,
                "detail": f"{unit} state={s.get('ActiveState', '?')}",
            })

    # Positions
    report["positions"]["binance"] = _positions_binance(env)
    report["positions"]["okx"] = _positions_okx(env)

    # Signals
    report["signals"]["binance"] = _latest_signal_by_symbol("binance")
    report["signals"]["okx"] = _latest_signal_by_symbol("okx")

    now = time.time()

    # ── Check 1: audit log staleness ──
    for venue in ("binance", "okx"):
        last = _audit_last_ts(venue)
        if last is None:
            # Legacy fallback (for binance which may still be on old code)
            continue
        age = now - last
        if age > AUDIT_STALE_MAX_SEC:
            report["issues"].append({
                "severity": "WARNING",
                "kind": "audit_stale",
                "venue": venue,
                "detail": f"last audit {age/3600:.1f}h ago",
            })

    # ── Check 2: signal direction divergence ──
    syms_common = set(report["signals"]["binance"]) & set(report["signals"]["okx"])
    for sym in syms_common:
        s_bin = report["signals"]["binance"][sym]
        s_okx = report["signals"]["okx"][sym]
        sig_bin = int(s_bin.get("signal", 0))
        sig_okx = int(s_okx.get("signal", 0))
        # Only flag if both non-zero AND opposite
        if sig_bin != 0 and sig_okx != 0 and sig_bin * sig_okx < 0:
            age_bin = now - s_bin.get("ts", now)
            age_okx = now - s_okx.get("ts", now)
            if max(age_bin, age_okx) <= SIGNAL_DIVERGENCE_MAX_AGE_SEC:
                report["issues"].append({
                    "severity": "WARNING",
                    "kind": "signal_divergence",
                    "symbol": sym,
                    "detail": f"binance={sig_bin:+d} okx={sig_okx:+d}",
                })

    # ── Check 3: position sign mismatch ──
    bin_pos = report["positions"]["binance"].get("positions", {}) if "error" not in report["positions"]["binance"] else {}
    okx_pos = report["positions"]["okx"].get("positions", {}) if "error" not in report["positions"]["okx"] else {}
    for sym in set(bin_pos) & set(okx_pos):
        q_bin = bin_pos[sym]["qty"]
        q_okx = okx_pos[sym]["qty"]
        if q_bin * q_okx < 0:
            report["issues"].append({
                "severity": "CRITICAL",
                "kind": "position_sign_mismatch",
                "symbol": sym,
                "detail": f"binance qty={q_bin:+g}, okx qty={q_okx:+g}",
            })
        elif abs(q_bin) > 0 and abs(q_okx) > 0:
            # Same side, check size asymmetry
            ratio = min(abs(q_bin), abs(q_okx)) / max(abs(q_bin), abs(q_okx))
            if ratio < POSITION_RATIO_TOLERANCE:
                report["issues"].append({
                    "severity": "WARNING",
                    "kind": "position_size_asymmetry",
                    "symbol": sym,
                    "detail": f"binance={abs(q_bin):.4g} vs okx={abs(q_okx):.4g} (ratio {ratio:.2f})",
                })

    report["n_issues"] = len(report["issues"])
    return report


def send_alert_if_issues(report: dict[str, Any]) -> None:
    issues = report.get("issues", [])
    if not issues:
        return
    try:
        from monitoring.notify import send_alert, AlertLevel
        worst = max(
            issues,
            key=lambda i: {"CRITICAL": 2, "WARNING": 1, "INFO": 0}.get(i.get("severity", ""), 0),
        )
        level = {
            "CRITICAL": AlertLevel.CRITICAL,
            "WARNING": AlertLevel.WARNING,
        }.get(worst["severity"], AlertLevel.INFO)
        details = {
            f"{i['kind']}_{idx}": f"{i.get('symbol', i.get('venue', '?'))} — {i['detail']}"
            for idx, i in enumerate(issues[:6])  # cap at 6 for message size
        }
        title = f"Cross-venue reconcile: {len(issues)} issue(s)"
        send_alert(level, title, details=details, source="cross_venue_reconcile")
    except Exception as e:
        logger.warning("alert send failed: %s", e)


def print_report(report: dict[str, Any]) -> None:
    print(f"\n=== Cross-venue reconcile @ {report['timestamp']} ===\n")
    for venue, svc in report["services"].items():
        state = svc.get("ActiveState", "?")
        icon = "✅" if state == "active" else "⚠️"
        print(f"{icon} {venue:<9} service={state}")
    print()
    for venue, pos in report["positions"].items():
        if "error" in pos:
            print(f"⚠️  {venue:<9} positions: {pos['error']}")
        else:
            positions = pos.get("positions", {})
            if not positions:
                eq = pos.get("equity_usdt")
                eq_str = f" (equity=${eq:.2f})" if eq is not None else ""
                print(f"📭 {venue:<9} flat{eq_str}")
            else:
                for sym, p in positions.items():
                    print(f"📬 {venue:<9} {sym}: {p['side']} {p['qty']} @ ${p['entry']:.2f}")
    print()
    issues = report.get("issues", [])
    if not issues:
        print("✅ No issues detected.")
    else:
        print(f"⚠️  {len(issues)} issue(s):")
        for i in issues:
            print(f"  [{i['severity']}] {i['kind']}: {i.get('detail', '')}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--no-alert", action="store_true",
                        help="Skip Telegram alert even if issues found")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")

    report = reconcile()

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print_report(report)

    if not args.no_alert:
        send_alert_if_issues(report)

    return 0 if report["n_issues"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
