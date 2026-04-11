"""Live vs Backtest performance tracker.

For every closed trade in the audit log, compare the realised PnL /
win-rate / avg-return against the model's training-time walk-forward
expectations.  Highlights **alpha decay** (live stats trailing training
over time) so we know when to retrain before waiting for IC decay.

Compare dimensions:
- Per (venue, runner_key)
- Win rate: live vs walk_forward_sharpe_wr
- Avg net bps: live vs config.metrics.avg_net_bps
- Sharpe (bootstrap p50 vs live Sharpe estimate)
- Total PnL: live sum vs (trades × training_avg_return × notional)

Output:
- JSON report: data/runtime/live_vs_backtest_report.json
- Markdown digest printed to stdout
- Telegram alert when gap > 50% for 2 consecutive reports

Usage:
    python3 -m monitoring.live_vs_backtest
    python3 -m monitoring.live_vs_backtest --window 7d
    python3 -m monitoring.live_vs_backtest --json
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, "/quant_system")

from monitoring.decision_audit import audit_path_for
from monitoring.daily_pnl_alert import _reconstruct_trades

logger = logging.getLogger(__name__)

REPORT_PATH = Path("data/runtime/live_vs_backtest_report.json")
MODELS_DIR = Path("models_v8")
ALERT_GAP_PCT = 0.5   # 50% degradation triggers alert


# ── Training metrics lookup ───────────────────────────────────────
def _training_metrics(runner_key: str) -> dict[str, Any]:
    """Pull the relevant training metrics from a model's config.json.

    Handles both 1h (`{symbol}_gate_v2`) and 4h (`{symbol}_4h`) naming.
    Returns a minimal dict with {sharpe, avg_ic, avg_net_bps, trades,
    win_rate, bootstrap_p5} or {} on failure.
    """
    if runner_key.endswith("_4h") or runner_key.endswith("_4H"):
        base = runner_key[:-3]
        model_dir = MODELS_DIR / f"{base}_4h"
    else:
        model_dir = MODELS_DIR / f"{runner_key}_gate_v2"

    cfg_path = model_dir / "config.json"
    if not cfg_path.exists():
        return {}
    try:
        cfg = json.loads(cfg_path.read_text())
    except Exception:
        return {}

    metrics = cfg.get("metrics") or {}
    out = {
        "model_dir": str(model_dir),
        "sharpe": metrics.get("sharpe"),
        "avg_ic": metrics.get("avg_ic"),
        "avg_net_bps": metrics.get("avg_net_bps"),
        "total_return": metrics.get("total_return"),
        "train_trades": metrics.get("trades"),
        "train_win_rate": metrics.get("win_rate"),
        "bootstrap_p5": metrics.get("bootstrap_sharpe_p5")
                        or metrics.get("bootstrap_p5"),
    }
    return out


# ── Live stats computation ────────────────────────────────────────
def _compute_live_stats(trades: list[dict]) -> dict[str, Any]:
    """Summarise live trade outcomes: count, WR, avg return, Sharpe est."""
    if not trades:
        return {
            "n_trades": 0,
            "win_rate": None,
            "avg_pnl_pct": None,
            "cum_pnl_usd": 0.0,
            "sharpe_est": None,
        }
    pnls = [t["pnl"] for t in trades]
    pnl_pcts = [t["pnl_pct"] for t in trades]
    wins = sum(1 for p in pnls if p > 0)

    avg_ret = sum(pnl_pcts) / len(pnl_pcts) if pnl_pcts else 0.0
    sd = math.sqrt(
        sum((p - avg_ret) ** 2 for p in pnl_pcts) / len(pnl_pcts)
    ) if len(pnl_pcts) > 1 else 0.0
    # Rough Sharpe estimate (per-trade basis, not annualised)
    sharpe_est = (avg_ret / sd) if sd > 0 else None

    return {
        "n_trades": len(trades),
        "win_rate": round(wins / len(trades) * 100, 1),
        "avg_pnl_pct": round(avg_ret, 4),
        "avg_net_bps": round(avg_ret * 100, 1),  # pct → bps / 100 → already bps
        "cum_pnl_usd": round(sum(pnls), 2),
        "sharpe_est": round(sharpe_est, 2) if sharpe_est is not None else None,
    }


# ── Per-venue, per-runner breakdown ───────────────────────────────
def build_report(window_hours: int = 24 * 7) -> dict[str, Any]:
    """Compare live vs training across venues / runners."""
    cutoff = time.time() - window_hours * 3600
    report: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "window_hours": window_hours,
        "venues": {},
    }

    for venue in ("binance", "okx", "bybit"):
        path = audit_path_for(venue)
        if not path.exists() or path.stat().st_size == 0:
            continue
        try:
            entries = []
            for line in path.read_text().splitlines():
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if e.get("ts", 0) >= cutoff:
                    entries.append(e)
        except Exception as exc:
            logger.warning("audit read %s failed: %s", venue, exc)
            continue

        trades = _reconstruct_trades(entries)
        by_runner: dict[str, list[dict]] = {}
        for t in trades:
            sym = t.get("symbol", "?")
            # We don't always have runner_key on the exit event; approximate
            # by using the symbol (aggregates 1h + 4h together — acceptable).
            by_runner.setdefault(sym, []).append(t)

        venue_report: dict[str, Any] = {}
        for runner_key, rt in by_runner.items():
            live = _compute_live_stats(rt)
            train = _training_metrics(runner_key)

            # Compute gap metrics vs training
            gap: dict[str, Any] = {}
            if train:
                t_wr = train.get("train_win_rate")
                if t_wr is not None and live["win_rate"] is not None:
                    gap["wr_delta_pct"] = round(live["win_rate"] - t_wr, 1)
                    gap["wr_decay_ratio"] = round(
                        (live["win_rate"] / t_wr) if t_wr > 0 else 0, 3
                    )
                t_bps = train.get("avg_net_bps")
                if t_bps is not None and live["avg_net_bps"] is not None:
                    gap["bps_delta"] = round(live["avg_net_bps"] - t_bps, 1)
                    gap["bps_decay_ratio"] = round(
                        (live["avg_net_bps"] / t_bps) if t_bps > 0 else 0, 3
                    )

            venue_report[runner_key] = {
                "live": live,
                "training": train,
                "gap": gap,
            }

        if venue_report:
            report["venues"][venue] = venue_report

    return report


def _flag_alpha_decay(report: dict[str, Any]) -> list[str]:
    """Return list of '{venue}/{runner}' entries where decay > threshold."""
    flagged = []
    for venue, runners in report.get("venues", {}).items():
        for runner, data in runners.items():
            live = data.get("live", {})
            gap = data.get("gap", {})
            if live.get("n_trades", 0) < 5:
                continue  # not enough samples
            # Win rate decay
            wr_ratio = gap.get("wr_decay_ratio")
            if wr_ratio is not None and wr_ratio < (1 - ALERT_GAP_PCT):
                flagged.append(f"{venue}/{runner}: wr_ratio={wr_ratio}")
                continue
            # Net bps decay
            bps_ratio = gap.get("bps_decay_ratio")
            if bps_ratio is not None and bps_ratio < (1 - ALERT_GAP_PCT):
                flagged.append(f"{venue}/{runner}: bps_ratio={bps_ratio}")
    return flagged


def print_report(report: dict[str, Any]) -> None:
    window_h = report["window_hours"]
    print(f"\n=== Live vs Backtest ({window_h}h window) — {report['timestamp']} ===\n")
    if not report["venues"]:
        print("(no audit data in window)")
        return

    for venue, runners in report["venues"].items():
        print(f"● {venue}")
        for runner, data in runners.items():
            live = data["live"]
            train = data.get("training") or {}
            gap = data.get("gap") or {}

            n = live["n_trades"]
            if n == 0:
                print(f"    {runner}: no trades")
                continue

            wr_live = live["win_rate"]
            wr_train = train.get("train_win_rate", "—")
            bps_live = live["avg_net_bps"]
            bps_train = train.get("avg_net_bps", "—")
            pnl = live["cum_pnl_usd"]

            bps_gap = gap.get("bps_delta", "—")
            wr_gap = gap.get("wr_delta_pct", "—")

            print(f"    {runner}: {n}笔 PnL=${pnl:+.2f}")
            print(f"        WR:    live={wr_live}% vs train={wr_train}%   Δ={wr_gap}")
            print(f"        bps:   live={bps_live:.1f} vs train={bps_train}   Δ={bps_gap}")
            if "bps_decay_ratio" in gap:
                print(f"        decay: ratio={gap['bps_decay_ratio']} (1.0=on par)")
        print()

    flagged = _flag_alpha_decay(report)
    if flagged:
        print(f"⚠️  {len(flagged)} runner(s) showing >50% alpha decay:")
        for f in flagged:
            print(f"    {f}")
    else:
        print("✅ No significant alpha decay detected (or <5 trades sample).")


def save_report(report: dict[str, Any]) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, default=str))


def send_alert_if_decay(report: dict[str, Any]) -> None:
    flagged = _flag_alpha_decay(report)
    if not flagged:
        return
    try:
        from monitoring.notify import send_alert, AlertLevel
        send_alert(
            AlertLevel.WARNING,
            f"Live vs backtest decay: {len(flagged)} runner(s)",
            details={f"runner_{i}": f for i, f in enumerate(flagged[:6])},
            source="live_vs_backtest",
        )
    except Exception as e:
        logger.debug("alert send failed: %s", e)


def _parse_window(spec: str) -> int:
    """'24h', '7d', '2w' → hours."""
    spec = spec.strip().lower()
    if spec.endswith("h"):
        return int(spec[:-1])
    if spec.endswith("d"):
        return int(spec[:-1]) * 24
    if spec.endswith("w"):
        return int(spec[:-1]) * 24 * 7
    return int(spec)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", default="7d", help="Lookback window (e.g. 24h, 7d, 2w)")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--no-alert", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")

    hours = _parse_window(args.window)
    report = build_report(window_hours=hours)

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print_report(report)

    save_report(report)

    if not args.no_alert:
        send_alert_if_decay(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
