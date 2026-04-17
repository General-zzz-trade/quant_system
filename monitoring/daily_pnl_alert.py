"""Daily PnL summary — sends Telegram alert with today's trading performance.

Reads decision_audit.jsonl, reconstructs trade pairs, and sends a summary
of today's trades, PnL, win rate, and any exceptional events.

Runs via systemd timer (daily at 23:00 UTC).

Usage:
    python3 -m monitoring.daily_pnl_alert
    python3 -m monitoring.daily_pnl_alert --dry-run  # print to console only
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

from monitoring.notify import send_alert, AlertLevel
from monitoring.decision_audit import audit_path_for

logger = logging.getLogger(__name__)

AUDIT_DIR = Path("data/runtime")
KNOWN_VENUES = ("binance", "okx", "bybit")


def _audit_paths_available() -> list[tuple[str, Path]]:
    """Return list of (venue, path) for every per-venue audit log present.

    Falls back to the legacy single-file path when no per-venue files exist,
    tagging it as venue 'legacy'.
    """
    out: list[tuple[str, Path]] = []
    for v in KNOWN_VENUES:
        p = audit_path_for(v)
        if p.exists() and p.stat().st_size > 0:
            out.append((v, p))
    if not out:
        legacy = AUDIT_DIR / "decision_audit.jsonl"
        if legacy.exists():
            out.append(("legacy", legacy))
    return out


def _is_realistic_price(symbol: str, price: float) -> bool:
    """Filter out warmup artifact prices ($90000 BTC, $3000 ETH, etc)."""
    if price <= 0:
        return False
    if symbol == "BTCUSDT":
        return 10000 < price < 200000 and price != 90000
    if symbol == "ETHUSDT":
        return 500 < price < 10000 and price != 3000
    return True


def _reconstruct_trades(entries: list[dict]) -> list[dict]:
    """Pair entries with exits to compute per-trade PnL."""
    trades: list[dict] = []
    open_pos: dict[str, dict] = {}

    for e in entries:
        sym = e.get("symbol", "")
        price = float(e.get("price", 0))
        if not _is_realistic_price(sym, price):
            continue

        if e["type"] == "entry":
            open_pos[sym] = e
        elif e["type"] == "exit" and sym in open_pos:
            entry = open_pos.pop(sym)
            ep = float(entry["price"])
            xp = price
            side = entry.get("side", "buy")
            qty = float(entry.get("qty", 0))
            if side == "buy":
                pnl = (xp - ep) * qty
                pnl_pct = (xp - ep) / ep * 100
            else:
                pnl = (ep - xp) * qty
                pnl_pct = (ep - xp) / ep * 100
            trades.append({
                "symbol": sym,
                "side": side,
                "qty": qty,
                "entry_price": ep,
                "exit_price": xp,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "exit_reason": e.get("reason", ""),
                "entry_ts": entry["ts"],
                "exit_ts": e["ts"],
            })

    return trades


def _read_recent_entries(path: Path, cutoff_ts: float, tag_venue: str | None = None) -> list[dict]:
    """Read audit entries newer than cutoff. Optionally force a venue tag
    on records that don't already have one (legacy files)."""
    out: list[dict] = []
    try:
        for line in path.read_text().splitlines():
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("ts", 0) < cutoff_ts:
                continue
            if tag_venue and "venue" not in e:
                e["venue"] = tag_venue
            out.append(e)
    except Exception as exc:
        logger.debug("audit read failed for %s: %s", path, exc)
    return out


def _summarize(entries: list[dict]) -> dict[str, Any]:
    trades = _reconstruct_trades(entries)
    signals = [e for e in entries if e.get("type") == "signal"]

    if not trades:
        return {
            "n_trades": 0,
            "total_pnl": 0.0,
            "n_signals": len(signals),
            "note": "no trades in this window",
        }

    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] < 0]
    total_pnl = sum(t["pnl"] for t in trades)
    win_rate = len(wins) / len(trades) * 100 if trades else 0.0

    by_symbol: dict[str, dict] = {}
    for t in trades:
        sym = t["symbol"]
        if sym not in by_symbol:
            by_symbol[sym] = {"n": 0, "pnl": 0.0, "wins": 0}
        by_symbol[sym]["n"] += 1
        by_symbol[sym]["pnl"] += t["pnl"]
        if t["pnl"] > 0:
            by_symbol[sym]["wins"] += 1

    return {
        "n_trades": len(trades),
        "n_wins": len(wins),
        "n_losses": len(losses),
        "win_rate": round(win_rate, 1),
        "total_pnl": round(total_pnl, 2),
        "best": round(max((t["pnl"] for t in trades), default=0), 2),
        "worst": round(min((t["pnl"] for t in trades), default=0), 2),
        "n_signals": len(signals),
        "by_symbol": {k: {**v, "pnl": round(v["pnl"], 2)} for k, v in by_symbol.items()},
    }


def build_venue_summaries() -> dict[str, dict[str, Any]]:
    """Return {venue: summary} for every venue with audit data in last 24h."""
    paths = _audit_paths_available()
    if not paths:
        return {}
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)).timestamp()
    result: dict[str, dict[str, Any]] = {}
    for venue, path in paths:
        entries = _read_recent_entries(path, cutoff, tag_venue=venue)
        result[venue] = _summarize(entries)
    return result


def build_daily_summary() -> dict[str, Any]:
    """Merged cross-venue summary (back-compat for dashboard).

    Aggregates all venues into a single object but preserves per-venue
    breakdown under the `by_venue` key.
    """
    per_venue = build_venue_summaries()
    if not per_venue:
        return {"error": "no audit log"}

    # Merge all entries into a global summary for the top-level view
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)).timestamp()
    entries: list[dict] = []
    for venue, path in _audit_paths_available():
        entries.extend(_read_recent_entries(path, cutoff, tag_venue=venue))

    trades = _reconstruct_trades(entries)
    signals = [e for e in entries if e.get("type") == "signal"]

    if not trades:
        return {
            "n_trades": 0,
            "total_pnl": 0.0,
            "n_signals": len(signals),
            "note": "no trades in last 24h",
            "by_venue": per_venue,
        }

    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] < 0]
    total_pnl = sum(t["pnl"] for t in trades)
    win_rate = len(wins) / len(trades) * 100 if trades else 0.0

    by_symbol: dict[str, dict] = {}
    for t in trades:
        sym = t["symbol"]
        if sym not in by_symbol:
            by_symbol[sym] = {"n": 0, "pnl": 0.0, "wins": 0}
        by_symbol[sym]["n"] += 1
        by_symbol[sym]["pnl"] += t["pnl"]
        if t["pnl"] > 0:
            by_symbol[sym]["wins"] += 1

    return {
        "n_trades": len(trades),
        "n_wins": len(wins),
        "n_losses": len(losses),
        "win_rate": round(win_rate, 1),
        "total_pnl": round(total_pnl, 2),
        "best": round(max((t["pnl"] for t in trades), default=0), 2),
        "worst": round(min((t["pnl"] for t in trades), default=0), 2),
        "n_signals": len(signals),
        "by_symbol": {k: {**v, "pnl": round(v["pnl"], 2)} for k, v in by_symbol.items()},
        "by_venue": per_venue,
    }


def _tca_summary() -> dict[str, Any]:
    """Read TCA logs and compute execution quality metrics."""
    result: dict[str, Any] = {}
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)).timestamp()
    for venue in KNOWN_VENUES:
        tca_path = AUDIT_DIR / f"tca_{venue}.jsonl"
        if not tca_path.exists():
            continue
        fills = []
        try:
            for line in tca_path.read_text().splitlines():
                try:
                    d = json.loads(line)
                    if d.get("ts", 0) >= cutoff:
                        fills.append(d)
                except json.JSONDecodeError:
                    continue
        except Exception:
            continue
        if not fills:
            continue
        slips = [f["slippage_bps"] for f in fills if "slippage_bps" in f]
        lats = [f["latency_ms"] for f in fills if "latency_ms" in f]
        result[venue] = {
            "fills": len(fills),
            "avg_slip_bps": round(sum(slips) / len(slips), 1) if slips else 0,
            "max_slip_bps": round(max(slips), 1) if slips else 0,
            "avg_lat_ms": round(sum(lats) / len(lats), 0) if lats else 0,
        }
    return result


def send_daily_summary(dry_run: bool = False) -> None:
    summary = build_daily_summary()

    if "error" in summary:
        logger.warning("Cannot build summary: %s", summary["error"])
        return

    n_trades = summary.get("n_trades", 0)
    total_pnl = summary.get("total_pnl", 0.0)

    # Build alert message
    if n_trades == 0:
        title = "Daily summary: no trades"
        details = {"signals_evaluated": summary.get("n_signals", 0)}
        level = AlertLevel.INFO
    else:
        pnl_str = f"${total_pnl:+.2f}"
        title = f"Daily PnL: {pnl_str} ({n_trades} trades, {summary['win_rate']}% WR)"
        details = {
            "trades": f"{n_trades} ({summary['n_wins']}W/{summary['n_losses']}L)",
            "total_pnl": pnl_str,
            "best_trade": f"${summary['best']:+.2f}",
            "worst_trade": f"${summary['worst']:+.2f}",
        }
        for sym, stats in summary.get("by_symbol", {}).items():
            details[sym] = f"{stats['n']} trades, ${stats['pnl']:+.2f}"
        # Critical if losing day with >3 trades
        if total_pnl < -100 and n_trades >= 3:
            level = AlertLevel.CRITICAL
        elif total_pnl < 0:
            level = AlertLevel.WARNING
        else:
            level = AlertLevel.INFO

    # TCA execution quality
    tca = _tca_summary()
    for venue, metrics in tca.items():
        details[f"tca_{venue}"] = (
            f"{metrics['fills']} fills, "
            f"slip={metrics['avg_slip_bps']:.0f}bps avg/{metrics['max_slip_bps']:.0f}bps max, "
            f"lat={metrics['avg_lat_ms']:.0f}ms"
        )

    if dry_run:
        print(f"[{level.value}] {title}")
        for k, v in details.items():
            print(f"  {k}: {v}")
        return

    send_alert(level, title, details=details, source="daily_pnl")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    send_daily_summary(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
