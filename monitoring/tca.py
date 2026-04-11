"""Transaction Cost Analysis (TCA) — per-fill slippage and latency logger.

Every execution adapter (bybit / binance / okx) hands each fill through
``TCALogger.record_fill()`` which captures:

* ``ref_price``  — best-known mark/last price at order-send time (pre-fill)
* ``fill_price`` — actual execution price reported by the exchange
* ``latency_ms`` — wall-clock time between send_order and fill confirmation

Results land in ``data/runtime/tca_{venue}.jsonl`` (JSON per line, append-only).

Downstream:
  * ``monitoring/daily_pnl_alert.py`` loads recent TCA rows and reports a
    per-venue slippage-bps median in the daily telegram.
  * ``monitoring/prometheus_exporter.py`` exposes
    ``quant_slippage_bps_median{venue,symbol}`` for Grafana.
  * A quick CLI (``python3 -m monitoring.tca``) prints a 24h summary.

Design notes:
  * Pure stdlib — no pandas/numpy.  Append-only JSONL, one line per fill,
    so the hot path stays sub-millisecond even with lock contention.
  * Fail-open: any write failure is logged at DEBUG and swallowed.  TCA is
    observability, not trading — it must never block a live order.
  * Test isolation: mirrors ``monitoring/decision_audit.py`` — pytest runs
    redirect to ``/tmp/test_tca_{venue}.jsonl`` so unit tests never touch
    the production audit.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

TCA_DIR = Path("/quant_system/data/runtime")


def _in_pytest() -> bool:
    return "pytest" in sys.modules or bool(os.environ.get("PYTEST_CURRENT_TEST"))


def tca_path_for(venue: str) -> Path:
    """Return the per-venue TCA log path (or /tmp under pytest)."""
    venue = (venue or "unknown").lower()
    if _in_pytest():
        return Path("/tmp") / f"test_tca_{venue}.jsonl"
    return TCA_DIR / f"tca_{venue}.jsonl"


class TCALogger:
    """Append-only JSONL logger for per-fill transaction costs.

    One instance per (venue, process).  Thread-safety: CPython GIL protects
    append-write to a single file handle in practice; for multi-writer
    scenarios the file is opened in ``"a"`` mode with OS-level atomic writes
    for records < 4KB (JSON is always well under).
    """

    def __init__(self, venue: str, path: Path | None = None):
        self._venue = (venue or "unknown").lower()
        self._path = path or tca_path_for(self._venue)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = None

    def _ensure_open(self) -> None:
        if self._file is None or self._file.closed:
            self._file = open(self._path, "a")

    def record_fill(
        self,
        *,
        symbol: str,
        side: str,
        qty: float,
        ref_price: float,
        fill_price: float,
        latency_ms: float,
        order_id: str | None = None,
        fill_id: str | None = None,
        **extra: Any,
    ) -> None:
        """Write one TCA record.

        Slippage convention:
          * Long fills pay at ``fill_price ≥ ref_price`` → positive slippage
            is an adverse cost (we paid more than mark).
          * Short fills earn at ``fill_price ≤ ref_price`` → positive
            slippage ``(ref - fill) / ref`` is an adverse cost.

        ``slippage_bps`` is always a *cost* (positive = worse for us).
        """
        try:
            if ref_price <= 0 or fill_price <= 0:
                raw_bps = 0.0
            else:
                side_lc = (side or "").lower()
                if side_lc in ("buy", "long"):
                    raw_bps = (fill_price - ref_price) / ref_price * 10_000.0
                elif side_lc in ("sell", "short"):
                    raw_bps = (ref_price - fill_price) / ref_price * 10_000.0
                else:
                    raw_bps = 0.0
            record = {
                "ts": time.time(),
                "venue": self._venue,
                "symbol": symbol,
                "side": side,
                "qty": float(qty),
                "ref_price": float(ref_price),
                "fill_price": float(fill_price),
                "slippage_bps": round(raw_bps, 2),
                "latency_ms": round(float(latency_ms), 1),
            }
            if order_id:
                record["order_id"] = str(order_id)
            if fill_id:
                record["fill_id"] = str(fill_id)
            if extra:
                record.update(extra)
            self._ensure_open()
            self._file.write(json.dumps(record, default=str) + "\n")
            self._file.flush()
        except Exception:
            logger.debug("TCA write failed", exc_info=True)

    def close(self) -> None:
        if self._file and not self._file.closed:
            self._file.close()


# ── Aggregation helpers ─────────────────────────────────────────────


def load_recent_tca(venue: str, since_ts: float) -> list[dict]:
    """Load TCA rows newer than ``since_ts`` from ``tca_{venue}.jsonl``."""
    path = tca_path_for(venue)
    if not path.exists():
        return []
    out: list[dict] = []
    try:
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("ts", 0) >= since_ts:
                out.append(row)
    except Exception:
        logger.debug("TCA read failed for %s", path, exc_info=True)
    return out


def slippage_summary(venue: str, lookback_hours: float = 24.0) -> dict[str, Any]:
    """Return per-symbol median and 90th-percentile slippage (bps) + fill count.

    ``lookback_hours`` rolls the window over which fills are considered.
    Returns ``{"symbols": {SYM: {"count": ..., "median_bps": ..., "p90_bps": ...}},
    "total_count": ..., "overall_median_bps": ...}``.
    """
    since = time.time() - lookback_hours * 3600
    rows = load_recent_tca(venue, since)
    per_sym: dict[str, list[float]] = {}
    per_sym_lat: dict[str, list[float]] = {}
    for r in rows:
        sym = r.get("symbol", "unknown")
        per_sym.setdefault(sym, []).append(float(r.get("slippage_bps", 0.0)))
        per_sym_lat.setdefault(sym, []).append(float(r.get("latency_ms", 0.0)))

    def _median(xs: list[float]) -> float:
        if not xs:
            return 0.0
        s = sorted(xs)
        n = len(s)
        mid = n // 2
        return s[mid] if n % 2 else 0.5 * (s[mid - 1] + s[mid])

    def _pct(xs: list[float], p: float) -> float:
        if not xs:
            return 0.0
        s = sorted(xs)
        k = int(round(p * (len(s) - 1)))
        return s[max(0, min(k, len(s) - 1))]

    symbols: dict[str, dict[str, float]] = {}
    all_bps: list[float] = []
    for sym, bps_list in per_sym.items():
        all_bps.extend(bps_list)
        symbols[sym] = {
            "count": len(bps_list),
            "median_bps": round(_median(bps_list), 2),
            "p90_bps": round(_pct(bps_list, 0.90), 2),
            "median_latency_ms": round(_median(per_sym_lat.get(sym, [])), 1),
        }
    return {
        "venue": venue,
        "lookback_hours": lookback_hours,
        "total_count": len(all_bps),
        "overall_median_bps": round(_median(all_bps), 2),
        "symbols": symbols,
    }


# ── CLI ────────────────────────────────────────────────────────────

def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--venue", default=None,
                        help="One venue (bybit|binance|okx) or all if omitted")
    parser.add_argument("--hours", type=float, default=24.0,
                        help="Lookback window in hours (default 24)")
    parser.add_argument("--json", action="store_true",
                        help="Output JSON instead of human-readable table")
    args = parser.parse_args()

    venues = [args.venue] if args.venue else ["bybit", "binance", "okx"]
    summaries = {v: slippage_summary(v, args.hours) for v in venues}

    if args.json:
        print(json.dumps(summaries, indent=2))
        return 0

    for venue, s in summaries.items():
        if s["total_count"] == 0:
            print(f"[{venue}] no fills in last {args.hours:.0f}h")
            continue
        print(f"[{venue}] {s['total_count']} fills in last {args.hours:.0f}h "
              f"— overall median slippage = {s['overall_median_bps']:+.1f} bps")
        for sym, st in s["symbols"].items():
            print(f"    {sym:10s} n={st['count']:4d} "
                  f"median={st['median_bps']:+.1f}bps p90={st['p90_bps']:+.1f}bps "
                  f"lat={st['median_latency_ms']:.0f}ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
