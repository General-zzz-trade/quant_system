"""TCA-driven deadzone calibrator.

Reads ``data/runtime/tca_{venue}.jsonl`` written by the execution adapters
and suggests a new deadzone value based on the **real round-trip cost**
observed on live fills.  Never auto-applies — prints a suggestion that
the operator can review and paste into ``strategy/config.py`` or a
retrain command.

Rationale
---------
Deadzones were fixed at ``deadzone = 1.2`` three months ago assuming a
flat 6 bps cost model.  If real OKX slippage + fees is 20 bps/side = 40
bps round trip, then the minimum z-score that still has positive net
expected value is **higher** than 1.2.  Formally, for a signal with
``avg_net_bps = N`` at the training deadzone:

    net_edge(dz) = N × (cost_train / cost_real)

A trade is only worth taking if its expected edge covers the real cost.
Heuristic: scale the deadzone linearly with the cost ratio, bounded to a
sane range.

    new_dz = 1.2 × (1 + (real_cost_bps - 6) / 30)

Usage
-----
::

    python3 -m monitoring.deadzone_calibrator                  # all venues
    python3 -m monitoring.deadzone_calibrator --venue okx      # single venue
    python3 -m monitoring.deadzone_calibrator --hours 168      # wider window
    python3 -m monitoring.deadzone_calibrator --json           # machine-readable

If the venue has fewer than ``--min-fills`` records, the tool prints a
warning and exits cleanly — will not suggest changes based on a tiny
sample.

Output format
-------------
Human-readable table by default, per-symbol recommendation.  ``--json``
emits the full suggestion dict for downstream tooling.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

sys.path.insert(0, "/quant_system")

from monitoring.tca import load_recent_tca


# ── Configuration ───────────────────────────────────────────────────
TRAINING_COST_BPS_ASSUMED = 6.0       # Backtest baseline (legacy)
BASELINE_DEADZONE = 1.2               # Current production deadzone
MIN_DEADZONE = 0.6                    # Floor — never below this
MAX_DEADZONE = 3.0                    # Ceiling — above = model is dead
MIN_FILLS_FOR_SUGGESTION = 10         # Below this, refuse to suggest


def _median(xs: list[float]) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else 0.5 * (s[mid - 1] + s[mid])


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def compute_real_cost_bps(rows: list[dict], symbol: str | None = None) -> dict:
    """Summarise real fill cost from TCA records.

    Real round-trip cost ≈ ``2 × (fee + |median slippage|)``.

    Fees are NOT in the TCA log (they're deducted post-fill by the
    exchange).  We assume OKX public taker 5 bps which is the reality for
    market orders.  Slippage is taken from the recorded ``slippage_bps``
    column — only its absolute value matters since negative = favorable
    and we don't want to subsidise the deadzone for it.
    """
    if symbol:
        rows = [r for r in rows if r.get("symbol") == symbol]

    if not rows:
        return {"n_fills": 0}

    slippage_abs = [abs(float(r.get("slippage_bps", 0.0))) for r in rows]
    latency_ms = [float(r.get("latency_ms", 0.0)) for r in rows]
    slip_med = _median(slippage_abs)
    slip_p75 = sorted(slippage_abs)[int(0.75 * (len(slippage_abs) - 1))] if slippage_abs else 0.0

    # Fee assumption: OKX-USDT-SWAP public taker = 5 bps.  Override with
    # --taker-fee if using Binance (~4 bps) or Bybit (~6 bps).
    fee_bps = 5.0
    rt_cost_bps = 2.0 * (fee_bps + slip_med)
    return {
        "n_fills": len(rows),
        "fee_bps": fee_bps,
        "slippage_median_bps": round(slip_med, 2),
        "slippage_p75_bps": round(slip_p75, 2),
        "rt_cost_bps_median": round(rt_cost_bps, 2),
        "rt_cost_bps_p75": round(2.0 * (fee_bps + slip_p75), 2),
        "latency_median_ms": round(_median(latency_ms), 0),
    }


def suggest_deadzone(rt_cost_bps: float) -> tuple[float, str]:
    """Linear interpolation from training cost → real cost.

    Returns (new_dz, reasoning).
    """
    if rt_cost_bps <= 0:
        return BASELINE_DEADZONE, "no cost data — keeping baseline"

    # Training cost is 6 bps (flat, one-side).  Round trip in training
    # was 12 bps.  Scale deadzone linearly in the excess.
    training_rt = 2.0 * TRAINING_COST_BPS_ASSUMED
    new_dz = BASELINE_DEADZONE * (1.0 + (rt_cost_bps - training_rt) / 30.0)
    new_dz = max(MIN_DEADZONE, min(MAX_DEADZONE, new_dz))
    reason = (f"training_rt_cost={training_rt:.0f} bps, "
              f"real_rt_cost={rt_cost_bps:.1f} bps → "
              f"dz_scale={new_dz / BASELINE_DEADZONE:.2f}x")
    return round(new_dz, 2), reason


def calibrate(venue: str, hours: float) -> dict:
    since = time.time() - hours * 3600
    rows = load_recent_tca(venue, since)
    if len(rows) < MIN_FILLS_FOR_SUGGESTION:
        return {
            "venue": venue,
            "window_hours": hours,
            "n_fills": len(rows),
            "status": "INSUFFICIENT_DATA",
            "message": (f"Only {len(rows)} fills in {hours:.0f}h — need "
                        f"≥{MIN_FILLS_FOR_SUGGESTION} for a reliable suggestion"),
        }

    symbols = sorted(set(r.get("symbol", "?") for r in rows))
    per_symbol = {}
    for sym in symbols:
        cost = compute_real_cost_bps(rows, symbol=sym)
        if cost.get("n_fills", 0) < MIN_FILLS_FOR_SUGGESTION:
            per_symbol[sym] = {"status": "INSUFFICIENT", **cost}
            continue
        dz_med, reason_med = suggest_deadzone(cost["rt_cost_bps_median"])
        dz_p75, _ = suggest_deadzone(cost["rt_cost_bps_p75"])
        per_symbol[sym] = {
            "status": "OK",
            "cost": cost,
            "baseline_deadzone": BASELINE_DEADZONE,
            "suggested_deadzone_median": dz_med,
            "suggested_deadzone_p75":    dz_p75,
            "reasoning": reason_med,
        }
    return {
        "venue": venue,
        "window_hours": hours,
        "n_fills": len(rows),
        "status": "OK",
        "symbols": per_symbol,
    }


def main() -> int:
    global MIN_FILLS_FOR_SUGGESTION  # noqa: PLW0603 — CLI override
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--venue", default=None,
                        help="One venue (bybit|binance|okx) — default all")
    parser.add_argument("--hours", type=float, default=168.0,
                        help="Lookback window (default 168h = 1 week)")
    parser.add_argument("--json", action="store_true",
                        help="Emit JSON instead of table")
    parser.add_argument("--min-fills", type=int, default=MIN_FILLS_FOR_SUGGESTION,
                        help=f"Skip suggestion if fewer than N fills "
                             f"(default {MIN_FILLS_FOR_SUGGESTION})")
    args = parser.parse_args()

    MIN_FILLS_FOR_SUGGESTION = int(args.min_fills)

    venues = [args.venue] if args.venue else ["bybit", "binance", "okx"]
    results = {v: calibrate(v, args.hours) for v in venues}

    if args.json:
        print(json.dumps(results, indent=2))
        return 0

    for venue, r in results.items():
        print(f"\n═══ {venue.upper()} ═══  window={args.hours:.0f}h  "
              f"n_fills={r['n_fills']}  status={r['status']}")
        if r["status"] != "OK":
            print(f"  {r.get('message', '(no data)')}")
            continue
        for sym, s in r["symbols"].items():
            if s.get("status") != "OK":
                print(f"  {sym:10s}  {s.get('n_fills', 0)} fills — insufficient")
                continue
            c = s["cost"]
            print(f"  {sym:10s}  n={c['n_fills']:>3d}  "
                  f"slip_med={c['slippage_median_bps']:+.1f} bps  "
                  f"slip_p75={c['slippage_p75_bps']:+.1f} bps  "
                  f"rt_cost_med={c['rt_cost_bps_median']:.1f} bps  "
                  f"lat={c['latency_median_ms']:.0f}ms")
            print(f"              baseline dz=1.20 → "
                  f"suggested dz={s['suggested_deadzone_median']:.2f} "
                  f"(p75 would be {s['suggested_deadzone_p75']:.2f})")
            print(f"              reasoning: {s['reasoning']}")

    print("\nTo apply: edit strategy/config.py SYMBOL_CONFIG[…][\"deadzone\"] "
          "or retrain with the new value.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
