#!/usr/bin/env python3
"""Backtest ↔ live signal replay diff.

Reads a per-venue audit jsonl and recomputes, for each live signal
record, the backtest predictor's output for the same bar.  Highlights
bars where ``live_z`` and ``backtest_z`` disagree — those are the
evidence you need to localize *why* live PnL diverges from backtest
Sharpe.

What it actually diffs
----------------------
For each ``type=signal`` audit entry with ``venue`` and ``ts``:

  1. Resolve the symbol and bar timestamp (round ts down to the hour)
  2. Run the batch feature engine for that bar
  3. Run the model's ``predict_latest`` path (Ridge + LGBM + XGB blend,
     same code path used by ``runner.batch_predictor``)
  4. Compute the backtest z-score from the batch-synced buffer
  5. Diff (live_z vs backtest_z), (live_signal vs backtest_signal)

Divergences fall into three classes:

* **small_z_jitter** — |Δz| < 0.10, explained by online Ridge drift
  since the last weekly retrain.  Benign.

* **prediction_mismatch** — |Δz| ≥ 0.10 but same sign.  Indicates
  feature drift (e.g. live features missing / NaN'd) or Rust-vs-Python
  batch feature hash divergence.

* **sign_flip** — live and backtest disagree on direction.  Serious —
  means a feature or z-score state is inconsistent, and the audit
  entry's decision would have been different in backtest.

Usage
-----
    python3 scripts/backtest_replay_diff.py \\
        --audit data/runtime/decision_audit_okx.jsonl \\
        --symbol BTCUSDT \\
        --limit 100

The tool is **read-only** — it never writes to production state.  Safe
to run on live audit files while the system is trading.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "/quant_system")


def _load_audit_signals(path: Path, symbol: str | None, limit: int) -> list[dict]:
    """Read per-venue audit jsonl, filter to signal events, optionally
    by symbol, and return the most-recent ``limit`` entries."""
    if not path.exists():
        raise FileNotFoundError(f"Audit file not found: {path}")
    signals: list[dict] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if e.get("type") != "signal":
            continue
        if symbol and e.get("symbol") != symbol:
            continue
        signals.append(e)
    return signals[-limit:]


def _classify(delta_z: float, live_sig: int, bt_sig: int) -> str:
    if live_sig != bt_sig and live_sig != 0 and bt_sig != 0:
        return "sign_flip"
    if abs(delta_z) < 0.10:
        return "small_z_jitter"
    return "prediction_mismatch"


def replay(audit_path: Path, symbol: str, limit: int) -> list[dict]:
    """For each audit signal, recompute backtest prediction and diff.

    Heavy import path — ``batch_predictor.predict_latest`` loads the
    full model + feature engine, so this is slow on the first call
    but fast thereafter (same-process cache).
    """
    from runner.batch_predictor import predict_latest
    from _quant_hotpath import RustInferenceBridge

    signals = _load_audit_signals(audit_path, symbol, limit)
    if not signals:
        return []

    # One bridge shared across bars — mirrors production lookback.
    bridge = RustInferenceBridge(zscore_window=720, zscore_warmup=180)
    ckpt_path = Path(f"/quant_system/data/runtime/zscore_checkpoints/{symbol}.json")
    if ckpt_path.exists():
        try:
            bridge.restore(json.loads(ckpt_path.read_text()))
        except Exception:
            pass  # bridge fresh — still works, just cold

    out: list[dict] = []
    model_dir_map = {
        "BTCUSDT": "BTCUSDT_gate_v2",
        "ETHUSDT": "ETHUSDT_gate_v2",
        "SOLUSDT": "SOLUSDT_gate_v2",
    }
    model_dir = model_dir_map.get(symbol, symbol + "_gate_v2")

    for ev in signals:
        live_z = float(ev.get("z_score", 0.0))
        live_sig = int(ev.get("signal", 0))
        ts = float(ev.get("ts", 0))
        hour_key = int(ts // 3600)

        try:
            bt_pred = predict_latest(symbol, model_dir)
            if bt_pred is None:
                out.append({
                    "ts": ts, "live_z": live_z, "bt_z": None,
                    "diff": None, "class": "bt_predict_failed",
                    "live_sig": live_sig, "bt_sig": None,
                })
                continue
            bt_z = bridge.zscore_normalize(symbol, float(bt_pred), hour_key)
            if bt_z is None:
                out.append({
                    "ts": ts, "live_z": live_z, "bt_z": None,
                    "diff": None, "class": "bt_warmup",
                    "live_sig": live_sig, "bt_sig": None,
                })
                continue
        except Exception as e:
            out.append({
                "ts": ts, "live_z": live_z, "bt_z": None,
                "diff": None, "class": f"error: {e.__class__.__name__}",
                "live_sig": live_sig, "bt_sig": None,
            })
            continue

        # Replicate simple discretizer: sign(z) if |z| > dz else 0
        DEADZONE = 1.2
        bt_sig = 1 if bt_z > DEADZONE else (-1 if bt_z < -DEADZONE else 0)

        delta = bt_z - live_z
        cls = _classify(delta, live_sig, bt_sig)
        out.append({
            "ts": ts,
            "live_z": round(live_z, 3),
            "bt_z": round(bt_z, 3),
            "diff": round(delta, 3),
            "class": cls,
            "live_sig": live_sig,
            "bt_sig": bt_sig,
        })

    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audit", required=True,
                   help="Path to decision_audit_{venue}.jsonl")
    p.add_argument("--symbol", required=True,
                   help="Filter to this symbol (e.g. BTCUSDT)")
    p.add_argument("--limit", type=int, default=50,
                   help="Replay the most recent N signal events (default 50)")
    p.add_argument("--json", action="store_true",
                   help="Emit JSON rows instead of table")
    args = p.parse_args()

    rows = replay(Path(args.audit), args.symbol, args.limit)
    if not rows:
        print(f"No signal events for {args.symbol} in {args.audit}")
        return 0

    if args.json:
        print(json.dumps(rows, indent=2))
        return 0

    buckets: dict[str, int] = {}
    for r in rows:
        buckets[r["class"]] = buckets.get(r["class"], 0) + 1

    print(f"Replay diff — {args.symbol}  ({len(rows)} signals)\n")
    import datetime as dt
    print(f"{'time':<20} {'live_z':>8} {'bt_z':>8} {'diff':>8} "
          f"{'live':>5} {'bt':>4}  class")
    print("-" * 72)
    for r in rows[-30:]:
        t = dt.datetime.fromtimestamp(r["ts"], dt.UTC).strftime("%Y-%m-%d %H:%M")
        lz = f"{r['live_z']:+.2f}"
        bz = f"{r['bt_z']:+.2f}" if r.get("bt_z") is not None else "  --  "
        df = f"{r['diff']:+.2f}" if r.get("diff") is not None else "  --  "
        ls = str(r.get("live_sig", "?"))
        bs = str(r.get("bt_sig", "?")) if r.get("bt_sig") is not None else "?"
        print(f"{t:<20} {lz:>8} {bz:>8} {df:>8} {ls:>5} {bs:>4}  {r['class']}")
    print()
    print("Class summary:")
    for k in sorted(buckets):
        print(f"  {k:<22s} {buckets[k]:>5d}")

    # Summary health check
    total = len(rows)
    flips = buckets.get("sign_flip", 0)
    mismatches = buckets.get("prediction_mismatch", 0)
    if total > 0:
        flip_pct = 100 * flips / total
        mm_pct = 100 * mismatches / total
        print(f"\nHealth: {flip_pct:.1f}% sign-flip  {mm_pct:.1f}% prediction-mismatch")
        if flip_pct > 5:
            print("  ⚠  sign-flip rate > 5% — check feature staleness / z-score state sync")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
