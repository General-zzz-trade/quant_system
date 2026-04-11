"""OKX status probe — hourly snapshot for the first-trade watch window.

Hits OKX REST for:
  * account USDT equity + avail
  * open positions (symbol, qty, notional, unrealized pnl)
  * recent fills (last hour)

Writes one JSON line to ``data/runtime/okx_status.jsonl`` each run so
the equity / position timeline is queryable.  When a new fill appears
vs the previous snapshot, pulls TCA rows for that fill window and
prints a one-line alert.

Intended to run every hour via systemd timer until OKX has 30+ real
fills and the D13 balanced config is validated.  After that the
existing daily_pnl_alert + ic_decay_monitor cover the same ground.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, "/quant_system")

logger = logging.getLogger(__name__)

STATUS_PATH = Path("/quant_system/data/runtime/okx_status.jsonl")


def _probe() -> dict:
    try:
        from execution.adapters.okx.adapter import OkxAdapter
        from execution.adapters.okx.config import OkxConfig
    except Exception as e:
        return {"error": f"import: {e}"}

    try:
        cfg = OkxConfig.from_env()
        adapter = OkxAdapter(cfg)
        if not adapter.connect():
            return {"error": "connect failed"}

        bal = adapter.get_balances()
        usdt = bal.get("USDT")
        equity = float(usdt.total) if usdt else 0.0
        avail = float(usdt.free) if usdt else 0.0

        positions = []
        try:
            pos = adapter.get_positions()
            for p in pos or []:
                if not p.is_flat:
                    positions.append({
                        "symbol": p.symbol,
                        "side": "long" if p.is_long else "short",
                        "qty": float(p.qty),
                        "entry": float(p.entry_price),
                    })
        except Exception as e:
            logger.debug("get_positions failed: %s", e)

        fills = []
        try:
            # get_recent_fills returns the last ~10 by default
            recent = adapter.get_recent_fills(limit=20)
            for f in recent:
                fills.append({
                    "ts_ms": int(f.ts_ms),
                    "symbol": f.symbol,
                    "side": f.side,
                    "qty": float(f.qty),
                    "price": float(f.price),
                })
        except Exception as e:
            logger.debug("get_recent_fills failed: %s", e)

        return {
            "ts": time.time(),
            "equity": round(equity, 2),
            "avail": round(avail, 2),
            "positions": positions,
            "recent_fills": fills,
        }
    except Exception as e:
        return {"error": f"probe: {e}"}


def _append(row: dict) -> None:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(STATUS_PATH, "a") as f:
        f.write(json.dumps(row, default=str) + "\n")


def _previous_fill_ids(row: dict) -> set:
    """Collect fill_ids from the most recent prior snapshot."""
    if not STATUS_PATH.exists():
        return set()
    try:
        lines = STATUS_PATH.read_text().splitlines()
        for line in reversed(lines[-10:]):
            if not line.strip():
                continue
            r = json.loads(line)
            return {f"{f['ts_ms']}_{f['symbol']}" for f in r.get("recent_fills", [])}
    except Exception:
        return set()
    return set()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    row = _probe()
    if "error" in row:
        logger.error("probe failed: %s", row["error"])
        _append({"ts": time.time(), **row})
        return 1

    prior = _previous_fill_ids(row)
    new_fills = [
        f for f in row["recent_fills"]
        if f"{f['ts_ms']}_{f['symbol']}" not in prior
    ]
    _append(row)

    pos_s = ", ".join(
        f"{p['symbol']}:{p['side']} {p['qty']:.4f}@{p['entry']:.1f}"
        for p in row["positions"]
    ) or "flat"
    print(f"OKX equity=${row['equity']:.2f} avail=${row['avail']:.2f}  {pos_s}")

    if new_fills:
        print(f"⚡ NEW FILLS ({len(new_fills)}):")
        for f in new_fills:
            import datetime as _dt
            t = _dt.datetime.fromtimestamp(f["ts_ms"] / 1000, _dt.timezone.utc)
            print(f"  {t.isoformat(timespec='seconds')} "
                  f"{f['symbol']} {f['side']} qty={f['qty']} @ ${f['price']:.2f}")
        try:
            from monitoring.tca import slippage_summary
            summary = slippage_summary("okx", lookback_hours=24)
            if summary.get("total_count", 0) > 0:
                print(f"  TCA 24h: n={summary['total_count']} "
                      f"median_bps={summary['overall_median_bps']:+.1f}")
        except Exception:
            pass
    else:
        print("  (no new fills since last probe)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
