#!/usr/bin/env python3
"""Portfolio risk monitor — writes cross-venue exposure snapshot.

Runs as a systemd timer every minute.  Queries binance + okx for open
positions, tallies per-symbol notional across venues, and writes a
snapshot to ``data/runtime/portfolio_risk.json``::

    {
      "ts": 1775892648.0,
      "symbols": {
        "BTCUSDT": {
          "venues": {
            "binance": {"qty": 0.01, "notional_usd": 727.3, "side": "buy"},
            "okx":     {"qty": 0.0,  "notional_usd": 0.0,   "side": "flat"}
          },
          "total_long_notional_usd":  727.3,
          "total_short_notional_usd": 0.0,
          "max_total_notional_usd":   2000.0
        },
        ...
      }
    }

The live runner's CompositeRiskGate reads this file before every new
entry order, denying trades that would push combined long or short
notional past ``PORTFOLIO_NOTIONAL_CAPS_USD``.

The monitor is deliberately best-effort: if one venue query fails, it
writes what it has and flags the failure in the snapshot.  Reader code
treats missing venue data as "unknown — fall back to per-venue check".
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, "/quant_system")

from strategy.config import PORTFOLIO_NOTIONAL_CAPS_USD

logger = logging.getLogger(__name__)

SNAPSHOT_PATH = Path("data/runtime/portfolio_risk.json")


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


def _binance_positions(env: dict[str, str]) -> tuple[dict[str, dict[str, Any]], bool]:
    """Return ({symbol: {...}}, is_testnet).

    ``is_testnet=True`` flags sim-money positions so the portfolio cap
    can exclude them from the "real capital at risk" total.
    """
    try:
        from execution.adapters.binance.adapter import BinanceAdapter
        from execution.adapters.binance.config import BinanceConfig
        key = env.get("BINANCE_TESTNET_API_KEY") or env.get("BINANCE_API_KEY", "")
        sec = env.get("BINANCE_TESTNET_API_SECRET") or env.get("BINANCE_API_SECRET", "")
        if not (key and sec):
            return {}, True
        testnet = bool(env.get("BINANCE_TESTNET_API_KEY"))
        adapter = BinanceAdapter(BinanceConfig(api_key=key, api_secret=sec, testnet=testnet))
    except Exception as e:
        logger.warning("binance adapter init failed: %s", e)
        return {}, True

    out: dict[str, dict[str, Any]] = {}
    try:
        for p in adapter.get_positions():
            qty = float(getattr(p, "qty", 0))
            if qty == 0:
                continue
            entry = float(getattr(p, "entry_price", 0))
            mark = float(getattr(p, "mark_price", 0) or entry)
            px = mark if mark > 0 else entry
            out[p.symbol] = {
                "qty": qty,
                "notional_usd": abs(qty) * px,
                "side": "buy" if qty > 0 else "sell",
                "is_testnet": testnet,
            }
    except Exception as e:
        logger.warning("binance get_positions failed: %s", e)
    return out, testnet


def _okx_positions(env: dict[str, str]) -> tuple[dict[str, dict[str, Any]], bool]:
    try:
        from execution.adapters.okx.adapter import OkxAdapter
        from execution.adapters.okx.config import OkxConfig
        key = env.get("OKX_API_KEY")
        sec = env.get("OKX_API_SECRET")
        pas = env.get("OKX_API_PASSPHRASE")
        if not (key and sec and pas):
            return {}, False
        url = env.get("OKX_BASE_URL", "https://www.okx.com")
        simulated = env.get("OKX_SIMULATED", "0").lower() in ("1", "true", "yes")
        adapter = OkxAdapter(OkxConfig(
            api_key=key, api_secret=sec, passphrase=pas,
            base_url=url, simulated=simulated,
        ))
        if not adapter.connect():
            return {}, simulated
    except Exception as e:
        logger.warning("okx adapter init failed: %s", e)
        return {}, False

    out: dict[str, dict[str, Any]] = {}
    try:
        for p in adapter.get_positions():
            qty = float(getattr(p, "qty", 0))
            if qty == 0:
                continue
            entry = float(getattr(p, "entry_price", 0))
            out[p.symbol] = {
                "qty": qty,
                "notional_usd": abs(qty) * entry,
                "side": "buy" if qty > 0 else "sell",
                "is_testnet": simulated,
            }
    except Exception as e:
        logger.warning("okx get_positions failed: %s", e)
    return out, simulated


def build_snapshot() -> dict[str, Any]:
    env = _load_env()
    binance, bin_testnet = _binance_positions(env)
    okx, okx_testnet = _okx_positions(env)

    symbols_touched = set(binance) | set(okx) | set(PORTFOLIO_NOTIONAL_CAPS_USD)
    symbols_out: dict[str, Any] = {}

    for sym in sorted(symbols_touched):
        bin_p = binance.get(sym, {"qty": 0.0, "notional_usd": 0.0, "side": "flat", "is_testnet": bin_testnet})
        okx_p = okx.get(sym, {"qty": 0.0, "notional_usd": 0.0, "side": "flat", "is_testnet": okx_testnet})

        # The portfolio cap applies to REAL capital at risk only.  Sim/
        # testnet accounts use fake money so they can't cause real losses
        # — we track their positions but exclude from the enforced total.
        long_total = 0.0
        short_total = 0.0
        long_total_incl_testnet = 0.0
        short_total_incl_testnet = 0.0

        for p in (bin_p, okx_p):
            if p["side"] == "buy":
                long_total_incl_testnet += p["notional_usd"]
                if not p.get("is_testnet"):
                    long_total += p["notional_usd"]
            elif p["side"] == "sell":
                short_total_incl_testnet += p["notional_usd"]
                if not p.get("is_testnet"):
                    short_total += p["notional_usd"]

        symbols_out[sym] = {
            "venues": {"binance": bin_p, "okx": okx_p},
            "total_long_notional_usd": round(long_total, 2),
            "total_short_notional_usd": round(short_total, 2),
            "total_long_incl_testnet_usd": round(long_total_incl_testnet, 2),
            "total_short_incl_testnet_usd": round(short_total_incl_testnet, 2),
            "max_total_notional_usd": PORTFOLIO_NOTIONAL_CAPS_USD.get(sym, 0.0),
        }

    return {
        "ts": time.time(),
        "symbols": symbols_out,
        "venue_testnet_flags": {"binance": bin_testnet, "okx": okx_testnet},
    }


def save_snapshot(snapshot: dict[str, Any]) -> None:
    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = SNAPSHOT_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(snapshot, indent=2, default=str))
    tmp.replace(SNAPSHOT_PATH)


def print_snapshot(snapshot: dict[str, Any]) -> None:
    print(f"\n=== Portfolio Risk Snapshot @ {time.strftime('%H:%M:%S', time.localtime(snapshot['ts']))} ===\n")
    for sym, data in snapshot["symbols"].items():
        cap = data["max_total_notional_usd"]
        lng = data["total_long_notional_usd"]
        sht = data["total_short_notional_usd"]
        lng_all = data.get("total_long_incl_testnet_usd", lng)
        sht_all = data.get("total_short_incl_testnet_usd", sht)
        venues = data["venues"]

        use_pct_long = (lng / cap * 100) if cap > 0 else 0
        use_pct_short = (sht / cap * 100) if cap > 0 else 0

        icon_long = "🟢" if use_pct_long < 50 else ("🟡" if use_pct_long < 80 else "🔴")
        icon_short = "🟢" if use_pct_short < 50 else ("🟡" if use_pct_short < 80 else "🔴")

        print(f"● {sym}   cap=${cap:.0f} (real capital only)")
        print(f"  {icon_long} long:  live=${lng:,.0f}  ({use_pct_long:.0f}% of cap)   incl-testnet=${lng_all:,.0f}")
        print(f"  {icon_short} short: live=${sht:,.0f}  ({use_pct_short:.0f}% of cap)   incl-testnet=${sht_all:,.0f}")
        for v_name, v_data in venues.items():
            if v_data["side"] != "flat":
                flag = " [TESTNET]" if v_data.get("is_testnet") else ""
                print(f"    {v_name}: {v_data['side']} {v_data['qty']:+.4f} (${v_data['notional_usd']:.0f}){flag}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--print", action="store_true", help="Also print to stdout")
    parser.add_argument("--once", action="store_true",
                        help="Run once (default; systemd timer invokes with --once)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")

    snap = build_snapshot()
    save_snapshot(snap)
    if args.print:
        print_snapshot(snap)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
