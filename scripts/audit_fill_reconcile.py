"""Cross-check decision audit entries/exits vs actual OKX fills.

For each audit entry/exit record we expect a matching fill on the
exchange within ±2 minutes. Reports mismatches grouped by severity:

  ORPHAN_AUDIT       audit says we entered/exited but no fill on exchange
  ORPHAN_FILL        fill on exchange but no audit record
  QTY_MISMATCH       audit qty differs from fill qty by > step_size
  PRICE_DRIFT        audit price differs from fill price by > 50 bps
  TIMING_LATE        fill arrived > 5 min after audit (slow execution)

Audit file: data/runtime/decision_audit_okx.jsonl
Fill source: OKX REST /api/v5/trade/fills-history
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, "/quant_system")


def _iter_audit():
    base = Path("data/runtime")
    for f in sorted(base.glob("decision_audit_okx.jsonl*")):
        op = gzip.open if f.suffix == ".gz" else open
        try:
            with op(f, "rt") as fp:
                for line in fp:
                    try:
                        yield json.loads(line)
                    except Exception:
                        continue
        except Exception as e:
            print(f"warn: {f}: {e}", file=sys.stderr)


def _make_client():
    from dotenv import load_dotenv

    load_dotenv("/quant_system/.env")
    from execution.adapters.okx.rest import OkxRestClient, OkxRestConfig

    return OkxRestClient(OkxRestConfig(
        api_key=os.environ.get("OKX_API_KEY", ""),
        api_secret=os.environ.get("OKX_API_SECRET", ""),
        passphrase=os.environ.get("OKX_API_PASSPHRASE", ""),
        base_url=os.environ.get("OKX_BASE_URL", "https://www.okx.com"),
    ))


def _paginated_fetch(client, path, base_params, days, page_size=100, page_field="billId"):
    cutoff_ms = int((datetime.now(timezone.utc).timestamp() - days * 86400) * 1000)
    out = []
    after = ""
    while True:
        params = dict(base_params)
        params["limit"] = str(page_size)
        if after:
            params["after"] = after
        try:
            resp = client.request_signed(method="GET", path=path, params=params)
        except Exception as e:
            print(f"warn: {path} error: {e}", file=sys.stderr)
            break
        data = resp.get("data") or []
        if not data:
            break
        oldest_ts = int(data[-1].get("ts") or data[-1].get("cTime") or 0)
        for r in data:
            ts = int(r.get("ts") or r.get("cTime") or 0)
            if ts < cutoff_ms:
                continue
            out.append(r)
        if oldest_ts < cutoff_ms or len(data) < page_size:
            break
        after = data[-1].get(page_field) or data[-1].get("ordId") or ""
        if not after:
            break
    return out


def _fetch_fills(symbol_inst: str, days: int):
    return _paginated_fetch(_make_client(), "/api/v5/trade/fills-history",
                            {"instType": "SWAP", "instId": symbol_inst}, days)


def _fetch_orders(symbol_inst: str, days: int):
    return _paginated_fetch(_make_client(), "/api/v5/trade/orders-history",
                            {"instType": "SWAP", "instId": symbol_inst}, days,
                            page_field="ordId")


SYMBOL_TO_INST = {
    "BTCUSDT": "BTC-USDT-SWAP",
    "ETHUSDT": "ETH-USDT-SWAP",
}
CT_VAL = {
    "BTCUSDT": 0.01,
    "ETHUSDT": 0.1,
}


def _to_coin(symbol: str, contracts: float) -> float:
    return contracts * CT_VAL.get(symbol, 1.0)


def _classify(audit_records, fills_by_symbol, orders_by_symbol, days):
    """Return list of (severity, dict) issues.

    Improvements vs v1:
      - Aggregate same-order partial fills (group by ordId)
      - Cross-reference orders-history to label orphan fills with ordType
        (post_only / market) and reduceOnly status
      - Tighter audit-side normalization: an "entry sell" should match a
        "sell" fill at NOT reduceOnly; an "exit sell" should match a "sell"
        fill at reduceOnly=true
    """
    issues = []
    cutoff = datetime.now(timezone.utc).timestamp() - days * 86400

    audit_filtered = [
        r for r in audit_records
        if r.get("type") in ("entry", "exit")
        and (not r.get("venue") or r.get("venue") == "okx")
        and float(r.get("ts", 0)) >= cutoff
        and r.get("symbol", "") in SYMBOL_TO_INST
    ]

    print(f"  Audit entries/exits in window: {len(audit_filtered)}")
    print("  Fills in window: " + ", ".join(
        f"{s}={len(fills_by_symbol.get(s, []))}" for s in SYMBOL_TO_INST))
    print("  Orders in window: " + ", ".join(
        f"{s}={len(orders_by_symbol.get(s, []))}" for s in SYMBOL_TO_INST))

    # Build orderId → orderType, reduceOnly map
    order_meta = {}  # symbol -> {ordId: {ordType, reduceOnly, intent_sz}}
    for sym, orders in orders_by_symbol.items():
        m = {}
        for o in orders:
            oid = o.get("ordId", "")
            if not oid:
                continue
            m[oid] = {
                "ordType": o.get("ordType", "?"),
                "reduceOnly": o.get("reduceOnly") == "true",
                "intent_sz": float(o.get("sz") or 0),
                "fillSz": float(o.get("fillSz") or 0),
                "state": o.get("state", "?"),
                "ts": int(o.get("cTime", 0)) / 1000,
            }
        order_meta[sym] = m

    # Aggregate fills by ordId
    fill_index = {}
    for sym, fills in fills_by_symbol.items():
        agg = {}  # ordId -> aggregated fill
        for f in fills:
            oid = str(f.get("ordId") or "")
            ts = int(f.get("ts") or 0) / 1000
            side = f.get("side")
            sz_contracts = float(f.get("fillSz") or 0)
            qty_coin = _to_coin(sym, sz_contracts)
            price = float(f.get("fillPx") or 0)
            fee = float(f.get("fee") or 0)
            if oid in agg:
                # Weighted average price for partial fills under same order
                old = agg[oid]
                tot_qty = old["qty_coin"] + qty_coin
                old["price"] = (old["price"] * old["qty_coin"]
                                + price * qty_coin) / max(tot_qty, 1e-9)
                old["qty_coin"] = tot_qty
                old["fee"] += fee
                old["ts"] = max(old["ts"], ts)
                old["n_fills"] += 1
            else:
                agg[oid] = {
                    "ordId": oid, "ts": ts, "side": side,
                    "qty_coin": qty_coin, "price": price, "fee": fee,
                    "matched": False, "n_fills": 1,
                }
        # Add metadata from order_meta
        for oid, fa in agg.items():
            meta = order_meta.get(sym, {}).get(oid, {})
            fa["ordType"] = meta.get("ordType", "?")
            fa["reduceOnly"] = meta.get("reduceOnly", False)
            fa["intent_qty_coin"] = _to_coin(sym, meta.get("intent_sz", 0))
        fill_index[sym] = sorted(agg.values(), key=lambda x: x["ts"])

    # Match each audit to its closest aggregated-order fill within ±5 min
    SAME_SIDE_TOL_S = 300
    for r in audit_filtered:
        sym = r["symbol"]
        a_ts = float(r["ts"])
        a_side = r.get("side", "?").lower()
        a_qty = float(r.get("qty") or 0)
        a_price = float(r.get("price") or 0)
        a_type = r["type"]
        a_reduce = (a_type == "exit")
        candidates = fill_index.get(sym, [])

        best = None
        best_dt = 99999
        for c in candidates:
            if c["matched"]:
                continue
            if c["side"] != a_side:
                continue
            # Prefer same reduceOnly flag (entry → !reduce; exit → reduce)
            if c.get("reduceOnly") != a_reduce:
                continue
            dt = abs(c["ts"] - a_ts)
            if dt < best_dt:
                best_dt = dt
                best = c

        if best is None or best_dt > SAME_SIDE_TOL_S:
            issues.append(("ORPHAN_AUDIT", {
                "audit_ts": datetime.fromtimestamp(a_ts, timezone.utc).isoformat(timespec="seconds"),
                "symbol": sym, "type": a_type, "side": a_side,
                "qty": a_qty, "price": a_price,
                "nearest_fill_dt_s": best_dt if best else None,
            }))
            continue

        best["matched"] = True
        qty_diff = a_qty - best["qty_coin"]
        qty_pct = abs(qty_diff) / max(a_qty, 1e-9) * 100 if a_qty > 0 else 0
        price_diff_bps = (best["price"] - a_price) / a_price * 10000 if a_price > 0 else 0

        if qty_pct > 5:
            issues.append(("QTY_MISMATCH", {
                "audit_ts": datetime.fromtimestamp(a_ts, timezone.utc).isoformat(timespec="seconds"),
                "symbol": sym, "type": a_type, "side": a_side,
                "audit_qty": a_qty,
                "fill_qty_coin": round(best["qty_coin"], 6),
                "intent_qty_coin": round(best.get("intent_qty_coin", 0), 6),
                "diff_pct": round(qty_pct, 1),
                "ordType": best.get("ordType"),
                "n_partial_fills": best.get("n_fills", 1),
                "audit_price": a_price, "fill_price": round(best["price"], 2),
            }))
        elif abs(price_diff_bps) > 50:
            issues.append(("PRICE_DRIFT", {
                "audit_ts": datetime.fromtimestamp(a_ts, timezone.utc).isoformat(timespec="seconds"),
                "symbol": sym, "type": a_type,
                "audit_price": a_price, "fill_price": round(best["price"], 2),
                "drift_bps": round(price_diff_bps, 1),
            }))
        if best_dt > 60:
            issues.append(("TIMING_LATE", {
                "audit_ts": datetime.fromtimestamp(a_ts, timezone.utc).isoformat(timespec="seconds"),
                "symbol": sym, "type": a_type,
                "delay_s": round(best_dt, 1),
            }))

    # Find unmatched aggregated orders (orphan fills)
    for sym, fills in fill_index.items():
        for c in fills:
            if c["matched"]:
                continue
            if c["ts"] < cutoff:
                continue
            issues.append(("ORPHAN_FILL", {
                "fill_ts": datetime.fromtimestamp(c["ts"], timezone.utc).isoformat(timespec="seconds"),
                "symbol": sym, "side": c["side"],
                "qty_coin": round(c["qty_coin"], 4),
                "intent_qty_coin": round(c.get("intent_qty_coin", 0), 4),
                "price": round(c["price"], 2),
                "ordType": c.get("ordType"),
                "reduceOnly": c.get("reduceOnly"),
                "n_partial_fills": c.get("n_fills", 1),
            }))

    return issues


def _send_telegram_summary(issues, days):
    """Send a brief summary of new mismatches to Telegram."""
    try:
        from monitoring.notify import send_alert, AlertLevel
    except Exception:
        return
    sev_counts = {}
    for sev, _ in issues:
        sev_counts[sev] = sev_counts.get(sev, 0) + 1
    blocking = (sev_counts.get("ORPHAN_AUDIT", 0)
                + sev_counts.get("QTY_MISMATCH", 0))
    if blocking == 0:
        return  # don't spam if only orphan-fills (those are mostly historical)
    details = {f"{k}": v for k, v in sev_counts.items()}
    details["window_days"] = days
    try:
        send_alert(
            AlertLevel.WARNING,
            f"Audit/fill reconcile: {blocking} blocking issues",
            details=details,
            source="audit_reconcile",
        )
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="Audit ↔ fill reconciliation")
    parser.add_argument("--days", type=int, default=30,
                        help="Window in days (default 30)")
    parser.add_argument("--symbols", nargs="+",
                        default=list(SYMBOL_TO_INST.keys()))
    parser.add_argument("--alert", action="store_true",
                        help="Send Telegram summary if any blocking issues")
    args = parser.parse_args()

    print("=" * 72)
    print(f"OKX audit ↔ fill reconciliation — last {args.days} days")
    print("=" * 72)

    audit_records = list(_iter_audit())
    print(f"  Loaded {len(audit_records)} total audit records")

    fills_by_symbol = {}
    orders_by_symbol = {}
    for sym in args.symbols:
        inst = SYMBOL_TO_INST[sym]
        fills_by_symbol[sym] = _fetch_fills(inst, args.days)
        orders_by_symbol[sym] = _fetch_orders(inst, args.days)

    issues = _classify(audit_records, fills_by_symbol, orders_by_symbol, args.days)

    # Summarize
    sev_counts = {}
    for sev, _ in issues:
        sev_counts[sev] = sev_counts.get(sev, 0) + 1

    print()
    if not issues:
        print("  No issues detected ✓")
        return

    print("Issues by severity:")
    for sev in ("ORPHAN_AUDIT", "ORPHAN_FILL", "QTY_MISMATCH",
                "PRICE_DRIFT", "TIMING_LATE"):
        if sev in sev_counts:
            print(f"  {sev:14s} {sev_counts[sev]}")

    print()
    for sev_filter in ("ORPHAN_AUDIT", "ORPHAN_FILL", "QTY_MISMATCH",
                       "PRICE_DRIFT", "TIMING_LATE"):
        sub = [(s, d) for s, d in issues if s == sev_filter]
        if not sub:
            continue
        print(f"--- {sev_filter} ({len(sub)} cases) ---")
        for _, d in sub[:10]:  # limit per category
            print(f"  {json.dumps(d, default=str)}")
        if len(sub) > 10:
            print(f"  ... and {len(sub) - 10} more")
        print()

    if args.alert:
        _send_telegram_summary(issues, args.days)


if __name__ == "__main__":
    main()
