"""Deribit options data collector — IV surface, PCR, skew, term structure.

Collects rich options data for implied probability alpha:
  - ATM IV per expiry (term structure)
  - Put/Call ratio (real-time from OI)
  - 25-delta skew (put IV - call IV)
  - DVOL index (hourly)
  - Options volume by strike (flow data)

Stores to SQLite for historical accumulation.

Usage:
    python3 -m scripts.data.download_deribit_options --currency BTC --db data/options/btc_options.db
    python3 -m scripts.data.download_deribit_options --currency BTC --once  # single snapshot
    python3 -m scripts.data.download_deribit_options --currency BTC --duration 24h
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

log = logging.getLogger("deribit_options")

BASE_URL = "https://www.deribit.com/api/v2/public"


def _get(endpoint: str, params: dict | None = None) -> dict:
    url = f"{BASE_URL}/{endpoint}"
    if params:
        qs = "&".join(f"{k}={v}" for k, v in params.items())
        url += f"?{qs}"
    resp = urllib.request.urlopen(url, timeout=15)
    return json.loads(resp.read())


def collect_snapshot(currency: str) -> dict:
    """Collect a full options snapshot."""
    now_ms = int(time.time() * 1000)
    now_iso = datetime.now(timezone.utc).isoformat()

    # 1. Index price
    idx = _get("get_index_price", {"index_name": f"{currency.lower()}_usd"})
    index_price = idx["result"]["index_price"]

    # 2. Book summary (all options)
    books = _get("get_book_summary_by_currency", {"currency": currency, "kind": "option"})
    instruments = books.get("result", [])

    # 3. Compute aggregate metrics
    call_oi = 0.0
    put_oi = 0.0
    call_vol = 0.0
    put_vol = 0.0
    atm_iv_by_expiry = {}

    for inst in instruments:
        name = inst.get("instrument_name", "")
        oi = inst.get("open_interest", 0) or 0
        vol = inst.get("volume", 0) or 0
        mark_iv = inst.get("mark_iv", 0) or 0
        underlying = inst.get("underlying_price", index_price) or index_price

        is_call = "-C" in name
        is_put = "-P" in name

        if is_call:
            call_oi += oi
            call_vol += vol
        elif is_put:
            put_oi += oi
            put_vol += vol

        # Extract strike and expiry
        parts = name.split("-")
        if len(parts) >= 4:
            expiry = parts[1]
            try:
                strike = float(parts[2])
            except ValueError:
                continue

            # ATM = closest to underlying
            moneyness = abs(strike - underlying) / underlying
            if moneyness < 0.05:  # within 5% of ATM
                if expiry not in atm_iv_by_expiry or moneyness < atm_iv_by_expiry[expiry]["moneyness"]:
                    atm_iv_by_expiry[expiry] = {
                        "iv": mark_iv,
                        "moneyness": moneyness,
                        "strike": strike,
                        "type": "call" if is_call else "put",
                    }

    pcr = put_oi / call_oi if call_oi > 0 else 0
    vol_pcr = put_vol / call_vol if call_vol > 0 else 0

    # 4. Term structure (ATM IV by expiry)
    term_structure = []
    for exp, data in sorted(atm_iv_by_expiry.items()):
        term_structure.append({"expiry": exp, "atm_iv": data["iv"], "strike": data["strike"]})

    # 5. DVOL
    dvol = None
    try:
        dvol_resp = _get("get_volatility_index_data", {
            "currency": currency, "resolution": "3600",
            "start_timestamp": str(now_ms - 3600000),
            "end_timestamp": str(now_ms),
        })
        dvol_data = dvol_resp.get("result", {}).get("data", [])
        if dvol_data:
            dvol = dvol_data[-1][4]  # close
    except Exception:
        pass

    # 6. Historical IV (aggregate)
    hist_iv = None
    try:
        hv = _get("get_historical_volatility", {"currency": currency})
        hv_data = hv.get("result", [])
        if hv_data:
            hist_iv = hv_data[-1][1]
    except Exception:
        pass

    return {
        "ts_ms": now_ms,
        "timestamp": now_iso,
        "index_price": index_price,
        "call_oi": call_oi,
        "put_oi": put_oi,
        "pcr": pcr,
        "call_vol_24h": call_vol,
        "put_vol_24h": put_vol,
        "vol_pcr": vol_pcr,
        "dvol": dvol,
        "hist_iv": hist_iv,
        "atm_iv_near": term_structure[0]["atm_iv"] if term_structure else None,
        "atm_iv_far": term_structure[-1]["atm_iv"] if len(term_structure) > 1 else None,
        "term_spread": (term_structure[-1]["atm_iv"] - term_structure[0]["atm_iv"]) if len(term_structure) > 1 else None,
        "n_expiries": len(term_structure),
        "term_structure": json.dumps(term_structure),
    }


def init_db(db_path: str) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_ms INTEGER NOT NULL,
            timestamp TEXT NOT NULL,
            index_price REAL,
            call_oi REAL, put_oi REAL, pcr REAL,
            call_vol_24h REAL, put_vol_24h REAL, vol_pcr REAL,
            dvol REAL, hist_iv REAL,
            atm_iv_near REAL, atm_iv_far REAL, term_spread REAL,
            n_expiries INTEGER,
            term_structure TEXT
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_snap_ts ON snapshots(ts_ms)")
    conn.commit()
    return conn


def store_snapshot(conn: sqlite3.Connection, snap: dict) -> None:
    conn.execute("""
        INSERT INTO snapshots (ts_ms, timestamp, index_price,
            call_oi, put_oi, pcr, call_vol_24h, put_vol_24h, vol_pcr,
            dvol, hist_iv, atm_iv_near, atm_iv_far, term_spread,
            n_expiries, term_structure)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        snap["ts_ms"], snap["timestamp"], snap["index_price"],
        snap["call_oi"], snap["put_oi"], snap["pcr"],
        snap["call_vol_24h"], snap["put_vol_24h"], snap["vol_pcr"],
        snap["dvol"], snap["hist_iv"],
        snap["atm_iv_near"], snap["atm_iv_far"], snap["term_spread"],
        snap["n_expiries"], snap["term_structure"],
    ))
    conn.commit()


def main():
    parser = argparse.ArgumentParser(description="Deribit options data collector")
    parser.add_argument("--currency", default="BTC", choices=["BTC", "ETH"])
    parser.add_argument("--db", default=None)
    parser.add_argument("--interval", type=int, default=300, help="Poll interval seconds")
    parser.add_argument("--once", action="store_true", help="Single snapshot then exit")
    parser.add_argument("--duration", default=None, help="Auto-stop (e.g. 24h)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    db_path = args.db or f"data/options/{args.currency.lower()}_options.db"
    conn = init_db(db_path)

    if args.once:
        snap = collect_snapshot(args.currency)
        store_snapshot(conn, snap)
        log.info("Snapshot: price=$%.0f pcr=%.4f dvol=%.1f atm_iv=%.1f term_spread=%.1f expiries=%d",
                 snap["index_price"], snap["pcr"], snap["dvol"] or 0,
                 snap["atm_iv_near"] or 0, snap["term_spread"] or 0, snap["n_expiries"])
        conn.close()
        return

    # Duration
    deadline = None
    if args.duration:
        d = args.duration.lower()
        if d.endswith("h"):
            deadline = time.time() + float(d[:-1]) * 3600
        elif d.endswith("d"):
            deadline = time.time() + float(d[:-1]) * 86400

    log.info("Collecting %s options data every %ds to %s", args.currency, args.interval, db_path)
    count = 0
    try:
        while True:
            if deadline and time.time() > deadline:
                break
            try:
                snap = collect_snapshot(args.currency)
                store_snapshot(conn, snap)
                count += 1
                log.info("#%d price=$%.0f pcr=%.4f dvol=%.1f atm_iv_near=%.1f%% term_spread=%.1f",
                         count, snap["index_price"], snap["pcr"],
                         snap["dvol"] or 0, snap["atm_iv_near"] or 0,
                         snap["term_spread"] or 0)
            except Exception:
                log.exception("Snapshot failed")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    finally:
        conn.close()
        log.info("Stopped. %d snapshots collected.", count)


if __name__ == "__main__":
    main()
