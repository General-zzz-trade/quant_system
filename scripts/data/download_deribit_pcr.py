#!/usr/bin/env python3
"""Download Deribit options PCR and max pain data.

Fetches current options book summary from Deribit (no auth needed), computes:
  - pcr_oi: put/call ratio by open interest
  - pcr_volume: put/call ratio by 24h volume
  - max_pain: strike where total option holder losses are maximized
  - max_pain_distance: (spot - max_pain) / spot
  - total_call_oi, total_put_oi, total_option_oi
  - atm_iv_near: nearest-expiry ATM implied vol

Appends one row per run to data_files/{currency}_options_daily.csv.
Designed for daily cron (also works hourly for higher frequency).

Usage:
    python3 -m scripts.data.download_deribit_pcr --currency BTC
    python3 -m scripts.data.download_deribit_pcr --all
    python3 -m scripts.data.download_deribit_pcr --all --hourly  # append to hourly CSV
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import time
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

DERIBIT_API = "https://www.deribit.com/api/v2/public"
RATE_LIMIT_S = 0.3


def _get(endpoint: str, params: dict | None = None) -> dict:
    """Fetch from Deribit public API."""
    url = f"{DERIBIT_API}/{endpoint}"
    if params:
        qs = "&".join(f"{k}={v}" for k, v in params.items())
        url += f"?{qs}"
    req = urllib.request.Request(url)
    req.add_header("User-Agent", "quant-system/1.0")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def fetch_index_price(currency: str) -> float:
    """Get current index price."""
    resp = _get("get_index_price", {"index_name": f"{currency.lower()}_usd"})
    return resp["result"]["index_price"]


def fetch_book_summary(currency: str) -> List[dict]:
    """Fetch all options book summaries for a currency."""
    resp = _get("get_book_summary_by_currency", {
        "currency": currency,
        "kind": "option",
    })
    return resp.get("result", [])


def compute_pcr(instruments: List[dict]) -> Dict[str, float]:
    """Compute put/call ratios from book summaries."""
    call_oi = 0.0
    put_oi = 0.0
    call_vol = 0.0
    put_vol = 0.0

    for inst in instruments:
        name = inst.get("instrument_name", "")
        oi = float(inst.get("open_interest", 0) or 0)
        vol = float(inst.get("volume", 0) or 0)

        if "-C" in name:
            call_oi += oi
            call_vol += vol
        elif "-P" in name:
            put_oi += oi
            put_vol += vol

    return {
        "call_oi": call_oi,
        "put_oi": put_oi,
        "pcr_oi": put_oi / call_oi if call_oi > 0 else 0.0,
        "call_vol_24h": call_vol,
        "put_vol_24h": put_vol,
        "pcr_volume": put_vol / call_vol if call_vol > 0 else 0.0,
        "total_option_oi": call_oi + put_oi,
    }


def compute_max_pain(
    instruments: List[dict], index_price: float
) -> Optional[float]:
    """Compute max pain strike from per-strike OI data.

    Max pain = the strike price where the total dollar value of outstanding
    options (both puts and calls) would cause the maximum loss to option
    holders if the underlying expires there.

    For each candidate strike K:
      - Each call with strike S < K has intrinsic value (K-S) * call_oi_at_S
      - Each put with strike S > K has intrinsic value (S-K) * put_oi_at_S
      - Total pain = sum of all intrinsic values (what holders lose)

    Max pain = strike K that MINIMIZES total intrinsic value (i.e., where
    options expire worthless to the greatest extent).
    """
    # Build per-strike OI maps for calls and puts
    call_oi_by_strike: Dict[float, float] = defaultdict(float)
    put_oi_by_strike: Dict[float, float] = defaultdict(float)
    all_strikes = set()

    for inst in instruments:
        name = inst.get("instrument_name", "")
        oi = float(inst.get("open_interest", 0) or 0)
        if oi <= 0:
            continue

        parts = name.split("-")
        if len(parts) < 4:
            continue
        try:
            strike = float(parts[2])
        except (ValueError, IndexError):
            continue

        all_strikes.add(strike)
        if "-C" in name:
            call_oi_by_strike[strike] += oi
        elif "-P" in name:
            put_oi_by_strike[strike] += oi

    if not all_strikes:
        return None

    # Only evaluate strikes near spot (within 30%)
    strikes = sorted(
        s for s in all_strikes
        if abs(s - index_price) / index_price < 0.30
    )
    if not strikes:
        strikes = sorted(all_strikes)

    min_pain = float("inf")
    max_pain_strike = None

    for k in strikes:
        total_pain = 0.0
        # Calls: holders lose nothing if strike >= K (OTM)
        # Calls with strike < K are ITM: holder gets (K - strike) * oi
        for s, oi in call_oi_by_strike.items():
            if k > s:
                total_pain += (k - s) * oi

        # Puts: holders lose nothing if strike <= K (OTM)
        # Puts with strike > K are ITM: holder gets (strike - K) * oi
        for s, oi in put_oi_by_strike.items():
            if k < s:
                total_pain += (s - k) * oi

        if total_pain < min_pain:
            min_pain = total_pain
            max_pain_strike = k

    return max_pain_strike


def compute_atm_iv(
    instruments: List[dict], index_price: float
) -> Optional[float]:
    """Get nearest-expiry ATM IV."""
    best_moneyness = float("inf")
    best_iv = None

    # Find nearest expiry first
    expiries = set()
    for inst in instruments:
        parts = inst.get("instrument_name", "").split("-")
        if len(parts) >= 4:
            expiries.add(parts[1])

    if not expiries:
        return None

    # Sort expiries by date proximity (alphabetical is close enough for
    # same-month; for accuracy we'd parse but Deribit uses DDMMMYY)
    nearest_expiry = min(expiries)  # rough heuristic

    for inst in instruments:
        name = inst.get("instrument_name", "")
        parts = name.split("-")
        if len(parts) < 4:
            continue
        if parts[1] != nearest_expiry:
            continue
        if "-C" not in name:
            continue  # use calls for ATM IV

        try:
            strike = float(parts[2])
        except ValueError:
            continue

        moneyness = abs(strike - index_price) / index_price
        mark_iv = inst.get("mark_iv", 0) or 0

        if moneyness < best_moneyness and mark_iv > 0:
            best_moneyness = moneyness
            best_iv = mark_iv

    return best_iv


def collect_pcr_snapshot(currency: str) -> Dict[str, float]:
    """Collect full PCR + max pain snapshot for a currency."""
    index_price = fetch_index_price(currency)
    instruments = fetch_book_summary(currency)

    pcr_data = compute_pcr(instruments)
    max_pain = compute_max_pain(instruments, index_price)
    atm_iv = compute_atm_iv(instruments, index_price)

    result = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "index_price": index_price,
        **pcr_data,
    }

    if max_pain is not None:
        result["max_pain"] = max_pain
        result["max_pain_distance"] = (index_price - max_pain) / index_price
    else:
        result["max_pain"] = ""
        result["max_pain_distance"] = ""

    result["atm_iv_near"] = atm_iv if atm_iv is not None else ""

    return result


FIELDNAMES = [
    "timestamp", "date", "index_price",
    "pcr_oi", "pcr_volume",
    "call_oi", "put_oi", "total_option_oi",
    "call_vol_24h", "put_vol_24h",
    "max_pain", "max_pain_distance",
    "atm_iv_near",
]


def append_to_csv(data: dict, out_path: Path) -> None:
    """Append a snapshot row to CSV file (creates header if new)."""
    exists = out_path.exists()
    with open(out_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not exists:
            writer.writeheader()
        writer.writerow({k: data.get(k, "") for k in FIELDNAMES})


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Download Deribit PCR and max pain data"
    )
    parser.add_argument("--currency", default="BTC", help="BTC or ETH")
    parser.add_argument("--all", action="store_true", help="Both BTC and ETH")
    parser.add_argument("--out-dir", default="data_files", help="Output directory")
    parser.add_argument(
        "--hourly", action="store_true",
        help="Use hourly CSV instead of daily (higher frequency accumulation)",
    )
    args = parser.parse_args()

    currencies = ["BTC", "ETH"] if args.all else [args.currency.upper()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for cur in currencies:
        try:
            snap = collect_pcr_snapshot(cur)
            suffix = "hourly" if args.hourly else "daily"
            out_path = out_dir / f"{cur.lower()}_options_{suffix}.csv"
            append_to_csv(snap, out_path)

            log.info(
                "%s: price=$%.0f  PCR(OI)=%.4f  PCR(Vol)=%.4f  "
                "max_pain=%s  max_pain_dist=%s  atm_iv=%s  "
                "total_oi=%.0f -> %s",
                cur,
                snap["index_price"],
                snap["pcr_oi"],
                snap["pcr_volume"],
                snap.get("max_pain", "N/A"),
                f'{snap["max_pain_distance"]:.4f}' if snap["max_pain_distance"] != "" else "N/A",
                f'{snap["atm_iv_near"]:.1f}%' if snap["atm_iv_near"] != "" else "N/A",
                snap["total_option_oi"],
                out_path,
            )
        except Exception:
            log.exception("Failed to collect %s options data", cur)
        time.sleep(RATE_LIMIT_S)


if __name__ == "__main__":
    main()
