"""The account's own market orders against the public tape (2026-11-16): what a real fill cost.

Input: the read-only export of the account's orders and trades (python3 -m scripts.account_check --history DIR,
private, not in the repository) and Binance's public aggregated trades of the same days
(data_files/binance_aggtrades_fills_2026-11-12, checksums verified). Orders up to 2026-11-12 only.

For each market order: the price just before it (the middle of the last taker buy and the last taker sell before
the order's first fill), the order's average fill price against that middle, in basis points, the fee actually
charged, and the time from the order to its first fill. Output: aggregate figures only.
"""
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
TAPE = ROOT / "data_files" / "binance_aggtrades_fills_2026-11-12"
HISTORY = Path(sys.argv[1])
CUT = int(pd.Timestamp("2026-11-13", tz="UTC").timestamp() * 1000)


def tape(symbol, day, cache={}):
    if (symbol, day) not in cache:
        with zipfile.ZipFile(TAPE / f"{symbol}-aggTrades-{day}.zip") as z:
            raw = z.read(z.namelist()[0])
        head = raw[:200].decode().split("\n")[0]
        df = pd.read_csv(io.BytesIO(raw), header=0 if "price" in head else None,
                         names=["id", "price", "qty", "first", "last", "time", "buyer_maker"])
        maker = df["buyer_maker"].astype(str).str.lower().eq("true").to_numpy()
        cache.clear()
        cache[(symbol, day)] = (df["time"].to_numpy(np.int64), df["price"].to_numpy(float), maker)
    return cache[(symbol, day)]


orders = pd.concat([pd.DataFrame(json.loads(p.read_text())) for p in HISTORY.glob("orders_*.json")])
trades = pd.concat([pd.DataFrame(json.loads(p.read_text())) for p in HISTORY.glob("trades_*.json")])
for c in ("price", "qty", "quoteQty", "commission"):
    trades[c] = trades[c].astype(float)
market = orders[(orders["type"] == "MARKET") & (orders["time"] < CUT) & (orders["status"] == "FILLED")].copy()
market["day"] = pd.to_datetime(market["time"], unit="ms").dt.strftime("%Y-%m-%d")
rows = []
for o in market.sort_values(["symbol", "time"]).itertuples():
    own = trades[trades["orderId"] == o.orderId].sort_values("time")
    if own.empty:
        continue
    t, p, buyer_maker = tape(o.symbol, o.day)
    first = int(own["time"].iloc[0])
    before = np.searchsorted(t, first, side="left")                 # strictly earlier trades
    lo = np.searchsorted(t, first - 2000, side="left")
    if before - lo < 1:
        continue
    seg_p, seg_m = p[lo:before], buyer_maker[lo:before]
    buys, sells = seg_p[~seg_m], seg_p[seg_m]
    mid = (buys[-1] + sells[-1]) / 2 if len(buys) and len(sells) else seg_p[-1]
    sign = 1 if o.side == "BUY" else -1
    avg, notional = float(o.avgPrice), float(o.cumQuote)
    rows.append({"symbol": o.symbol, "notional": notional, "fills": len(own), "client": str(o.clientOrderId)[:3],
                 "slip_bp": sign * (avg - mid) / mid * 1e4,
                 "worst_fill_bp": sign * ((own["price"].max() if sign > 0 else own["price"].min()) - mid) / mid * 1e4,
                 "fee_bp": own["commission"].sum() / own["quoteQty"].sum() * 1e4,
                 "delay_ms": first - int(o.time)})
df = pd.DataFrame(rows)


def stats(part):
    s = part["slip_bp"]
    return {"orders": int(len(part)), "median_notional": round(float(part["notional"].median())),
            "slip_bp_mean": round(float(s.mean()), 2), "slip_bp_median": round(float(s.median()), 2),
            "slip_bp_p90": round(float(s.quantile(0.9)), 2), "slip_bp_max": round(float(s.max()), 2),
            "worst_fill_bp_p90": round(float(part["worst_fill_bp"].quantile(0.9)), 2),
            "fills_per_order_median": float(part["fills"].median()), "fills_per_order_max": int(part["fills"].max())}


person = df[df["client"] != "aut"]                      # the exchange's own closes (liquidations) apart
out = {"orders_measured": int(len(df)), "of_which_exchange_closes": int((df["client"] == "aut").sum()),
       "all_person_orders": stats(person), "by_notional": {}, "by_symbol": {},
       "fee_bp": {"median": round(float(person["fee_bp"].median()), 3), "min": round(float(person["fee_bp"].min()), 3),
                  "max": round(float(person["fee_bp"].max()), 3)},
       "order_to_first_fill_ms": {"median": float(person["delay_ms"].median()),
                                  "p90": float(person["delay_ms"].quantile(0.9))}}
for name, lo, hi in (("under 500 USDT", 0, 500), ("500 to 2,000", 500, 2000), ("2,000 to 8,000", 2000, 8000),
                     ("over 8,000", 8000, 1e12)):
    part = person[(person["notional"] >= lo) & (person["notional"] < hi)]
    if len(part):
        out["by_notional"][name] = stats(part)
for symbol, part in person.groupby("symbol"):
    out["by_symbol"][symbol] = stats(part)
if (df["client"] == "aut").any():
    out["exchange_closes"] = stats(df[df["client"] == "aut"])
Path(__file__).with_name("real_fills.json").write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps(out, indent=1))
