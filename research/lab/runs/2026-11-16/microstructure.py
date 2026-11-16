"""Backtest assumptions against the real tape (2026-11-16): Binance's public aggregated trades of six contracts on
four days (two quiet, two violent), in data_files/binance_aggtrades_2026-11-12 (checksums verified).

1. Hourly entry: the backtest fills a market order at the next bar's open, 3 bp worse. Here a small market order
   sent 5 s after each hour starts fills at the first trade whose taker is on its side; a buy and a sell together
   give the effective half-spread (price drift cancels) and the drift's size.
2. Stop: the backtest fills a stop at the trigger, 10 bp worse. Here, for every level on a 0.5 % grid under (over)
   each hour's open, the first trade through the level triggers and the stop fills at the next seller- (buyer-)
   initiated trade.
3. Delay: how far the price is from the hour's open after 2, 5 and 10 seconds.
Small orders only (a 30 to 150 USDT order takes the top of the book); nothing here measures market impact.
"""
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
DATA = ROOT / "data_files" / "binance_aggtrades_2026-11-12"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "SUIUSDT"]
DAYS = {"2026-08-15": "quiet", "2026-10-03": "quiet", "2026-11-05": "violent", "2026-11-08": "violent"}
H = 3_600_000


def load(symbol, day):
    with zipfile.ZipFile(DATA / f"{symbol}-aggTrades-{day}.zip") as z:
        raw = z.read(z.namelist()[0])
    head = raw[:200].decode().split("\n")[0]
    names = ["id", "price", "qty", "first", "last", "time", "buyer_maker"]
    df = pd.read_csv(io.BytesIO(raw), header=0 if "price" in head else None, names=names)
    maker = df["buyer_maker"].astype(str).str.lower().eq("true").to_numpy()
    return df["time"].to_numpy(np.int64), df["price"].to_numpy(float), maker


def first_at(times, mask_idx, t):
    """Index (into the full arrays) of the first trade of ``mask_idx`` at or after time ``t``; -1 if none."""
    k = np.searchsorted(times[mask_idx], t, side="left")
    return int(mask_idx[k]) if k < len(mask_idx) else -1


def analyse(symbol, day):
    t, p, buyer_maker = load(symbol, day)
    buys, sells = np.nonzero(~buyer_maker)[0], np.nonzero(buyer_maker)[0]      # taker buys / taker sells
    start = (t[0] // H) * H
    entry, delay, stops = [], {2: [], 5: [], 10: []}, []
    for hour in range(start, start + 24 * H, H):
        i0 = np.searchsorted(t, hour, side="left")
        if i0 >= len(t) or t[i0] >= hour + H:
            continue
        open_ = p[i0]
        for secs in delay:
            b, s = first_at(t, buys, hour + secs * 1000), first_at(t, sells, hour + secs * 1000)
            if b < 0 or s < 0 or max(t[b], t[s]) > hour + secs * 1000 + 60_000:
                continue
            mid = (p[b] + p[s]) / 2
            delay[secs].append(abs(mid / open_ - 1) * 1e4)
            if secs == 5:
                entry.append(((p[b] - p[s]) / 2 / open_ * 1e4, (p[b] / open_ - 1) * 1e4, (1 - p[s] / open_) * 1e4))
        i1 = np.searchsorted(t, hour + H, side="left")
        seg_p, seg_t = p[i0:i1], t[i0:i1]
        for side in (-1, 1):                                               # -1: sell stops under the open
            for k in range(1, 21):
                level = open_ * (1 + side * 0.005 * k)
                hit = np.nonzero(seg_p <= level)[0] if side < 0 else np.nonzero(seg_p >= level)[0]
                if not len(hit):
                    break
                j = i0 + int(hit[0])
                pool = sells if side < 0 else buys
                n = np.searchsorted(pool, j, side="right")
                if n >= len(pool):
                    break
                fill = p[pool[n]]
                slip = (level - fill) / level * 1e4 if side < 0 else (fill - level) / level * 1e4
                before = np.searchsorted(seg_t, t[j] - 60_000, side="left")
                speed = (p[j] / seg_p[before] - 1) * 100 if before < len(seg_p) else 0.0
                stops.append((slip, abs(speed), (t[pool[n]] - t[j])))
    return entry, delay, stops


def q(values, *qs):
    a = np.asarray(values, float)
    return [round(float(np.quantile(a, x)), 2) for x in qs] if len(a) else None


out = {}
for symbol in SYMBOLS:
    for kind in ("quiet", "violent"):
        entry, delay, stops = [], {2: [], 5: [], 10: []}, []
        for day, k in DAYS.items():
            if k != kind:
                continue
            e, d, s = analyse(symbol, day)
            entry += e
            stops += s
            for secs in d:
                delay[secs] += d[secs]
        e = np.array(entry) if entry else np.zeros((0, 3))
        s = np.array(stops) if stops else np.zeros((0, 3))
        fast = s[s[:, 1] >= 1.0] if len(s) else s
        out[f"{symbol} {kind}"] = {
            "hours": len(e),
            "half_spread_bp": {"median": q(e[:, 0], 0.5), "p90": q(e[:, 0], 0.9),
                               "mean": round(float(e[:, 0].mean()), 2)},
            "round_trip_side_cost_bp_mean": round(float((e[:, 1].mean() + e[:, 2].mean()) / 2), 2),
            "drift_abs_bp_median": {str(k): q(v, 0.5) for k, v in delay.items()},
            "drift_abs_bp_p90": {str(k): q(v, 0.9) for k, v in delay.items()},
            "stops": {"n": int(len(s)), "slip_bp": {"mean": round(float(s[:, 0].mean()), 2) if len(s) else None,
                                                    "median_p90_p99_max": q(s[:, 0], 0.5, 0.9, 0.99, 1.0)},
                      "fast_moves_n": int(len(fast)),
                      "fast_slip_bp": {"mean": round(float(fast[:, 0].mean()), 2) if len(fast) else None,
                                       "median_p90_p99_max": q(fast[:, 0], 0.5, 0.9, 0.99, 1.0)}}}
        print(f"{symbol} {kind}", json.dumps(out[f"{symbol} {kind}"]), flush=True)
Path(__file__).with_name("microstructure.json").write_text(json.dumps(
    {"data": "data_files/binance_aggtrades_2026-11-12 (data.binance.vision, aggTrades, sha256 verified)",
     "days": DAYS, "results": out}, indent=1) + "\n")
