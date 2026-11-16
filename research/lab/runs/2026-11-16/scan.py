"""Scout scan (2026-11-16 run): what every USDT perpetual looked like, by regime, on the selection window only
(2022-11-13..2026-05-12), plus a three-line description of the last 30 days of the snapshot (background only; it
is inside the holdout window and chooses nothing). Reads the snapshot; writes scan.json."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from lab import um_lab as lab  # noqa: E402

DAY, H = lab.DAY, lab.H
out = Path(__file__).with_name("scan.json")
btc = lab.frame("BTCUSDT")
regime = lab.regimes(btc)                                   # per day: bear / chop / other, known at the day's close
sel_days = (lab.START // DAY, lab.SELECT_END // DAY)
rows, recent = [], []
for sym in lab.contracts():
    df = pd.read_csv(lab.SNAPSHOT / f"{sym}_1h.csv.gz", usecols=["open_time", "close", "quote_volume"]).set_index("open_time")  # noqa: E501
    day = df.index // DAY
    close = df["close"].groupby(day).last()
    quote = df["quote_volume"].groupby(day).sum()
    ret = close.pct_change()
    fpath = lab.SNAPSHOT / f"{sym}_funding.csv.gz"
    fund = pd.read_csv(fpath) if fpath.is_file() else pd.DataFrame({"funding_time": [], "rate": []})
    fday = fund.groupby(fund["funding_time"] // DAY)["rate"].sum() if len(fund) else pd.Series(dtype=float)
    d = pd.DataFrame({"ret": ret, "next": ret.shift(-1), "mom30": close / close.shift(30) - 1,
                      "rev1": ret, "vol30": ret.rolling(30).std(), "quote": quote,
                      "fund7": fday.reindex(close.index).fillna(0).rolling(7).sum(),
                      "volsurge": quote / quote.rolling(30).median().shift(1)})
    d["sym"] = sym
    d["regime"] = regime.reindex(d.index)
    sel = d[(d.index >= sel_days[0]) & (d.index < sel_days[1])].dropna(subset=["next", "mom30", "vol30"])
    rows.append(sel)
    last = d[d.index >= (lab.END // DAY) - 30]
    if len(last) >= 25 and len(close) > 31:
        recent.append({"sym": sym, "ret30": float(close.iloc[-1] / close.iloc[-31] - 1),
                       "fund30": float(fday[fday.index >= (lab.END // DAY) - 30].sum()),
                       "quote": float(quote.iloc[-30:].median())})
panel = pd.concat(rows)
panel = panel[panel["quote"] >= 5e6]                        # a contract-day counts when it traded 5M USDT that day


def spread(col, by=None):
    """Mean next-day return of the top fifth minus the bottom fifth of ``col`` within each day (per cent)."""
    def one(g):
        if len(g) < 20:
            return np.nan
        q = g[col].rank(pct=True)
        return g.loc[q > 0.8, "next"].mean() - g.loc[q < 0.2, "next"].mean()
    daily = panel.groupby(level=0).apply(one).dropna()
    res = {"all": [float(daily.mean() * 100), float(daily.mean() / daily.std() * np.sqrt(365)), int(len(daily))]}
    lab_reg = regime.reindex(daily.index)
    for name in ("bear", "chop", "other"):
        part = daily[lab_reg == name]
        res[name] = [float(part.mean() * 100), float(part.mean() / part.std() * np.sqrt(365)) if len(part) > 30 else None,  # noqa: E501
                     int(len(part))]
    return res


scan = {"window": "2022-11-13..2026-05-12 (selection only)", "contract_days": int(len(panel)),
        "contracts": int(panel["sym"].nunique()),
        "regime_days": {k: int(v) for k, v in regime[(regime.index >= sel_days[0]) & (regime.index < sel_days[1])].value_counts().items()},  # noqa: E501
        "note": "top-minus-bottom fifth, next-day return in per cent per day, annualised ratio, days; gross, no costs",
        "momentum_30d": spread("mom30"), "reversal_1d": spread("rev1"), "funding_7d": spread("fund7"),
        "volatility_30d": spread("vol30"), "volume_surge": spread("volsurge")}
market = panel.groupby(level=0)["ret"].mean()
scan["equal_weight_market_by_regime_pct_per_day"] = {
    name: float(market[regime.reindex(market.index) == name].mean() * 100) for name in ("bear", "chop", "other")}
r = pd.DataFrame(recent)
liquid = r[r["quote"] >= 5e6]
scan["last_30_days_background"] = {
    "as_of": "2026-11-12", "btc_30d_return": float(btc["close"].iloc[-1] / btc["close"].iloc[-1 - 30 * 24] - 1),
    "btc_regime": str(regime.iloc[-1]), "contracts_liquid": int(len(liquid)),
    "share_up_30d": float((liquid["ret30"] > 0).mean()), "median_30d_return": float(liquid["ret30"].median()),
    "median_30d_funding": float(liquid["fund30"].median()),
    "btc_30d_realised_vol_annual": float(btc["close"].pct_change().iloc[-720:].std() * np.sqrt(24 * 365))}
out.write_text(json.dumps(scan, indent=1) + "\n")
print(json.dumps(scan, indent=1))
