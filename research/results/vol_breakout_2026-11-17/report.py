"""Report-only diagnostic of the breakout study, written AFTER its results (nothing here changes a verdict):

run.py's account adds every contract's money made to 1500 USDT, each contract replayed from one slot of 150. With
hundreds of contracts the losses add up past the capital: the account's money goes below zero and its daily returns
(a change over a negative base) no longer mean anything, so the Sharpe, drawdown and yearly figures of results.json
are not valid from that day on. The verdicts stand: none rests on those figures being right (every configuration
fails "total return > 0", and the main test uses each trade's own return).

This script replays the four configurations once more at 1x costs (the same configurations, already registered;
no new trial) and reports: the money made in USDT, the first day the account's money is at or below zero, and an
approximation with compounding (each trade's stake is a tenth of the money realized when it opens; the trade's
return as replayed). Writes report.json.
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE))
import run as R  # noqa: E402

OUT = HERE / "report.json"


def compounding(trades: list, returns: dict, *, capital: float, slots: int) -> dict:
    """Realized money when each trade stakes 1/slots of the money realized at its entry."""
    events = sorted([(t.entry, 1, t) for t in trades] + [(t.exit, 0, t) for t in trades], key=lambda e: (e[0], e[1]))
    money, stake, path = capital, {}, []
    for at, kind, t in events:
        if kind == 1:
            stake[t] = max(money, 0.0) / slots
        else:
            money += stake.pop(t) * returns[t]
            path.append((at, money))
    values = np.array([capital] + [m for _, m in path])
    peak = np.maximum.accumulate(values)
    return {"final_money": float(values[-1]), "total_return": float(values[-1] / capital - 1),
            "max_drawdown": float(((peak - values) / peak).max())}


def main() -> None:
    times = np.arange(R.EVAL_START + R.DAY, R.DEV_END + 1, R.DAY, dtype=np.int64)
    panel, y, _ = R.P.build_panel(times)
    panel["log_rv_30"] = np.log(panel["rv_30"])
    x = panel[list(R.P.ps.PANEL_FEATURES) + ["log_rv_30"]].astype(np.float32)
    preds = R.predictions(panel, y, x)
    passed, sizes_s1 = R.vb.top_share(preds["trend_14"]), R.vb.vol_sizes(preds["fwd_vol_7"])
    members = panel[["symbol"]].assign(time=panel.index.to_numpy(dtype=np.int64))
    members = members[members["time"] >= R.TEST_START]
    trades, volumes, first = [], {}, {}
    for symbol, rows in members.groupby("symbol"):
        h, _ = R.P.ps.load_symbol(R.SNAP, symbol, R.DEV_END)
        allowed = pd.Series(True, index=pd.Index(rows["time"].to_numpy(dtype=np.int64)))
        trades += R.vb.breakout_trades(symbol, R.fp.daily(h), allowed=allowed)
        volumes[symbol], first[symbol] = R.P.ps.daily_volume(h), int(h["open_time"].iloc[0])
    medians = R.vb.median_volumes(volumes, first, times)
    priority = {(t.symbol, t.entry): float(medians.at[t.entry, t.symbol]) for t in trades}
    accepted = {t: t.symbol in passed.get(t.entry, set()) for t in trades}
    chosen = {"none": R.vb.allocate(trades, lambda t: priority[(t.symbol, t.entry)]),
              "trend_top20": R.vb.allocate([t for t in trades if accepted[t]],
                                           lambda t: priority[(t.symbol, t.entry)])}
    out = {}
    with ProcessPoolExecutor(2) as pool:
        for name, (filt, how) in R.CONFIGS.items():
            ts = chosen[filt]
            sz = [R.ON if how == "equal" else sizes_s1.get((t.symbol, t.entry), R.ON) for t in ts]
            res = R.run_config(ts, sz, 1, R.SLOT, R.CAPITAL, pool)
            money = R.CAPITAL * (1 + res["daily"]).cumprod()
            made = float(sum(res["trade_returns"].values()) * R.SLOT)
            below = money[money <= 0]
            day = str(pd.Timestamp(int(below.index[0]) * R.DAY, unit="ms").date()) if len(below) else None
            out[name] = {"money_made_usdt_fixed_150": made, "first_day_money_at_or_below_0": day,
                         "lowest_money_usdt": float(money.min()),
                         "mean_trade_return": float(np.mean(list(res["trade_returns"].values()))),
                         "compounding_approximation": compounding(ts, res["trade_returns"], capital=R.CAPITAL,
                                                                  slots=R.vb.SLOTS)}
            print(name, json.dumps(out[name]), flush=True)
    OUT.write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
