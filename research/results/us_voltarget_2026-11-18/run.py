"""Volatility-sized holding of SPY, QQQ and GLD on the perpetuals' costs.

Protocol: research/protocols/2026-11-18-us-voltarget.md, committed before any result. One rule, three tests on data
it was not chosen on; one step:

    python research/results/us_voltarget_2026-11-18/run.py

Writes results.json next to this file and appends the one trial to research/trials.jsonl.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from lab import trials as registry  # noqa: E402
from lab import um_lab as lab  # noqa: E402
from lab import us_equity as us  # noqa: E402

DAY = us.DAY
U3, NAV, CAPITAL = ("SPY", "QQQ", "GLD"), {"GLD"}, 150.0
LONG, INDEX = ("SPY", "GLD"), ("NASDAQCOM",)
FRED_SNAP = ROOT / "data_files" / "us_index_fred_2026-11-12"
PAUSE_MS = 4 * 3_600_000


def ms(day: str) -> int:
    return int(pd.Timestamp(day, tz="UTC").value // 10**6)


HOLDOUT, START_U = (ms("2024-11-17"), ms("2026-11-13")), ms("2016-11-17")
LONG_WINDOW, START_LONG = (ms("2005-11-18"), ms("2016-11-17")), ms("2004-11-18")
NAS_WINDOW, START_INDEX = (ms("1972-02-07"), ms("2016-11-17")), ms("1971-02-05")


def vol20(close: pd.Series) -> pd.Series:
    return close.pct_change().rolling(20).std() * math.sqrt(252)


def rule(close: pd.Series) -> pd.Series:
    return (0.15 / vol20(close)).clip(upper=1.0).fillna(0.0)


def hold(close: pd.Series) -> pd.Series:
    return pd.Series(1.0, index=close.index)


def kinds(t: str, group: str) -> tuple[dict, dict]:
    if group == "index":
        return {"snapshot": FRED_SNAP, "index": True}, {"us_snapshot": FRED_SNAP, "index": True}
    nav = group == "long" or t in NAV
    return {"nav": nav}, {"nav": nav}


def account(names, weight_of, *, group: str, start: int, capital: float, stop: bool, stress: bool = False) -> pd.Series:
    """Calendar-day returns (weekends 0) of 1/N slots replayed from ``start``."""
    curves = []
    for t in names:
        frame_kw, replay_kw = kinds(t, group)
        df = us.daily_frame(t, **frame_kw)
        rate = 0.10 / 365 if stress else max(us.daily_funding(us.perpetual(t)) or 0.0, 0.0)
        kw = {"cost_multiplier": 2 if stress else 1, "daily_rate": rate}
        if stop:
            kw.update(stop_pct="0.15", pause_ms=PAUSE_MS, reenter_after_stop="pause")
        r = us.replay_daily(t, weight_of(df["close"]), start=start, starting_balance=str(capital / len(names)),
                            **replay_kw, **kw)
        curves.append(r["curve"])
    daily = lab.account_daily(curves, slots=len(names), slot_capital=capital / len(names))
    return daily.reindex(range(int(daily.index.min()), int(daily.index.max()) + 1)).fillna(0.0)


def cut(daily: pd.Series, window: tuple[int, int]) -> pd.Series:
    return daily[(daily.index >= window[0] // DAY) & (daily.index < window[1] // DAY)]


def compare(names, *, group: str, start: int, capital: float, window: tuple[int, int]) -> dict:
    kw = {"group": group, "start": start, "capital": capital}
    return {"rule": lab.summary(cut(account(names, rule, stop=True, **kw), window)),
            "hold": lab.summary(cut(account(names, hold, stop=False, **kw), window))}


def commit() -> str:
    return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


def main() -> None:
    holdout = compare(U3, group="u", start=START_U, capital=CAPITAL, window=HOLDOUT)
    stress = lab.summary(cut(account(U3, rule, group="u", start=START_U, capital=CAPITAL, stop=True, stress=True),
                             HOLDOUT))
    long_check = compare(LONG, group="long", start=START_LONG, capital=1000.0, window=LONG_WINDOW)
    nasdaq = compare(INDEX, group="index", start=START_INDEX, capital=1000.0, window=NAS_WINDOW)
    weights = {}
    for t in U3:
        w = rule(us.daily_frame(t, nav=t in NAV)["close"])
        w = w[(w.index >= HOLDOUT[0]) & (w.index < HOLDOUT[1])]
        small = (w > 0) & (w * CAPITAL / len(U3) < 5)                       # a position under 5 USDT: not placed
        weights[t] = {"mean": round(float(w.mean()), 3),
                      "share_of_days_below_min_notional": round(float(small.mean()), 3)}
    better = lambda c: c["rule"]["sharpe"] >= c["hold"]["sharpe"]           # noqa: E731
    checks = {"holdout_sharpe_and_drawdown": better(holdout)
              and holdout["rule"]["max_drawdown"] <= holdout["hold"]["max_drawdown"],
              "holdout_stress_return_above_0": stress["total_return"] > 0,
              "long_checks_sharpe_and_drawdown": all(better(c) and c["rule"]["max_drawdown"] < c["hold"]["max_drawdown"]
                                                     for c in (long_check, nasdaq))}
    registry.record([{"family": "us_voltarget_2026-11-18", "study": "us_voltarget_2026-11-18", "trial": "vt15_stop15",
                      "config": {"target": 0.15, "stop": "0.15", "after_stop": "pause", "names": list(U3)},
                      "window": ["2024-11-17", "2026-11-12"], "sharpe": holdout["rule"]["sharpe"], "count": 1,
                      "note": "chosen after the trend study's folds; tested on its unopened holdout and two long "
                              "checks"}],
                    path=ROOT / "research/trials.jsonl")
    out = {"code_commit": commit(), "holdout": holdout, "holdout_stress": stress,
           "long_check_spy_gld": long_check, "long_check_nasdaq_composite": nasdaq,
           "report_only": {"weights_holdout": weights,
                           "cagr_gap_holdout": holdout["rule"]["cagr"] - holdout["hold"]["cagr"]},
           "checks": checks, "passes": all(checks.values())}
    (HERE / "results.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
