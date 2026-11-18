"""The US-equity trend study (research/protocols/2026-11-18-us-equity-trend.md, committed before any result).

    python research/results/us_trend_2026-11-18/run.py screen   # 32 configurations, benchmarks, the line
    python research/results/us_trend_2026-11-18/run.py final    # survivors only: the holdout, once

Writes results.json next to this file and appends the 32 trials to research/trials.jsonl.
"""
from __future__ import annotations

import itertools
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from lab import trials as registry  # noqa: E402
from lab import um_judge as judge  # noqa: E402
from lab import um_lab as lab  # noqa: E402
from lab import us_equity as us  # noqa: E402
from lab.metrics import deflated_sharpe_probability_k  # noqa: E402

DAY = us.DAY
CAPITAL = 1000.0
U8 = ("SPY", "QQQ", "GLD", "NVDA", "TSLA", "AAPL", "GOOGL", "META")
NAV = {"GLD"}
LONG = ("SPY", "GLD")
FAMILY, STUDY = "us_equity_trend_2026-11-18", "us_trend_2026-11-18"
PAUSE_MS = 4 * 3_600_000


def ms(day: str) -> int:
    return int(pd.Timestamp(day, tz="UTC").value // 10**6)


PERIODS = [(ms("2017-11-17"), ms("2019-11-17"))] + \
    [(ms(f"{y}-11-17"), ms(f"{y + 1}-11-17")) for y in range(2019, 2024)] + [(ms("2024-11-17"), ms("2026-11-13"))]
FOLDS, HOLDOUT = [1, 2, 3, 4, 5], 6
LONG_WINDOW = (ms("2005-11-18"), ms("2016-11-17"))
# Amendment of 2026-11-18 (research/protocols/2026-11-18-us-equity-trend-amendment.md, before any result):
INDEX, FRED_SNAP = ("NASDAQCOM",), ROOT / "data_files" / "us_index_fred_2026-11-12"
NAS_WINDOW = (ms("1972-02-07"), ms("2016-11-17"))    # after a year of warm-up; before the 10 years
SMALL, SMALL_CAPITAL = ("SPY", "QQQ", "GLD"), 150.0


# -- the rules ---------------------------------------------------------------------------------------------------
def vol20(close: pd.Series) -> pd.Series:
    return close.pct_change().rolling(20).std() * math.sqrt(252)


def weights(close: pd.Series, rule: str, n: int, target: float) -> pd.Series:
    """The weight decided at each day's close from closes up to that day."""
    on = close > close.rolling(n).mean() if rule == "sma" else close / close.shift(n) - 1 > 0
    size = (target / vol20(close)).clip(upper=1.0)
    w = (on.astype(float) * size).where(close.rolling(n + 1).count() > n, 0.0)
    return w.fillna(0.0)


def bench_weights(close: pd.Series, kind: str) -> pd.Series:
    return pd.Series(1.0, index=close.index) if kind == "hold" else (0.15 / vol20(close)).clip(upper=1.0).fillna(0.0)


CONFIGS = [{"rule": r, "n": n, "target": t, "leverage": lev, "stop": s}
           for r, ns in (("sma", (100, 200)), ("tsmom", (126, 252))) for n in ns
           for t, lev, s in itertools.product((0.10, 0.20), ("1", "2"), ("0.08", "0.15"))]


def label(c: dict) -> str:
    return f"{c['rule']}{c['n']}_t{int(c['target'] * 100)}_x{c['leverage']}_s{c['stop']}"


# -- replays -----------------------------------------------------------------------------------------------------
def base_rate(ticker: str) -> float:
    return max(us.daily_funding(us.perpetual(ticker)) or 0.0, 0.0)


START_U8, START_LONG, START_INDEX = ms("2016-11-17"), ms("2004-11-18"), ms("1971-02-05")   # slots start together


def source(t: str, names) -> tuple[dict, dict, int]:
    """(daily_frame keywords, replay_daily keywords, start) of one name in this group."""
    if t in INDEX:
        return {"snapshot": FRED_SNAP, "index": True}, {"us_snapshot": FRED_SNAP, "index": True}, START_INDEX
    long = tuple(names) == LONG
    if long or t in NAV:
        return {"nav": True}, {"nav": True}, START_LONG if long else START_U8
    return {}, {}, START_U8


def account(names, weight_of, *, stop="0", leverage="1", stress=False, capital=CAPITAL) -> dict:
    """Replay each name from the same day with an equal slot, combine 1/N; returns calendar-day returns (weekends
    0), the round trips and the times of the exchange stops. The signals read the whole history before the start."""
    curves, trips, stops = [], [], []
    for t in names:
        frame_kw, replay_kw, start = source(t, names)
        df = us.daily_frame(t, **frame_kw)
        kw = {"cost_multiplier": 2, "daily_rate": 0.10 / 365} if stress else {"daily_rate": base_rate(t)}
        if stop != "0":
            kw.update(stop_pct=stop, pause_ms=PAUSE_MS, reenter_after_stop="pause_then_reset")
        r = us.replay_daily(t, weight_of(df["close"]), leverage=leverage, start=start,
                            starting_balance=str(capital / len(names)), **replay_kw, **kw)
        curves.append(r["curve"])
        trips += lab.round_trips(r["fills"])
        stops += [int(f["at"]) for f in r["fills"] if f["order"] == "protective_stop"]
    daily = lab.account_daily(curves, slots=len(names), slot_capital=capital / len(names))
    full = daily.reindex(range(int(daily.index.min()), int(daily.index.max()) + 1)).fillna(0.0)
    return {"daily": full, "trips": trips, "stops": stops}


def cut(result: dict, start: int, end: int) -> dict:
    d = result["daily"]
    return {"daily": d[(d.index >= start // DAY) & (d.index < end // DAY)],
            "trips": [x for x in result["trips"] if start <= x[0] < end], "entries": [],
            "stops": sum(1 for x in result["stops"] if start <= x < end)}


def per_period(result: dict) -> dict:
    return {k: cut(result, a, b) for k, (a, b) in enumerate(PERIODS)}


def config_run(c: dict, *, stress=False, names=U8, capital=CAPITAL) -> dict:
    return account(names, lambda close: weights(close, c["rule"], c["n"], c["target"]), stop=c["stop"],
                   leverage=c["leverage"], stress=stress, capital=capital)


def concentration(chosen: list[dict]) -> dict:
    """Report only: the 8 underlyings' daily-return correlations over the selection and folds, and each fold's mean
    target weight per name for the configuration chosen for it."""
    closes = pd.concat({t: us.daily_frame(t, nav=t in NAV)["close"] for t in U8}, axis=1).sort_index()
    window = closes[(closes.index >= PERIODS[0][0]) & (closes.index < PERIODS[FOLDS[-1]][1])]
    corr = window.pct_change(fill_method=None).corr().round(2)
    exposure = {}
    for c, k in zip(chosen, FOLDS):
        a, b = PERIODS[k]
        exposure[str(k)] = {t: round(float(weights(closes[t].dropna(), c["rule"], c["n"], c["target"])
                                           .loc[a:b - 1].mean() * float(c["leverage"])), 3) for t in U8}
    return {"correlation": {t: corr[t].to_dict() for t in U8}, "mean_target_weight_by_fold": exposure}


def commit() -> str:
    return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


def joined(periods: dict, ks) -> pd.Series:
    return pd.concat([periods[k]["daily"] for k in ks]) if ks else pd.Series(dtype=float)


def screen() -> dict:
    runs = [(judge.Config("us_trend", c), per_period(config_run(c))) for c in CONFIGS]
    bench = {k: per_period(account(U8, lambda close, k=k: bench_weights(close, k))) for k in ("hold", "vol")}
    wf = judge.walk_forward(runs, FOLDS)
    wf_daily = wf["daily"]
    s = lab.summary(wf_daily)
    bars = {k: lab.summary(joined(v, FOLDS)) for k, v in bench.items()}
    selection = [lab.summary(p[0]["daily"]).get("sharpe", 0.0) for _, p in runs]
    std = float(np.std(selection, ddof=1))
    prob = deflated_sharpe_probability_k(wf_daily, std, len(CONFIGS))
    chosen = [c.params for c in wf["chosen"]]
    stress_runs = {label(c): per_period(config_run(c, stress=True)) for c in {label(c): c for c in chosen}.values()}
    stressed = pd.concat([stress_runs[label(c)][k]["daily"] for c, k in zip(chosen, FOLDS)])
    trips = len(wf["trips"])
    final = wf["final"].params
    long_run = cut(config_run(final, names=LONG), *LONG_WINDOW)
    long_hold = cut(account(LONG, lambda close: bench_weights(close, "hold")), *LONG_WINDOW)
    long_s, long_b = lab.summary(long_run["daily"]), lab.summary(long_hold["daily"])
    nas_s = lab.summary(cut(config_run(final, names=INDEX), *NAS_WINDOW)["daily"])
    nas_b = lab.summary(cut(account(INDEX, lambda close: bench_weights(close, "hold")), *NAS_WINDOW)["daily"])
    small = {f"{int(cap)}_usdt": lab.summary(joined(per_period(config_run(final, names=SMALL, capital=cap)), FOLDS))
             for cap in (CAPITAL, SMALL_CAPITAL)}
    checks = {"return_above_0": s["total_return"] > 0, "beats_hold": s["sharpe"] > bars["hold"]["sharpe"],
              "beats_vol_target": s["sharpe"] > bars["vol"]["sharpe"], "deflated_at_least_0.5": prob >= 0.5,
              "stress_return_above_0": lab.summary(stressed)["total_return"] > 0, "at_least_50_trades": trips >= 50,
              "long_check_not_below_hold": long_s["sharpe"] >= long_b["sharpe"] and nas_s["sharpe"] >= nas_b["sharpe"]}
    rows = [{"family": FAMILY, "study": STUDY, "trial": label(c), "config": c,
             "window": [PERIODS[0][0], PERIODS[FOLDS[-1]][1]],
             "sharpe": lab.summary(joined(p, [0] + FOLDS)).get("sharpe", 0.0), "count": 1,
             "note": "selection + folds, base costs; holdout not read"} for c, (_, p) in zip(CONFIGS, runs)]
    registry.record(rows, path=ROOT / "research/trials.jsonl")
    return {"configs": {label(c): {"selection": lab.summary(p[0]["daily"]), "folds": lab.summary(joined(p, FOLDS)),
                                   "round_trips_folds": sum(len(p[k]["trips"]) for k in FOLDS)}
                        for c, (_, p) in zip(CONFIGS, runs)},
            "benchmarks_folds": bars,
            "benchmarks_by_fold": {k: {f: lab.summary(v[f]["daily"]) for f in FOLDS} for k, v in bench.items()},
            "walk_forward": {"chosen": [label(c) for c in chosen], "final": label(final), "summary": s,
                             "deflated_probability": prob, "selection_sharpe_std": std, "n_trials": len(CONFIGS),
                             "stress": lab.summary(stressed), "round_trips": trips},
            "long_check": {"spy_gld": {"window": ["2005-11-18", "2016-11-16"], "final_rule": long_s, "hold": long_b},
                           "nasdaq_composite": {"window": ["1972-02-07", "2016-11-16"], "final_rule": nas_s,
                                                "hold": nas_b}},
            "small_capital_check": {"names": list(SMALL), "folds": small},
            "concentration": concentration(chosen),
            "checks": checks, "survives": all(checks.values())}


def no_lookahead(c: dict) -> bool:
    close = us.daily_frame("SPY")["close"]
    full = weights(close, c["rule"], c["n"], c["target"])
    for i in (len(close) // 3, len(close) // 2, 2 * len(close) // 3):
        later = close.copy()
        later.iloc[i + 1:] *= 3.0
        if not weights(close.iloc[: i + 1], c["rule"], c["n"], c["target"]).equals(full.iloc[: i + 1]):
            return False
        if not weights(later, c["rule"], c["n"], c["target"]).iloc[: i + 1].equals(full.iloc[: i + 1]):
            return False
    return True


def final(screened: dict) -> dict:
    if not screened["survives"]:
        return {"opened_holdout": False, "reason": "no configuration passed the line"}
    choice = next(c for c in CONFIGS if label(c) == screened["walk_forward"]["final"])
    unseen = per_period(config_run(choice))[HOLDOUT]["daily"]
    bench = {k: lab.summary(per_period(account(U8, lambda close, k=k: bench_weights(close, k)))[HOLDOUT]["daily"])
             for k in ("hold", "vol")}
    keys = ("rule", "n", "target", "leverage", "stop")
    neighbours = [label(c) for c in CONFIGS
                  if sum(c[k] != choice[k] for k in keys) == 1 and c["rule"] == choice["rule"]]
    ok_neighbours = all(screened["configs"][n]["folds"]["total_return"] > 0 for n in neighbours)
    s = lab.summary(unseen)
    checks = {"holdout_return_above_0": s["total_return"] > 0, "neighbours_return_above_0": ok_neighbours,
              "no_lookahead": no_lookahead(choice)}
    return {"opened_holdout": True, "choice": label(choice), "holdout": s, "holdout_benchmarks": bench,
            "neighbours": neighbours, "checks": checks, "passes": all(checks.values())}


def main(step: str) -> None:
    path = HERE / "results.json"
    if step == "screen":
        out = {"code_commit": commit(), "screen": screen()}
        path.write_text(json.dumps(out, indent=1, default=str))
    elif step == "final":
        out = json.loads(path.read_text())
        out["final"] = final(out["screen"])
        out["final_commit"] = commit()
        path.write_text(json.dumps(out, indent=1, default=str))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "")
