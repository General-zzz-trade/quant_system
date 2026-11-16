"""Judge (2026-11-16 run): every idea of ideas.py under the live terms, as ideas.md says before any result.

    python judge.py universe      # the yearly symbol lists -> universe.json
    python judge.py screen        # every parameter set on the selection periods -> cache, trials, screen.json
    python judge.py final         # the final check of the survivors (the holdout is read here only) -> results.json

On 2026-11-16 no idea survived the screen, so ``final`` was not run and the holdout was not read; results.json was put
together from screen.json and the two diagnostic screens (LAB_VARIANT=nostop, cap5000), which were then removed.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time
from decimal import Decimal as D
from multiprocessing import get_context
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3]))
sys.path.insert(0, str(HERE))
import ideas as lib  # noqa: E402
from lab import um_lab as lab  # noqa: E402

H, DAY = lab.H, lab.DAY
# Diagnostics only (never a candidate): LAB_VARIANT=nostop drops the exchange stop and its pause; cap5000 gives the
# same lists 5000 USDT. LAB_ONLY limits the ideas.
VARIANT = os.environ.get("LAB_VARIANT", "")
TERMS = lab.LiveTerms(capital="5000") if VARIANT == "cap5000" else lab.LIVE
STOP = "0" if VARIANT == "nostop" else TERMS.stop_pct
ONLY = [n for n in os.environ.get("LAB_ONLY", "").split(",") if n]
SLOTS = 10
SLOT = float(D(TERMS.capital) / SLOTS)
F1, F2, F3 = (f[0] for f in lab.FOLDS)
PERIODS = [(lab.START + 30 * DAY, F1), (F1, F2), (F2, F3), (F3, lab.SELECT_END), (lab.SELECT_END, lab.END)]
STABLE = ("USDCUSDT", "BUSDUSDT", "TUSDUSDT", "FDUSDUSDT", "USDPUSDT", "DAIUSDT")
CACHE = Path("/tmp/lab_2026-11-16" + (f"_{VARIANT}" if VARIANT else ""))
RUN_DATE = "2026-11-16"


def universe() -> list[list[str]]:
    rows = []
    for sym in lab.contracts():
        if sym in STABLE:
            continue
        df = pd.read_csv(lab.SNAPSHOT / f"{sym}_1h.csv.gz", usecols=["open_time", "close", "quote_volume"]).set_index("open_time")  # noqa: E501
        rules = lab.rules(sym)
        for k, (t0, _) in enumerate(PERIODS):
            past = df[(df.index >= t0 - 30 * DAY) & (df.index < t0)]
            daily = past["quote_volume"].groupby(past.index // DAY).sum()
            if len(daily) < 25:
                continue
            smallest = max(float(rules.min_notional), float(rules.step) * float(past["close"].iloc[-1]))
            if smallest <= SLOT * 2 / 3:
                rows.append((k, sym, float(daily.median())))
    table = pd.DataFrame(rows, columns=["period", "symbol", "quote"])
    lists = [table[table["period"] == k].nlargest(SLOTS, "quote")["symbol"].tolist() for k in range(len(PERIODS))]
    (HERE / "universe.json").write_text(json.dumps({
        "rule": f"top {SLOTS} by median daily quote volume of the 30 days before each date; smallest order <= "
                f"{SLOT * 2 / 3:.0f} USDT; no stablecoin contracts",
        "dates": [pd.Timestamp(t0, unit="ms").strftime("%Y-%m-%d") for t0, _ in PERIODS], "lists": lists}, indent=1) + "\n")  # noqa: E501
    return lists


def lists() -> list[list[str]]:
    return json.loads((HERE / "universe.json").read_text())["lists"]


def run_config(job: tuple) -> dict:
    """One parameter set of one idea over ``periods`` (indexes into PERIODS): per period the account's daily
    returns, the round trips, the entry orders and the exchange stops."""
    name, params, cost, periods = job
    fn = lib.IDEAS[name][0]
    btc, all_lists, out, weights = lab.frame("BTCUSDT"), lists(), {}, {}
    for k in periods:
        start, end = PERIODS[k]
        symbols, curves, trips, entries, stops, fills = all_lists[k], [], [], [], 0, 0
        for sym in symbols:
            df = lab.frame(sym)
            key = (sym, tuple(symbols) if name in lib.XS else None)
            if key not in weights:
                ctx = {"params": params, "btc": btc, "symbol": sym, "universe": symbols}
                weights[key] = fn(df, ctx).reindex(df.index).fillna(0.0)
            w = weights[key]
            if not (w[(w.index >= start - H) & (w.index < end)] != 0).any():
                continue
            r = lab.replay(sym, w, cost_multiplier=cost, start=start, end=end, starting_balance=str(SLOT),
                           stop_pct=STOP, pause_ms=TERMS.pause_hours * H if STOP != "0" else None)
            if not len(r["curve"]):
                continue
            curves.append(r["curve"])
            trips += lab.round_trips(r["fills"])
            entries += [(at, n) for at, n, reduce in r["orders"] if not reduce]
            stops += sum(1 for f in r["fills"] if f["order"] == "protective_stop")
            fills += len(r["fills"])
        daily = lab.account_daily(curves, slots=SLOTS, slot_capital=SLOT)
        days = pd.RangeIndex(start // DAY + 1, end // DAY)
        out[k] = {"daily": daily.reindex(days).fillna(0.0), "trips": sorted(trips), "entries": sorted(entries),
                  "stops": stops, "fills": fills}
    return {"idea": name, "params": params, "cost": cost, "periods": out}


def _path(name: str, params: dict, cost: int) -> Path:
    tag = "_".join(f"{k}{v}" for k, v in sorted(params.items()))
    return CACHE / f"{name}__{tag}__x{cost}.pkl"


def run_all(jobs: list[tuple], processes: int = 2) -> None:
    """Run the jobs not in the cache; a cached result with fewer periods is extended, not rerun."""
    CACHE.mkdir(parents=True, exist_ok=True)
    todo = []
    for name, params, cost, periods in jobs:
        have = pickle.loads(_path(name, params, cost).read_bytes())["periods"] if _path(name, params, cost).exists() else {}  # noqa: E501
        missing = tuple(k for k in periods if k not in have)
        if missing:
            todo.append((name, params, cost, missing))
    for sym in sorted({s for lst in lists() for s in lst} | {"BTCUSDT"}):
        lab.frame(sym)                                           # loaded once, shared by the forked workers
    t0, rows = time.time(), []
    with get_context("fork").Pool(processes) as pool:
        for i, res in enumerate(pool.imap_unordered(run_config, todo), 1):
            path = _path(res["idea"], res["params"], res["cost"])
            old = pickle.loads(path.read_bytes())["periods"] if path.exists() else {}
            res["periods"] = {**old, **res["periods"]}
            path.write_bytes(pickle.dumps(res))
            for k, p in res["periods"].items():
                if k not in old:
                    s = lab.summary(p["daily"])
                    rows.append({"date": RUN_DATE, "idea": res["idea"], "params": res["params"],
                                 "cost_multiplier": res["cost"], "terms": f"{TERMS.capital} USDT, 10 slots, yearly list, stop {STOP}"  # noqa: E501
                                 + (f" (diagnostic {VARIANT})" if VARIANT else ""),
                                 "window": [pd.Timestamp(t, unit="ms").strftime("%Y-%m-%d") for t in PERIODS[k]],
                                 "holdout": k == 4, "total_return": s.get("total_return"), "sharpe": s.get("sharpe"),
                                 "max_drawdown": s.get("max_drawdown"), "round_trips": len(p["trips"]),
                                 "fills": p["fills"]})
            print(f"{i}/{len(todo)} {res['idea']} {res['params']} x{res['cost']} {time.time() - t0:.0f}s", flush=True)
    lab.record(rows)


def load(name: str, params: dict, cost: int = 1) -> dict:
    return pickle.loads(_path(name, params, cost).read_bytes())["periods"]


def joined(periods: dict, ks) -> pd.Series:
    return pd.concat([periods[k]["daily"] for k in ks])


def longest_losing(trips: list) -> int:
    run = best = 0
    for _, money in sorted(trips):
        run = run + 1 if money < 0 else 0
        best = max(best, run)
    return best


def eq_weight_hold(ks) -> pd.Series:
    """Buy and hold each period's list in equal parts from the period's first day (no costs, no stops)."""
    parts = []
    for k in ks:
        start, end = PERIODS[k]
        cols = {}
        for sym in lists()[k]:
            c = lab.frame(sym)["close"]
            c = c[(c.index >= start) & (c.index < end)]
            if len(c):
                d = c.groupby(c.index // DAY).last()
                cols[sym] = d / d.iloc[0]
        value = pd.DataFrame(cols).ffill().mean(axis=1)
        parts.append(value.pct_change().dropna())
    return pd.concat(parts)


def score(wf: dict, trips: list, regime: dict) -> dict:
    def band(x, edges):                                        # 1 + number of edges passed
        return 1 + sum(x > e for e in edges)
    bear, chop = regime["bear"], regime["chop"]
    both = 5 if min(bear, chop) > 0.05 else 4 if min(bear, chop) >= 0 else 3 if bear + chop >= 0 \
        else 2 if max(bear, chop) >= 0 else 1
    streak = longest_losing(trips)
    parts = {"return": band(wf["cagr"], (0, 0.10, 0.25, 0.50)),
             "drawdown": 6 - band(wf["max_drawdown"], (0.15, 0.25, 0.35, 0.50)),
             "trades": band(len(trips), (29, 49, 99, 199)),
             "losing_streak": 5 if streak <= 3 else 4 if streak <= 5 else 3 if streak <= 7 else 2 if streak <= 10 else 1,  # noqa: E501
             "bear_and_chop": both}
    return {**parts, "total": sum(parts.values())}


def assemble(name: str, grid: list[dict], cost: int = 1, chosen: list[dict] | None = None) -> dict:
    """Walk-forward of one idea from the cache: for fold k (= period k) the parameter set with the best Sharpe on
    the periods before it (or ``chosen``, to replay the same choices at another cost)."""
    runs = {json.dumps(p, sort_keys=True): load(name, p, cost) for p in grid if _path(name, p, cost).exists()}
    if chosen is None:
        wf = lab.walk_forward([lab.Run(json.loads(key), joined(per, (0, 1, 2, 3))) for key, per in runs.items()])
        chosen, final = wf["chosen_per_fold"], wf["final_params"]
    else:
        final = None
    picks = [runs[json.dumps(p, sort_keys=True)][k] for k, p in zip((1, 2, 3), chosen)]
    daily = pd.concat([p["daily"] for p in picks])
    trips = [t for p in picks for t in p["trips"]]
    entries = [e for p in picks for e in p["entries"]]
    return {"chosen": chosen, "final": final, "daily": daily, "trips": trips, "entries": entries,
            "stops": sum(p["stops"] for p in picks), "runs": runs}


def screen() -> None:
    todo = {n: v for n, v in lib.IDEAS.items() if not ONLY or n in ONLY}
    jobs = [(name, p, 1, (0, 1, 2, 3)) for name, (_, grid) in todo.items() for p in grid]
    run_all(jobs)
    btc = lab.frame("BTCUSDT")
    hold_btc = lab.summary(lab.buy_and_hold("BTCUSDT", F1, lab.SELECT_END))
    hold_list = lab.summary(eq_weight_hold((1, 2, 3)))
    table = {}
    for name, (_, grid) in todo.items():
        a = assemble(name, grid)
        wf, regime = lab.summary(a["daily"]), lab.by_regime(a["daily"], btc)
        streaks = lab.losing_streaks(a["trips"], TERMS.losing_trades)
        survives = wf["total_return"] > 0 and wf["sharpe"] > hold_btc["sharpe"]
        table[name] = {
            "chosen_per_fold": a["chosen"], "final_params": a["final"], "walk_forward": wf, "regime": regime,
            "round_trips": len(a["trips"]), "longest_losing_trades": longest_losing(a["trips"]),
            "exchange_stops": a["stops"], "score": score(wf, a["trips"], regime), "survives": bool(survives),
            "selection_sharpe_by_params": {key: lab.summary(joined(per, (0, 1, 2, 3)))["sharpe"]
                                           for key, per in a["runs"].items()},
            "pilot": {"losing_streak_stops": len(streaks), "per_30_days": len(streaks) * 30 / max(1, len(a["daily"])),
                      "days_below_daily_stop": int((a["daily"] < -TERMS.daily_stop).sum())},
            "limits_needed": lab.limits_needed(a["entries"])}
    (HERE / f"screen{'_' + VARIANT if VARIANT else ''}.json").write_text(json.dumps({"buy_and_hold_btc": hold_btc, "buy_and_hold_list": hold_list,  # noqa: E501
                                                  "ideas": table}, indent=1, default=str) + "\n")
    print("BTC hold", {k: round(v, 3) for k, v in hold_btc.items() if isinstance(v, float)})
    print("list hold", {k: round(v, 3) for k, v in hold_list.items() if isinstance(v, float)})
    for name, r in sorted(table.items(), key=lambda kv: -kv[1]["walk_forward"]["sharpe"]):
        w = r["walk_forward"]
        print(f"{name:26s} ret {w['total_return']:+7.1%} sharpe {w['sharpe']:+5.2f} dd {w['max_drawdown']:5.1%} "
              f"trips {r['round_trips']:4d} streak {r['longest_losing_trades']:2d} bear {r['regime']['bear']:+6.1%} "
              f"chop {r['regime']['chop']:+6.1%} score {r['score']['total']:2d} {'SURVIVES' if r['survives'] else ''}")


def causal(name: str, params: dict) -> list[str]:
    """lab/causality.py on real frames (the first symbols of the 2023 list, 2023-01..2025-01)."""
    from lab.causality import violations
    symbols = lists()[1][:5]
    cut = lambda f: f[(f.index >= lab.ms("2023-01-01")) & (f.index < lab.ms("2025-01-01"))]   # noqa: E731
    frames = {s: cut(lab.frame(s)) for s in symbols + ["BTCUSDT"]}
    original = lab.frame

    def run(inp):
        lib._PANELS.clear()
        lab.frame = lambda s, snapshot=None: inp[s]
        try:
            return lib.IDEAS[name][0](inp[symbols[0]], {"params": params, "btc": inp["BTCUSDT"], "symbol": symbols[0],
                                                        "universe": symbols})
        finally:
            lab.frame = original
    inputs = frames if name in lib.XS else {symbols[0]: frames[symbols[0]], "BTCUSDT": frames["BTCUSDT"]}
    try:
        return violations(run, inputs, [lab.ms("2024-03-05") + 7 * H, lab.ms("2024-09-20") + 23 * H])
    except Exception as exc:
        return [f"{type(exc).__name__}: {exc}"[:200]]
    finally:
        lib._PANELS.clear()


def final() -> None:
    screened = json.loads((HERE / "screen.json").read_text())
    names = [n for n, r in screened["ideas"].items() if r["survives"]]
    jobs = []
    for n in names:
        r = screened["ideas"][n]
        jobs.append((n, r["final_params"], 1, (4,)))
        jobs += [(n, p, 2, (k,)) for k, p in zip((1, 2, 3), r["chosen_per_fold"])]
    run_all(jobs)
    btc = lab.frame("BTCUSDT")
    hold = {"btc": lab.summary(lab.buy_and_hold("BTCUSDT", lab.SELECT_END, lab.END)),
            "list": lab.summary(eq_weight_hold((4,)))}
    wf_hold = screened["buy_and_hold_btc"]
    out = {}
    for n in names:
        r, grid = screened["ideas"][n], lib.IDEAS[n][1]
        h = load(n, r["final_params"], 1)[4]
        hs = lab.summary(h["daily"])
        twice = assemble(n, grid, cost=2, chosen=r["chosen_per_fold"])
        ts = lab.summary(twice["daily"])
        bad = causal(n, r["final_params"])
        streaks = lab.losing_streaks(h["trips"], TERMS.losing_trades)
        checks = {"unseen_data": bool(hs["total_return"] > 0 and hs["sharpe"] > hold["btc"]["sharpe"]),
                  "at_least_50_trades": r["round_trips"] >= 50,
                  "double_fees": bool(ts["total_return"] > 0 and ts["sharpe"] > wf_hold["sharpe"]),
                  "neighbouring_settings": all(v > 0 for v in r["selection_sharpe_by_params"].values()),
                  "no_lookahead": not bad}
        out[n] = {"final_params": r["final_params"], "holdout": hs, "holdout_regime": lab.by_regime(h["daily"], btc),
                  "holdout_round_trips": len(h["trips"]), "holdout_exchange_stops": h["stops"],
                  "holdout_pilot": {"losing_streak_stops": len(streaks),
                                    "days_below_daily_stop": int((h["daily"] < -TERMS.daily_stop).sum())},
                  "holdout_limits_needed": lab.limits_needed(h["entries"]),
                  "double_fees_walk_forward": ts, "lookahead_violations": bad, "checks": checks,
                  "passes": all(checks.values()), "score": r["score"]["total"]}
        print(n, r["final_params"], f"holdout ret {hs['total_return']:+.1%} sharpe {hs['sharpe']:+.2f} dd "
              f"{hs['max_drawdown']:.1%} trips {len(h['trips'])} | x2 ret {ts['total_return']:+.1%} sharpe "
              f"{ts['sharpe']:+.2f} | {checks}")
    order = sorted(out, key=lambda n: (-int(out[n]["passes"]), -out[n]["score"],
                                       -screened["ideas"][n]["walk_forward"]["sharpe"]))
    (HERE / "results.json").write_text(json.dumps({
        "date": RUN_DATE, "terms": {"capital": TERMS.capital, "slots": SLOTS, "stop_pct": TERMS.stop_pct,
                                    "pause_hours": TERMS.pause_hours}, "universe": json.loads((HERE / "universe.json").read_text()),  # noqa: E501
        "buy_and_hold": {"walk_forward_btc": wf_hold, "walk_forward_list": screened["buy_and_hold_list"],
                         "holdout_btc": hold["btc"], "holdout_list": hold["list"]},
        "screen": screened["ideas"], "final_check": out, "ranking": order,
        "champion": order[0] if order and out[order[0]]["passes"] else None}, indent=1, default=str) + "\n")
    print("holdout BTC hold", {k: round(v, 3) for k, v in hold["btc"].items() if isinstance(v, float)})
    print("holdout list hold", {k: round(v, 3) for k, v in hold["list"].items() if isinstance(v, float)})
    print("ranking", order)


if __name__ == "__main__":
    step = sys.argv[1]
    if step == "universe":
        for date, lst in zip(json.loads((HERE / "universe.json").read_text())["dates"] if False else
                             [pd.Timestamp(t0, unit="ms").strftime("%Y-%m-%d") for t0, _ in PERIODS], universe()):
            print(date, lst)
    elif step == "screen":
        screen()
    elif step == "final":
        final()
