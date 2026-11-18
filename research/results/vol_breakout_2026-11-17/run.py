"""The volatility-expansion breakout study (README.md, registered before any code; owner 2026-11-17 "跑回测验证").

Development data only (to lab.study.DEV_END). Builds the panel exactly as research/results/panel_2026-11-17/run.py,
takes the yearly walk-forward predictions (trend_14 logistic: the filter; fwd_vol_7 ridge: the sizes), finds the
breakout trades on the universe's days from 2022-01-01, hands out 10 slots, replays every contract through
lab.um_lab.replay at 1x and 2x costs, registers every replay in research/lab/trials.jsonl, then writes results.json.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from lab import feature_pool as fp  # noqa: E402
from lab import um_lab as lab  # noqa: E402
from lab import um_portfolio as portfolio  # noqa: E402
from lab import vol_breakout as vb  # noqa: E402
from lab.metrics import deflated_sharpe_probability_k  # noqa: E402
from lab.study import DEV_END, EVAL_START  # noqa: E402
from trading.domain.strategy.ideas import ON  # noqa: E402

_spec = importlib.util.spec_from_file_location("panel_run", ROOT / "research/results/panel_2026-11-17/run.py")
P = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P)

SNAP, DAY, H = P.SNAP, vb.DAY, vb.H
TEST_START = lab.ms("2022-01-01")
CAPITAL, SLOT, LEVERAGE, STOP = 1500.0, 150.0, "2", "0.08"
PAUSE_MS = lab.STOP_PAUSE_SECONDS * 1000
BTC_ETH = ("BTCUSDT", "ETHUSDT")
CONFIGS = {"F0S0": ("none", "equal"), "F1S0": ("trend_top20", "equal"), "F0S1": ("none", "pred_vol"),
           "F1S1": ("trend_top20", "pred_vol")}
RUN_DATE, IDEA = "2026-11-17f", "V1_vol_breakout"
STOP_NAME = "stop0.08_pause4h_then_reset"
NOTE = "2026-11-17f volatility-expansion breakout on the 2020+ snapshot (results/vol_breakout_2026-11-17/README.md)"
OUT = HERE / "results.json"


def replay_job(job: tuple) -> dict:
    """One contract's trades of one configuration through the shared broker."""
    symbol, trades, sizes, cost, balance = job
    f = lab.frame(symbol, SNAP)
    w = vb.weights(trades, dict(zip(trades, sizes)), f.index)
    start = max(TEST_START, min(t.entry for t in trades) - DAY)
    end = min(max(t.exit for t in trades) + 2 * DAY, DEV_END)
    r = lab.replay(symbol, w, start=start, end=end, snapshot=SNAP, starting_balance=str(balance), stop_pct=STOP,
                   pause_ms=PAUSE_MS, reenter_after_stop="pause_then_reset", leverage=LEVERAGE, cost_multiplier=cost)
    curve = r["curve"]
    daily = curve.groupby(curve.index.to_numpy(dtype=np.int64) // DAY * DAY).last()
    return {"symbol": symbol, "daily": daily, "returns": vb.trade_returns(trades, curve, slot=balance),
            "trips": len(lab.round_trips(r["fills"])), "fills": len(r["fills"])}


def run_config(trades: list, sizes: list, cost: int, balance: float, capital: float, pool) -> dict:
    by_symbol: dict[str, list] = {}
    for t, s in zip(trades, sizes):
        by_symbol.setdefault(t.symbol, []).append((t, s))
    jobs = [(sym, [t for t, _ in v], [s for _, s in v], cost, balance) for sym, v in sorted(by_symbol.items())]
    results = list(pool.map(replay_job, jobs, chunksize=4))
    days = np.arange(TEST_START, DEV_END, DAY, dtype=np.int64)
    daily = vb.account_daily({r["symbol"]: r["daily"] for r in results}, capital=capital, slot=balance, days=days)
    daily.index = daily.index // DAY
    returns = {}
    for (sym, ts, _, _, _), r in zip(jobs, results):
        for t, x in zip(ts, r["returns"]):
            returns[t] = x
    return {"daily": daily, "summary": lab.summary(daily), "trips": sum(r["trips"] for r in results),
            "fills": sum(r["fills"] for r in results), "trade_returns": returns}


def registry_rows(name: str, params: dict, cost: int, res: dict, terms: str, extra: str = "") -> dict:
    s = res["summary"]
    return {"date": RUN_DATE, "idea": IDEA, "params": params, "stop": STOP_NAME, "cost_multiplier": cost,
            "holdout": False, "terms": terms, "window": ["2022-01-01", "2025-11-16"],
            "total_return": s.get("total_return"), "sharpe": s.get("sharpe"), "max_drawdown": s.get("max_drawdown"),
            "round_trips": res["trips"], "fills": res["fills"], "note": NOTE + extra}


def n_trials_and_std() -> tuple[int, float]:
    seen, sharpes = set(), []
    for line in lab.TRIALS.read_text().splitlines():
        row = json.loads(line)
        if "2020+ snapshot" not in str(row.get("note", "")) or int(row.get("cost_multiplier", 1)) != 1:
            continue
        seen.add((row["idea"], json.dumps(row["params"], sort_keys=True), str(row["stop"]), str(row.get("terms"))))
        if not row.get("holdout") and (row.get("round_trips") or 0) > 0 and row.get("sharpe") is not None \
                and np.isfinite(row["sharpe"]):
            sharpes.append(float(row["sharpe"]))
    return len(seen), float(np.std(sharpes, ddof=1))


def predictions(panel: pd.DataFrame, y: pd.DataFrame, x: pd.DataFrame) -> dict[str, pd.DataFrame]:
    t = panel.index.to_numpy(dtype=np.int64)
    out = {}
    for name, kind, model in (("trend_14", "binary", "logistic"), ("fwd_vol_7", "regression", "ridge")):
        _, end, _ = P.TARGETS[name]
        target = y[name].to_numpy(float)
        label_end = y[end].fillna(pd.Series(t + 14 * DAY, index=y.index)).to_numpy(dtype=np.int64)
        pred = vb.walk_forward_predictions(x, target, t, label_end, P.YEARS, kind=kind, model=model)
        out[name] = pd.DataFrame({"time": t, "symbol": panel["symbol"].to_numpy(), "pred": pred})
        print(name, model, "test rows with a prediction", int(np.isfinite(pred).sum()), flush=True)
    return out


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"{OUT} exists: the study runs once")
    times = np.arange(EVAL_START + DAY, DEV_END + 1, DAY, dtype=np.int64)
    panel, y, info = P.build_panel(times)
    panel["log_rv_30"] = np.log(panel["rv_30"])
    x = panel[list(P.ps.PANEL_FEATURES) + ["log_rv_30"]].astype(np.float32)
    preds = predictions(panel, y, x)
    passed = vb.top_share(preds["trend_14"])
    sizes_s1 = vb.vol_sizes(preds["fwd_vol_7"])
    members = panel[["symbol"]].assign(time=panel.index.to_numpy(dtype=np.int64))
    members = members[members["time"] >= TEST_START]
    trades, volumes, first = [], {}, {}
    for symbol, rows in members.groupby("symbol"):
        h, _ = P.ps.load_symbol(SNAP, symbol, DEV_END)
        allowed = pd.Series(True, index=pd.Index(rows["time"].to_numpy(dtype=np.int64)))
        trades += vb.breakout_trades(symbol, fp.daily(h), allowed=allowed)
        volumes[symbol], first[symbol] = P.ps.daily_volume(h), int(h["open_time"].iloc[0])
    medians = vb.median_volumes(volumes, first, times)
    priority = {(t.symbol, t.entry): float(medians.at[t.entry, t.symbol]) for t in trades}
    accepted = {t: t.symbol in passed.get(t.entry, set()) for t in trades}
    print("signal trades", len(trades), "accepted by the filter", sum(accepted.values()), flush=True)
    chosen = {"none": vb.allocate(trades, lambda t: priority[(t.symbol, t.entry)]),
              "trend_top20": vb.allocate([t for t in trades if accepted[t]], lambda t: priority[(t.symbol, t.entry)])}
    missing_size = sum((t.symbol, t.entry) not in sizes_s1 for t in trades)

    def size(t, how):
        return ON if how == "equal" else sizes_s1.get((t.symbol, t.entry), ON)

    terms = f"1500 USDT, 10 slots, daily top-100 panel universe, {STOP_NAME}, 2x"
    out: dict = {"info": info, "signal_trades": len(trades), "accepted": sum(accepted.values()),
                 "missing_vol_prediction": missing_size, "configs": {}, "btc_eth": {}}
    rows, results = [], {}
    with ProcessPoolExecutor(2) as pool:
        for name, (filt, how) in CONFIGS.items():
            ts = chosen[filt]
            for cost in (1, 2):
                res = run_config(ts, [size(t, how) for t in ts], cost, SLOT, CAPITAL, pool)
                results[(name, cost)] = res
                rows.append(registry_rows(name, {"filter": filt, "size": how}, cost, res, terms))
                print(name, "cost", cost, "round trips", res["trips"], "fills", res["fills"], flush=True)   # no result
        for name, (filt, how) in CONFIGS.items():            # report only: BTC and ETH, no slot limit, 150 USDT
            ts = [t for t in trades if t.symbol in BTC_ETH and (filt == "none" or accepted[t])]
            res = run_config(ts, [size(t, how) for t in ts], 1, 75.0, 150.0, pool)
            results[("btc_eth " + name, 1)] = res
            rows.append(registry_rows(name, {"filter": filt, "size": how, "symbols": "BTCUSDT,ETHUSDT"}, 1, res,
                                      f"150 USDT, BTCUSDT and ETHUSDT halves, {STOP_NAME}, 2x", "; report only"))
    n_trials, std = n_trials_and_std()
    n_trials += len({json.dumps(r["params"], sort_keys=True) for r in rows if r["cost_multiplier"] == 1})
    period = (TEST_START, DEV_END)
    hold = lab.summary(lab.buy_and_hold("BTCUSDT", *period, SNAP))
    vol = lab.summary(portfolio.vol_target_hold(SNAP, [period], capital="1500")[0])
    out["benchmarks"], out["n_trials"], out["sharpe_std"] = {"btc_hold": hold, "btc_vol_target": vol}, n_trials, std
    for name, (filt, how) in CONFIGS.items():
        res, doubled = results[(name, 1)], results[(name, 2)]
        s, d = res["summary"], doubled["summary"]
        p = deflated_sharpe_probability_k(res["daily"], std, n_trials)
        checks = {"return_above_0": s["total_return"] > 0, "beats_btc_hold": s["sharpe"] > hold["sharpe"],
                  "beats_btc_vol_target": s["sharpe"] > vol["sharpe"], "deflated_at_least_0.5": p >= 0.5,
                  "round_trips_at_least_50": res["trips"] >= 50, "double_costs_return_above_0": d["total_return"] > 0}
        years = pd.to_datetime(res["daily"].index * DAY, unit="ms").year
        ts = chosen[filt]
        rets = np.array([res["trade_returns"][t] for t in ts])
        sides = np.array([t.side for t in ts])
        notional = np.array([15.0 * 2 * size(t, how) for t in ts])
        minimum = np.array([float(lab.rules(t.symbol, SNAP).min_notional) for t in ts])
        out["configs"][name] = {
            "filter": filt, "size": how, "summary": s, "double_costs": d, "round_trips": res["trips"],
            "trades": len(ts), "deflated_probability": p, "checks": checks, "line": all(checks.values()),
            "by_year": {int(v): lab.summary(res["daily"][years == v]) for v in sorted(set(years))},
            "long": {"trades": int((sides == 1).sum()), "mean_return": float(rets[sides == 1].mean())},
            "short": {"trades": int((sides == -1).sum()), "mean_return": float(rets[sides == -1].mean())},
            "at_150_usdt_below_minimum": float((notional < minimum).mean())}
    f0 = chosen["none"]
    base = results[("F0S0", 1)]["trade_returns"]
    r = np.array([base[t] for t in f0])
    a = np.array([accepted[t] for t in f0])
    yrs = np.array([pd.Timestamp(t.entry, unit="ms").year for t in f0])
    out["filter_value"] = vb.filter_value(r, a, yrs)
    sd = np.array([t.side for t in f0])
    out["filter_value_by_side"] = {side: vb.filter_value(r[sd == k], a[sd == k], yrs[sd == k])
                                   for side, k in (("long", 1), ("short", -1))}
    c = out["configs"]
    pairs = (("F0S1", "F0S0"), ("F1S1", "F1S0"))
    out["sizing_value"] = {"pairs": {f"{s1} vs {s0}": {k: [c[s1]["summary"][k], c[s0]["summary"][k]]
                                                        for k in ("sharpe", "max_drawdown")} for s1, s0 in pairs}}
    out["sizing_value"]["value"] = all(
        c[s1]["summary"]["sharpe"] > c[s0]["summary"]["sharpe"]
        and c[s1]["summary"]["max_drawdown"] <= c[s0]["summary"]["max_drawdown"]
        for s1, s0 in pairs)
    for name in CONFIGS:
        res = results[("btc_eth " + name, 1)]
        out["btc_eth"][name] = {"summary": res["summary"], "round_trips": res["trips"],
                                "trades": len(res["trade_returns"])}
    with lab.TRIALS.open("a") as fh:                            # every replay registered before anything is shown
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    OUT.write_text(json.dumps(out, indent=1, default=lambda v: v.item() if hasattr(v, "item") else str(v)) + "\n")
    print(json.dumps({"filter_value": out["filter_value"]["value"], "sizing_value": out["sizing_value"]["value"],
                      "lines": {k: v["line"] for k, v in c.items()}}, indent=1))


if __name__ == "__main__":
    main()
