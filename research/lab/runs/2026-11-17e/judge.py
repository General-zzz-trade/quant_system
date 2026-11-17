"""The 2026-11-17e cross-sectional long/short study (plan.md, registered before any result): the 10 most liquid crypto
USDT perpetuals of each period (BTC and ETH included when they qualify), long 3 and short 3 each day; momentum,
carry, low volatility, reversal and a trend/range regime switch; 1500 USDT notional (owner 2026-11-17), positions up
to 2x a slot on isolated margin; development periods only (the holdout period is never replayed).

    python research/lab/runs/2026-11-17e/judge.py screen    # selection, walk-forward -> screen.json
    python research/lab/runs/2026-11-17e/judge.py line      # benchmarks, deflated Sharpe, double costs -> line.json
    python research/lab/runs/2026-11-17e/judge.py report    # regimes, 150 USDT feasibility -> report.json

Every replay is registered in research/lab/trials.jsonl and cached under /root/lab_cache/2026-11-17e.
"""
from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
from lab import um_judge as judge  # noqa: E402
from lab import um_lab as lab  # noqa: E402
from lab import um_portfolio as portfolio  # noqa: E402
from lab import xs_study as xs  # noqa: E402
from lab.metrics import deflated_sharpe_probability_k  # noqa: E402

SNAP = ROOT / "data_files" / "binance_usdt_perps_2020_2026-11-12"
START, END = lab.ms("2020-01-01"), lab.ms("2026-11-13")
FOLDS = (*((lab.ms(f"{y}-05-13"), lab.ms(f"{y + 1}-05-13")) for y in range(2021, 2025)),
         (lab.ms("2025-05-13"), lab.ms("2025-11-16")))
PERIODS = judge.periods(START, FOLDS, END)
CACHE = Path("/root/lab_cache/2026-11-17e")
NOTE = "2026-11-17e cross-sectional long/short on the 2020+ snapshot, 1500 USDT notional (runs/2026-11-17e/plan.md)"
TERMS = lab.LiveTerms(capital="1500", leverage="2")              # 8 % stop, 4-hour pause: the runtime's
SLOTS = 10
LISTED_DAYS = 365
TRADFI = json.loads((ROOT / "research/lab/runs/2026-11-17b/tradfi_contracts.json").read_text())
EXCLUDED = ("Binance TRADIFI_PERPETUAL contracts (exchangeInfo 2026-11-17 03:20 UTC, "
            "runs/2026-11-17b/tradfi_contracts.json)")
VARIANTS = (judge.StopVariant("0.08", 4, "pause"), judge.StopVariant("0.08", 4, "pause_then_reset"))
LINE_PROBABILITY, MIN_TRIPS, REGIME_THR = 0.5, 50, 0.3
SMALL = "150"                                                    # the owner's capital today: feasibility report only


def ideas() -> dict:
    found = {name: (lab.legacy(fn, cross=True, reads=(judge.BTC,)), grid) for name, (fn, grid) in xs.IDEAS.items()}
    found["B2_xs_reversal_ls"] = lab.lab_ideas()["B2_xs_reversal_ls"]
    return found


def lists() -> list[list[str]]:
    path = HERE / "universe.json"
    if path.is_file():
        return json.loads(path.read_text())["lists"]
    dates = [start for start, _ in PERIODS]
    found = judge.universe(SNAP, dates, capital=TERMS.capital, slots=SLOTS, listed_days=LISTED_DAYS,
                           data_start=START, exclude=tuple(TRADFI["symbols"]))
    record = judge.universe_record(found, dates, capital=TERMS.capital, slots=SLOTS, listed_days=LISTED_DAYS,
                                   excluded=EXCLUDED)
    path.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    return found


def make(terms: lab.LiveTerms = TERMS, cache: Path = CACHE, note: str = NOTE) -> judge.Judge:
    return judge.Judge(ideas=ideas(), lists=lists(), cache=cache, run_date="2026-11-17e", snapshot=SNAP,
                       periods=PERIODS, note=note, terms=terms, slots=SLOTS)


def commit() -> str:
    return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


def n_trials() -> int:
    """Distinct configurations (idea, parameters, stop variant, terms; costs once) replayed on the 2020+ snapshot."""
    seen = set()
    for line in (ROOT / "research/lab/trials.jsonl").read_text().splitlines():
        row = json.loads(line)
        if "2020+ snapshot" in str(row.get("note", "")) and int(row.get("cost_multiplier", 1)) == 1:
            seen.add((row["idea"], json.dumps(row["params"], sort_keys=True), str(row["stop"]), str(row.get("terms"))))
    return len(seen)


def _chosen(name: str, row: dict, cost: int = 1) -> list[tuple[int, judge.Config]]:
    return [(k, judge.Config(name, label["params"], judge.StopVariant(**label["stop"]), cost))
            for k, label in zip(range(1, len(PERIODS) - 1), row["chosen_per_fold"])]


def _walk(j: judge.Judge, picks: list[tuple[int, judge.Config]]) -> tuple[pd.Series, list]:
    j.run_all([(config, (k,)) for k, config in picks], processes=2)
    parts = [j.load(config)[k] for k, config in picks]
    return pd.concat([p["daily"] for p in parts]), [t for p in parts for t in p["trips"]]


def line(j: judge.Judge, screened: dict) -> dict:
    folds = list(j.folds)
    hold = {k: lab.buy_and_hold(judge.BTC, a, b, SNAP) for k, (a, b) in enumerate(j.periods[:-1])}
    vol = portfolio.vol_target_hold(SNAP, j.periods[:-1], capital=TERMS.capital)
    bars = {"btc_hold": lab.summary(judge.joined({k: {"daily": hold[k]} for k in folds}, folds)),
            "btc_vol_target": lab.summary(judge.joined({k: {"daily": vol[k]} for k in folds}, folds))}
    sharpes = [v for row in screened["ideas"].values() for v in row["selection_sharpe"].values()]
    std, trials = float(np.std(sharpes, ddof=1)), n_trials()
    out = {}
    for name, row in screened["ideas"].items():
        daily, trips = _walk(j, _chosen(name, row))
        doubled, _ = _walk(j, _chosen(name, row, 2))
        s, d = lab.summary(daily), lab.summary(doubled)
        p = deflated_sharpe_probability_k(daily, std, trials)
        checks = {"return_above_0": s["total_return"] > 0, "beats_btc_hold": s["sharpe"] > bars["btc_hold"]["sharpe"],
                  "beats_btc_vol_target": s["sharpe"] > bars["btc_vol_target"]["sharpe"],
                  "deflated_at_least_0.5": p >= LINE_PROBABILITY, "round_trips_at_least_50": len(trips) >= MIN_TRIPS,
                  "double_costs_return_above_0": d["total_return"] > 0}
        out[name] = {"walk_forward": s, "double_costs": d, "round_trips": len(trips), "deflated_probability": p,
                     "checks": checks, "survives": all(checks.values())}
    return {"benchmarks": bars, "n_trials": trials, "selection_sharpe_std": std, "ideas": out}


def regime_split(daily: pd.Series) -> dict:
    """Compounded return and Sharpe on trend days (BTC's 30-day efficiency ratio, known at the day before's close,
    at least REGIME_THR) and on range days."""
    close = lab.frame(judge.BTC, SNAP)["close"]
    er = xs.efficiency(close.groupby(close.index // lab.DAY).last(), xs.REGIME_DAYS).shift(1).reindex(daily.index)
    out = {}
    for label, mask in (("trend", er >= REGIME_THR), ("range", er < REGIME_THR)):
        part = daily[mask]
        out[label] = {"days": int(len(part)), "return": float((1 + part).prod() - 1),
                      "sharpe": float(part.mean() / part.std() * np.sqrt(365)) if part.std() > 0 else 0.0}
    return out


def report(j: judge.Judge, screened: dict, verdict: dict) -> dict:
    regimes = {}
    for name, row in screened["ideas"].items():
        regimes[name] = regime_split(_walk(j, _chosen(name, row))[0])
    best = max(verdict["ideas"], key=lambda n: verdict["ideas"][n]["walk_forward"]["sharpe"])
    small = make(replace(TERMS, capital=SMALL), CACHE / "capital150",
                 NOTE + "; 150 USDT feasibility report (plan.md)")
    picks = _chosen(best, screened["ideas"][best])
    daily, trips = _walk(small, picks)
    entries = {c: sum(len(jj.load(config)[k]["entries"]) for k, config in picks) for c, jj in (("1500", j),
                                                                                              (SMALL, small))}
    return {"regimes": regimes, "regime_threshold": REGIME_THR,
            "capital_150": {"idea": best, "walk_forward": lab.summary(daily), "round_trips": len(trips),
                            "entry_orders": entries}}


def main(step: str) -> None:
    j = make()
    if step == "screen":
        screened = j.screen(variants=VARIANTS, processes=2)
        (HERE / "screen.json").write_text(json.dumps({"code_commit": commit(), **screened}, indent=1, default=str))
    elif step == "line":
        screened = json.loads((HERE / "screen.json").read_text())
        (HERE / "line.json").write_text(json.dumps({"code_commit": commit(), **line(j, screened)}, indent=1,
                                                   default=str))
    elif step == "report":
        screened = json.loads((HERE / "screen.json").read_text())
        verdict = json.loads((HERE / "line.json").read_text())
        (HERE / "report.json").write_text(json.dumps({"code_commit": commit(), **report(j, screened, verdict)},
                                                     indent=1, default=str))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "")
