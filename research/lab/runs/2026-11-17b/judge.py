"""The 2026-11-17b lab run (plan.md, registered before any result): the new symbol lists, the 6 stop variants the
runtime can run, the higher elimination line and the fold-by-fold portfolio.

    python research/lab/runs/2026-11-17b/judge.py screen      # selection and walk-forward -> screen.json, bench.json
    python research/lab/runs/2026-11-17b/judge.py line        # the elimination line and the portfolio -> line.json
    python research/lab/runs/2026-11-17b/judge.py final       # survivors only: the holdout -> results.json

Every replay is registered in research/lab/trials.jsonl and cached under /root/lab_cache/2026-11-17b.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
from lab import um_judge as judge  # noqa: E402
from lab import um_lab as lab  # noqa: E402
from lab import um_portfolio as portfolio  # noqa: E402
from lab.causality import violations  # noqa: E402
from lab.metrics import deflated_sharpe_probability_k  # noqa: E402
from trading.domain.strategy.base import Window  # noqa: E402

SNAP = ROOT / "data_files" / "binance_usdt_perps_2020_2026-11-12"
START, END = lab.ms("2020-01-01"), lab.ms("2026-11-13")
FOLDS = tuple((lab.ms(f"{y}-05-13"), lab.ms(f"{y + 1}-05-13")) for y in range(2021, 2026))
PERIODS = judge.periods(START, FOLDS, END)
CACHE = Path("/root/lab_cache/2026-11-17b")
NOTE = "2026-11-17b run on the 2020+ snapshot (runs/2026-11-17b/plan.md)"
VARIANTS = tuple(judge.StopVariant(stop, 4, after) for stop in ("0.08", "0.12", "0.15")
                 for after in ("pause", "pause_then_reset"))
LISTED_DAYS = 365
TRADFI = json.loads((HERE / "tradfi_contracts.json").read_text())
EXCLUDED = "Binance TRADIFI_PERPETUAL contracts (exchangeInfo 2026-11-17 03:20 UTC, tradfi_contracts.json)"
LINE_PROBABILITY = 0.5
H = lab.H


def lists() -> list[list[str]]:
    path = HERE / "universe.json"
    if path.is_file():
        return json.loads(path.read_text())["lists"]
    dates = [start for start, _ in PERIODS]
    found = judge.universe(SNAP, dates, capital=lab.LIVE.capital, slots=10, listed_days=LISTED_DAYS,
                           data_start=START, exclude=tuple(TRADFI["symbols"]))
    record = judge.universe_record(found, dates, capital=lab.LIVE.capital, slots=10, listed_days=LISTED_DAYS,
                                   excluded=EXCLUDED)
    path.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    return found


def causal(name: str, params: dict) -> list[str]:
    """As runs/2026-11-17/judge.py: lab/causality.py on real data through the strategy interface."""
    strategy = lab.lab_ideas()[name][0](params)
    first = list(lists()[3][:5])
    symbols = first + [s for s in strategy.references() if s not in first]
    lo, hi = lab.ms("2023-01-01"), lab.ms("2025-01-01")
    inputs = {}
    for s in symbols:
        w = lab.snapshot_window(s, SNAP)
        hourly = w.hourly[(w.hourly["open_time"] >= lo) & (w.hourly["open_time"] < hi)].reset_index(drop=True)
        inputs[s] = {"hourly": hourly, "funding": w.funding[(w.funding.index >= lo) & (w.funding.index < hi)]}

    def run(inp: dict) -> dict:
        return strategy.history({s: Window(v["hourly"].reset_index(drop=True), v["funding"]) for s, v in inp.items()})
    try:
        return violations(run, inputs, [lab.ms("2024-03-05") + 7 * H, lab.ms("2024-09-20") + 23 * H])
    except Exception as exc:
        return [f"{type(exc).__name__}: {exc}"[:200]]


def make() -> judge.Judge:
    return judge.Judge(ideas=lab.lab_ideas(), lists=lists(), cache=CACHE, run_date="2026-11-17b", snapshot=SNAP,
                       periods=PERIODS, note=NOTE, lookahead=causal)


def commit() -> str:
    return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


def runnable(j: judge.Judge, name: str) -> bool:
    reads = set(j.ideas[name][0](j.ideas[name][1][0]).references())
    return all(reads <= set(symbols) for symbols in j.lists)


def n_trials() -> int:
    """Distinct configurations (idea, parameters, stop variant; costs once) replayed on the 2020+ snapshot."""
    seen = set()
    for line in (ROOT / "research/lab/trials.jsonl").read_text().splitlines():
        row = json.loads(line)
        if "2020+ snapshot" in str(row.get("note", "")) and int(row.get("cost_multiplier", 1)) == 1:
            seen.add((row["idea"], json.dumps(row["params"], sort_keys=True), str(row["stop"])))
    return len(seen)


def benchmarks(j: judge.Judge) -> tuple[dict, dict]:
    hold = {k: lab.buy_and_hold(judge.BTC, a, b, SNAP) for k, (a, b) in enumerate(j.periods[:-1])}   # no holdout
    vol = portfolio.vol_target_hold(SNAP, j.periods[:-1], capital=lab.LIVE.capital)
    return hold, vol


def line(j: judge.Judge, screened: dict) -> dict:
    hold, vol = benchmarks(j)
    folds = list(j.folds)
    bars = {"btc_hold": lab.summary(judge.joined({k: {"daily": hold[k]} for k in folds}, folds)),
            "btc_vol_target": lab.summary(judge.joined({k: {"daily": vol[k]} for k in folds}, folds))}
    sharpes = [v for row in screened["ideas"].values() for v in row["selection_sharpe"].values()]
    std, trials = float(np.std(sharpes, ddof=1)), n_trials()
    ideas = {}
    for name in screened["ideas"]:
        runs = [(config, j.load(config)) for config in j.candidates(name, VARIANTS)]
        wf = judge.walk_forward(runs, folds)
        s = lab.summary(wf["daily"])
        p = deflated_sharpe_probability_k(wf["daily"], std, trials)
        checks = {"return_above_0": s["total_return"] > 0, "beats_btc_hold": s["sharpe"] > bars["btc_hold"]["sharpe"],
                  "beats_btc_vol_target": s["sharpe"] > bars["btc_vol_target"]["sharpe"],
                  "deflated_at_least_0.5": p >= LINE_PROBABILITY}
        ideas[name] = {"walk_forward": s, "deflated_probability": p, "checks": checks, "survives": all(checks.values())}
    runs = {name: [(c, j.load(c)) for c in j.candidates(name, VARIANTS)] for name in screened["ideas"]}
    pf = portfolio.walk_forward_portfolio(runs, j.periods, folds, [hold, vol], eligible=lambda n: runnable(j, n))
    ps = lab.summary(pf["daily"])
    pp = deflated_sharpe_probability_k(pf["daily"], std, trials + 1)
    trips = 0
    for i, fold in enumerate(folds):
        for name in pf["members"][i]:
            label = pf["chosen"][name][i]
            config = judge.Config(name, label["params"], judge.StopVariant(**label["stop"]))
            trips += len(j.load(config)[fold]["trips"])
    pchecks = {"return_above_0": ps["total_return"] > 0, "beats_btc_hold": ps["sharpe"] > bars["btc_hold"]["sharpe"],
               "beats_btc_vol_target": ps["sharpe"] > bars["btc_vol_target"]["sharpe"],
               "deflated_at_least_0.5": pp >= LINE_PROBABILITY, "at_least_50_trades": trips >= 50}
    return {"benchmarks": bars, "n_trials": trials, "selection_sharpe_std": std, "ideas": ideas,
            "portfolio": {"members": pf["members"], "walk_forward": ps, "regime": lab.by_regime(pf["daily"],
                          lab.frame(judge.BTC, SNAP)), "round_trips": trips, "deflated_probability": pp,
                          "checks": pchecks, "survives": all(pchecks.values())}}


def main(step: str) -> None:
    j = make()
    if step == "screen":
        screened = j.screen(variants=VARIANTS, processes=2)
        (HERE / "screen.json").write_text(json.dumps({"code_commit": commit(), **screened}, indent=1, default=str))
    elif step == "line":
        screened = json.loads((HERE / "screen.json").read_text())
        (HERE / "line.json").write_text(json.dumps({"code_commit": commit(), **line(j, screened)}, indent=1,
                                                   default=str))
    elif step == "final":
        screened = json.loads((HERE / "screen.json").read_text())
        verdict = json.loads((HERE / "line.json").read_text())
        for name, row in screened["ideas"].items():          # the higher line of plan.md replaces the screen's own
            row["survives"] = verdict["ideas"][name]["survives"]
        checked = j.final(screened, processes=2)
        (HERE / "results.json").write_text(json.dumps({"code_commit": commit(), "screen": screened, "final": checked},
                                                      indent=1, default=str))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "")
