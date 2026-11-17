"""The 2026-11-17c lab run (plan.md, registered before any result): BTCUSDT and ETHUSDT only at lab.LIVE_BTC_ETH (150
USDT, half each, positions up to 2x on isolated margin); the 29 registered ideas and the frozen overlay x the 6 stop
variants the runtime can run, on the 2020+ snapshot; the higher elimination line of 2026-11-17b; the final check of
the survivors.

    python research/lab/runs/2026-11-17c/judge.py screen   # selection and walk-forward -> screen.json
    python research/lab/runs/2026-11-17c/judge.py line     # the elimination line -> line.json
    python research/lab/runs/2026-11-17c/judge.py final    # survivors only: the holdout -> results.json

Every replay is registered in research/lab/trials.jsonl and cached under /root/lab_cache/2026-11-17c.
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
from trading.domain.strategy.base import OverlayStrategy, Window  # noqa: E402
from trading.domain.strategy.spec import PARAMS  # noqa: E402

SNAP = ROOT / "data_files" / "binance_usdt_perps_2020_2026-11-12"
START, END = lab.ms("2020-01-01"), lab.ms("2026-11-13")
FOLDS = tuple((lab.ms(f"{y}-05-13"), lab.ms(f"{y + 1}-05-13")) for y in range(2021, 2026))
PERIODS = judge.periods(START, FOLDS, END)
CACHE = Path("/root/lab_cache/2026-11-17c")
NOTE = "2026-11-17c run on the 2020+ snapshot, BTCUSDT and ETHUSDT at LIVE_BTC_ETH (runs/2026-11-17c/plan.md)"
TERMS = lab.LIVE_BTC_ETH
VARIANTS = tuple(judge.StopVariant(stop, 4, after) for stop in ("0.08", "0.12", "0.15")
                 for after in ("pause", "pause_then_reset"))
LINE_PROBABILITY = 0.5
OVERLAY = json.loads((ROOT / "research/results/overlay_2026-11-13/final_spec.json").read_text())["spec"]
H = lab.H


def ideas() -> dict:
    found = lab.lab_ideas()
    found["overlay"] = (lambda params: OverlayStrategy(PARAMS["overlay"](params)), [OVERLAY])
    return found


def lists() -> list[list[str]]:
    path = HERE / "universe.json"
    if path.is_file():
        return json.loads(path.read_text())["lists"]
    found = judge.fixed_lists(TERMS.symbols, PERIODS)
    record = {"lists": found, "capital": TERMS.capital, "slots": len(TERMS.symbols), "leverage": TERMS.leverage,
              "rule": "BTCUSDT and ETHUSDT only, the same list in every period (owner 2026-11-17)"}
    path.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    return found


def causal(name: str, params: dict) -> list[str]:
    """lab/causality.py on real data through the strategy interface: BTCUSDT and ETHUSDT, 2023-01..2025-01."""
    strategy = ideas()[name][0](params)
    symbols = list(TERMS.symbols)
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
    return judge.Judge(ideas=ideas(), lists=lists(), cache=CACHE, run_date="2026-11-17c", snapshot=SNAP,
                       periods=PERIODS, note=NOTE, lookahead=causal, terms=TERMS, slots=len(TERMS.symbols))


def commit() -> str:
    return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


def n_trials() -> int:
    """Distinct configurations (idea, parameters, stop variant; costs once) replayed on the 2020+ snapshot."""
    seen = set()
    for line in (ROOT / "research/lab/trials.jsonl").read_text().splitlines():
        row = json.loads(line)
        if "2020+ snapshot" in str(row.get("note", "")) and int(row.get("cost_multiplier", 1)) == 1:
            seen.add((row["idea"], json.dumps(row["params"], sort_keys=True), str(row["stop"]), str(row.get("terms"))))
    return len(seen)


def line(j: judge.Judge, screened: dict) -> dict:
    folds = list(j.folds)
    hold = {k: lab.buy_and_hold(judge.BTC, a, b, SNAP) for k, (a, b) in enumerate(j.periods[:-1])}
    vol = portfolio.vol_target_hold(SNAP, j.periods[:-1], capital=TERMS.capital)
    bars = {"btc_hold": lab.summary(judge.joined({k: {"daily": hold[k]} for k in folds}, folds)),
            "btc_vol_target": lab.summary(judge.joined({k: {"daily": vol[k]} for k in folds}, folds))}
    sharpes = [v for row in screened["ideas"].values() for v in row["selection_sharpe"].values()]
    std, trials = float(np.std(sharpes, ddof=1)), n_trials()
    out = {}
    for name in screened["ideas"]:
        runs = [(config, j.load(config)) for config in j.candidates(name, VARIANTS)]
        wf = judge.walk_forward(runs, folds)
        s = lab.summary(wf["daily"])
        p = deflated_sharpe_probability_k(wf["daily"], std, trials)
        checks = {"return_above_0": s["total_return"] > 0, "beats_btc_hold": s["sharpe"] > bars["btc_hold"]["sharpe"],
                  "beats_btc_vol_target": s["sharpe"] > bars["btc_vol_target"]["sharpe"],
                  "deflated_at_least_0.5": p >= LINE_PROBABILITY}
        out[name] = {"walk_forward": s, "deflated_probability": p, "checks": checks, "survives": all(checks.values())}
    return {"benchmarks": bars, "n_trials": trials, "selection_sharpe_std": std, "ideas": out}


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
