"""How stable the stop choice is (asked by the tech lead 2026-11-17: does the best stop change from year to year?).
Reads this run's screen.json and cache only; the holdout is not read.

    python research/lab/runs/2026-11-17/stop_stability.py      # -> stop_stability.json

Per idea: the stop variant each fold chose (with the parameters, from data before the fold), how many distinct ones,
and the walk-forward with each of the 12 stop variants held fixed (parameters still chosen per fold): its Sharpe,
return and per-fold Sharpe, so the spread across stops is visible.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import judge as run  # noqa: E402  (this run's judge.py)
from lab import um_judge as judge  # noqa: E402
from lab import um_lab as lab  # noqa: E402


def _fold_sharpes(daily: pd.Series, j: judge.Judge) -> list[float]:
    out = []
    for k in j.folds:
        a, b = j.periods[k]
        out.append(lab.summary(daily[(daily.index >= a // lab.DAY) & (daily.index < b // lab.DAY)])["sharpe"])
    return out


def _walk(j: judge.Judge, name: str, variants: list[judge.StopVariant]) -> dict:
    runs = [(config, j.load(config)) for config in j.candidates(name, variants)]
    wf = judge.walk_forward(runs, list(j.folds))
    return {"sharpe": lab.summary(wf["daily"])["sharpe"], "total_return": lab.summary(wf["daily"])["total_return"],
            "folds": _fold_sharpes(wf["daily"], j), "chosen": [c.label for c in wf["chosen"]]}


def main() -> None:
    j = run.make()
    screen = json.loads((HERE / "screen.json").read_text())["ideas"]
    out = {}
    for name, row in screen.items():
        chosen = [judge.StopVariant(**label["stop"]).name for label in row["chosen_per_fold"]]
        per_stop = {v.name: _walk(j, name, [v]) for v in judge.STOP_VARIANTS}
        out[name] = {"walk_forward_sharpe": row["walk_forward"]["sharpe"], "chosen_stop_per_fold": chosen,
                     "distinct_stops": len(set(chosen)), "changes": sum(a != b for a, b in zip(chosen, chosen[1:])),
                     "fixed_stop": {k: {"sharpe": v["sharpe"], "total_return": v["total_return"], "folds": v["folds"]}
                                    for k, v in per_stop.items()}}
    (HERE / "stop_stability.json").write_text(json.dumps({"code_commit": run.commit(), "ideas": out}, indent=1))


if __name__ == "__main__":
    main()
