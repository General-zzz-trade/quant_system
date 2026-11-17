"""Register the configurations of the two 2026-11-17 pilot studies in research/trials.jsonl (family
overlay_pilot_leverage, as the 2026-11-16 leverage study): the 2-10x leverage study (75 runs; not registered when it
ran, an omission found on 2026-11-17) and the settings study (60 settings, and the reruns at double costs)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from lab import trials  # noqa: E402
from lab.study import DEV_END, EVAL_START  # noqa: E402

WINDOW = [EVAL_START, DEV_END]
FAMILY = "overlay_pilot_leverage"
NOTE = "development window only; the frozen overlay at 150 USDT; the owner decides the settings"


def main():
    leverage = json.loads((ROOT / "research/results/overlay_pilot_leverage_2026-11-17/results.json").read_text())
    rows = [{"family": FAMILY, "study": "overlay_pilot_leverage_2026-11-17", "trial": key,
             "config": {"run": key, "spec": "lab.overlay_study.SPEC", "capital_usdt": 150}, "window": WINDOW,
             "sharpe": run["stats"]["sharpe"], "count": 1, "note": NOTE} for key, run in leverage["runs"].items()]
    params = json.loads((ROOT / "research/results/pilot_params_2026-11-17/results.json").read_text())
    rows += [{"family": FAMILY, "study": "pilot_params_2026-11-17", "trial": key, "config": row["setting"],
              "window": WINDOW, "sharpe": round(row["whole"]["sharpe"], 4), "count": 1, "note": NOTE + "; cost x1"}
             for key, row in params["settings"].items()]
    rows.append({"family": FAMILY, "study": "pilot_params_2026-11-17", "trial": f"{params['final']['choice']} cost x2",
                 "config": {"setting": params["final"]["choice"], "cost": 2}, "window": WINDOW,
                 "sharpe": round(params["final"]["double_costs"]["sharpe"], 4), "count": 1, "note": NOTE})
    trials.record(rows, path=ROOT / "research/trials.jsonl")
    print(len(rows), "rows registered;", trials.family_stats(FAMILY, path=ROOT / "research/trials.jsonl"))


if __name__ == "__main__":
    main()
