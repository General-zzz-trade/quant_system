"""The volatility-tightened stop (vol_stop.md, registered before any result): the 14 new stop variants on top of the
12 of this run, the comparison vol_stop.md defines, and the final check of whoever survives with all 26.

    python research/lab/runs/2026-11-17/vol_stop.py screen     # 26 variants (the 12 come from the cache)
    python research/lab/runs/2026-11-17/vol_stop.py compare    # vol_stop.md's comparison -> vol_compare.json
    python research/lab/runs/2026-11-17/vol_stop.py final      # survivors only: the holdout, doubled costs, look-ahead

Same data, periods, lists, costs and cache as judge.py (windows.md); every replay goes to research/lab/trials.jsonl.
"""
from __future__ import annotations

import json
import sys
from statistics import median
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import judge as run  # noqa: E402  (this run's judge.py: snapshot, periods, lists, cache)
from lab import um_judge as judge  # noqa: E402
from stop_stability import _walk  # noqa: E402

CAPS, KS, AFTER = ("0.045", "0.08"), ("1.5", "2", "3"), ((4, "pause"), (None, "signal_reset"))
NEW = tuple([judge.StopVariant("0.045", hours, reenter) for hours, reenter in AFTER]
            + [judge.StopVariant(cap, hours, reenter, vol_k=k) for cap in CAPS for k in KS for hours, reenter in AFTER])
VARIANTS = judge.STOP_VARIANTS + NEW            # the 12 first: ties go to the variant listed first
TOP = 10


def _fixed(cap: str, hours: int | None, reenter: str) -> judge.StopVariant:
    return judge.StopVariant(cap, hours, reenter)


def compare(j: judge.Judge) -> dict:
    base = json.loads((HERE / "screen.json").read_text())["ideas"]
    top = sorted(base, key=lambda n: -base[n]["walk_forward"]["sharpe"])[:TOP]
    groups = {}
    for cap in CAPS:
        for hours, reenter in AFTER:
            key = f"cap{cap}_{'pause4h' if hours else 'signal_reset'}"
            rows = {}
            for name in top:
                fixed = _walk(j, name, [_fixed(cap, hours, reenter)])
                vol = _walk(j, name, [judge.StopVariant(cap, hours, reenter, vol_k=k) for k in KS])
                rows[name] = {"fixed": fixed, "vol": vol, "gain": vol["sharpe"] - fixed["sharpe"],
                              "folds_better": sum(v > f for v, f in zip(vol["folds"], fixed["folds"]))}
            gains = [r["gain"] for r in rows.values()]
            checks = {"majority_better": sum(g > 0 for g in gains) >= 6,
                      "median_gain_at_least_0.1": median(gains) >= 0.1,
                      "median_idea_3_folds_better": median(r["folds_better"] for r in rows.values()) >= 3}
            groups[key] = {"ideas": rows, "checks": checks, "useful": all(checks.values())}
    return {"top": top, "groups": groups, "useful": any(g["useful"] for g in groups.values())}


def main(step: str) -> None:
    j = run.make()
    if step == "screen":
        screened = j.screen(variants=VARIANTS, processes=2)
        (HERE / "vol_screen.json").write_text(json.dumps({"code_commit": run.commit(), **screened}, indent=1,
                                                         default=str))
    elif step == "compare":
        (HERE / "vol_compare.json").write_text(json.dumps({"code_commit": run.commit(), **compare(j)}, indent=1,
                                                          default=str))
    elif step == "final":
        screened = json.loads((HERE / "vol_screen.json").read_text())
        checked = j.final(screened, processes=2)
        (HERE / "vol_results.json").write_text(json.dumps({"code_commit": run.commit(), "screen": screened,
                                                           "final": checked}, indent=1, default=str))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "")
