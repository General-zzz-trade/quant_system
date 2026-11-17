"""The live runtime's new rule in the lab (pause_then_reset.md, registered before any result): the top 10 ideas of
judge.md with "8 % stop, pause 4 hours, then wait for the target to return to 0".

    python research/lab/runs/2026-11-17/pause_then_reset.py     # -> pause_then_reset.json

Same data, periods, lists, costs and cache as judge.py; every replay goes to research/lab/trials.jsonl.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import judge as run  # noqa: E402  (this run's judge.py)
from lab import um_judge as judge  # noqa: E402
from stop_stability import _walk  # noqa: E402

TOP = ("D2_taker_imbalance", "E3_vol_target_trend", "F1_btc_trend_beta", "C5_trend_not_crowded", "A6_voladj_mom",
       "C2_funding_negative_long", "E1_squeeze_breakout", "A1_tsmom_long", "A5_btc_gate_tsmom", "E2_calm_trend")
NEW = judge.StopVariant("0.08", 4, "pause_then_reset")
LIVE = judge.StopVariant("0.08", 4)
RESET = judge.StopVariant("0.08", None, "signal_reset")


def main() -> None:
    j = run.make()
    j.screen(names=list(TOP), variants=(NEW,), processes=2)          # replays and caches the new configurations
    out = {}
    for name in TOP:
        rows = {v.name: _walk(j, name, [v]) for v in (LIVE, RESET, NEW)}
        identical = []
        for params in j.ideas[name][1]:
            a, b = j.load(judge.Config(name, params, NEW)), j.load(judge.Config(name, params, RESET))
            identical.append(all(a[k]["daily"].equals(b[k]["daily"]) and
                                 (a[k]["trips"], a[k]["stops"], a[k]["fills"]) == (b[k]["trips"], b[k]["stops"],
                                                                                   b[k]["fills"]) for k in a))
        out[name] = {"walk_forward": {k: {"sharpe": v["sharpe"], "total_return": v["total_return"],
                                          "folds": v["folds"]} for k, v in rows.items()},
                     "identical_to_signal_reset_per_params": identical}
    (HERE / "pause_then_reset.json").write_text(json.dumps({"code_commit": run.commit(), "ideas": out}, indent=1))


if __name__ == "__main__":
    main()
