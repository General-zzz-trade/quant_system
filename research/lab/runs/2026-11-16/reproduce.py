"""Regression check of lab/um_judge.py against this run's recorded numbers (not new research).

    python research/lab/runs/2026-11-16/reproduce.py      # -> reproduction.json; exit 1 when a number differs

The tool that replaces judge.py must give judge.py's numbers. Checked here: the five yearly symbol lists
(universe.json; the last one uses only bars from before 2026-05-13) and, for A6_voladj_mom and D2_taker_imbalance
with the 8 % stop and the 4-hour pause, the walk-forward numbers of results.json. Only the selection periods are
replayed: the holdout period is not read.

judge.py left the first day of every period out of the daily returns (and ended a day at 23:00); the tool counts
every day. So the comparison runs with ``legacy_days=True``, and the same six configurations are replayed once
more the tool's own way to show what the difference is. Every replay is in research/lab/trials.jsonl, marked as a
reproduction.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3]))
sys.path.insert(0, str(HERE))
import ideas as lib  # noqa: E402
from lab import um_judge as judge  # noqa: E402
from lab import um_lab as lab  # noqa: E402

NAMES = ("A6_voladj_mom", "D2_taker_imbalance")
CACHE = Path("/tmp/lab_judge_reproduce_2026-11-16")
NOTE = "reproduction of research/lab/runs/2026-11-16 with lab/um_judge.py: a regression check, not a new test"


def _judge(lists: list, legacy: bool) -> judge.Judge:
    how = "days as judge.py counted them" if legacy else "every day of a period counted"
    return judge.Judge(ideas=lib.IDEAS, lists=lists, cache=CACHE / ("legacy" if legacy else "days"),
                       run_date="2026-11-16", cross_sectional=frozenset(lib.XS), note=f"{NOTE}; {how}",
                       legacy_days=legacy)


def _compare(expected: dict, screened: dict) -> list[dict]:
    """One row per recorded number: the four the check is about first (``required``), then the rest."""
    rows = []

    def add(idea, number, want, got, required=False):
        rows.append({"idea": idea, "number": number, "expected": want, "obtained": got, "match": want == got,
                     "required": required})
    for name in NAMES:
        old, new = expected["screen"][name], screened["ideas"][name]
        for key in ("total_return", "sharpe", "max_drawdown"):
            add(name, f"walk_forward.{key}", old["walk_forward"][key], new["walk_forward"][key], True)
        add(name, "round_trips", old["round_trips"], new["round_trips"], True)
        for key in ("days", "cagr", "worst_month", "worst_month_label", "longest_losing_days"):
            add(name, f"walk_forward.{key}", old["walk_forward"][key], new["walk_forward"][key])
        add(name, "chosen_per_fold", old["chosen_per_fold"], [c["params"] for c in new["chosen_per_fold"]])
        add(name, "final_params", old["final_params"], new["final_choice"]["params"])
        add(name, "selection_sharpe_by_params", sorted(old["selection_sharpe_by_params"].values()),
            sorted(new["selection_sharpe"].values()))
        for key in ("regime", "longest_losing_trades", "exchange_stops", "score", "survives", "limits_needed"):
            add(name, key, old[key], new[key])
        for key in ("losing_streak_stops", "per_30_days", "days_below_daily_stop"):
            add(name, f"pilot.{key}", old["pilot"][key], new["pilot"][key])
    hold = expected["buy_and_hold"]
    add("buy and hold BTC", "walk_forward", hold["walk_forward_btc"], screened["buy_and_hold_btc"])
    add("buy and hold the list", "walk_forward", hold["walk_forward_list"], screened["buy_and_hold_list"])
    return rows


def main() -> int:
    expected = json.loads((HERE / "results.json").read_text())
    recorded = json.loads((HERE / "universe.json").read_text())
    lists = judge.universe(lab.SNAPSHOT, [start for start, _ in judge.PERIODS], capital=lab.LIVE.capital, slots=10)
    record = judge.universe_record(lists, [start for start, _ in judge.PERIODS], capital=lab.LIVE.capital, slots=10)
    print("universe", "matches" if record == recorded else "DIFFERS", flush=True)

    rows = _compare(expected, _judge(recorded["lists"], legacy=True).screen(NAMES))
    for row in rows:
        if row["required"] or not row["match"]:
            print(f"{row['idea']:22s} {row['number']:28s} expected {row['expected']!r} obtained {row['obtained']!r} "
                  f"{'ok' if row['match'] else 'DIFFERS'}", flush=True)
    counted = _judge(recorded["lists"], legacy=False).screen(NAMES)
    every_day = {"buy_and_hold_btc": counted["buy_and_hold_btc"],
                 "ideas": {name: {key: counted["ideas"][name][key] for key in (
                     "walk_forward", "round_trips", "chosen_per_fold", "exchange_stops", "score", "survives")}
                     for name in NAMES}}
    for name in NAMES:
        wf = every_day["ideas"][name]["walk_forward"]
        print(f"{name:22s} every day counted: days {wf['days']} return {wf['total_return']!r} sharpe "
              f"{wf['sharpe']!r} max drawdown {wf['max_drawdown']!r}", flush=True)
    same = record == recorded and all(row["match"] for row in rows)
    (HERE / "reproduction.json").write_text(json.dumps({
        "what": "lab/um_judge.py against results.json and universe.json of this run; selection periods only, the "
                "holdout period was not read",
        "snapshot": lab.SNAPSHOT.name, "ideas": list(NAMES), "stop": judge.LIVE_STOP.name,
        "universe": {"matches": record == recorded, "obtained": record},
        "all_match": same, "numbers": rows,
        "with_every_day_counted": {
            "note": "the same configurations with lab/um_judge.py's own days: each period's first day (and the cost "
                    "of entering) is in the returns and a day ends at midnight; judge.py left the first day out",
            **every_day}}, indent=1) + "\n")
    print("all recorded numbers reproduced" if same else "DIFFERENCES: see reproduction.json")
    return 0 if same else 1


if __name__ == "__main__":
    raise SystemExit(main())
