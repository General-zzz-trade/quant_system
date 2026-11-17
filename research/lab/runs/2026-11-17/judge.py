"""The 2026-11-17 lab run (windows.md, registered before any result): the 29 registered ideas x the 12 stop variants
on the 2020+ snapshot, then the final check of the survivors.

    python research/lab/runs/2026-11-17/judge.py screen     # selection and walk-forward; the holdout is not read
    python research/lab/runs/2026-11-17/judge.py final      # survivors only: the holdout, doubled costs, look-ahead

Every replay is registered in research/lab/trials.jsonl and cached under /root/lab_cache/2026-11-17.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
from lab import um_judge as judge  # noqa: E402
from lab import um_lab as lab  # noqa: E402
from lab.causality import violations  # noqa: E402
from trading.domain.strategy.base import Window  # noqa: E402

SNAP = ROOT / "data_files" / "binance_usdt_perps_2020_2026-11-12"
START, END = lab.ms("2020-01-01"), lab.ms("2026-11-13")
FOLDS = tuple((lab.ms(f"{y}-05-13"), lab.ms(f"{y + 1}-05-13")) for y in range(2021, 2026))
PERIODS = judge.periods(START, FOLDS, END)
CACHE = Path("/root/lab_cache/2026-11-17")
NOTE = "2026-11-17 run on the 2020+ snapshot (runs/2026-11-17/windows.md)"
H = lab.H


def lists() -> list[list[str]]:
    path = HERE / "universe.json"
    if path.is_file():
        return json.loads(path.read_text())["lists"]
    dates = [start for start, _ in PERIODS]
    found = judge.universe(SNAP, dates, capital=lab.LIVE.capital, slots=10)
    record = judge.universe_record(found, dates, capital=lab.LIVE.capital, slots=10)
    path.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    return found


def causal(name: str, params: dict) -> list[str]:
    """lab/causality.py on real data through the strategy interface: the first five symbols of the 2023 list (and
    BTCUSDT when the idea reads it), 2023-01..2025-01; outputs must not change when later inputs change or vanish."""
    strategy = lab.lab_ideas()[name][0](params)
    symbols = list(lists()[3][:5]) + [s for s in strategy.references() if s not in lists()[3][:5]]
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
    return judge.Judge(ideas=lab.lab_ideas(), lists=lists(), cache=CACHE, run_date="2026-11-17", snapshot=SNAP,
                       periods=PERIODS, note=NOTE, lookahead=causal)


def commit() -> str:
    return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


def main(step: str) -> None:
    j = make()
    if step == "screen":
        screened = j.screen(variants=judge.STOP_VARIANTS, processes=2)
        (HERE / "screen.json").write_text(json.dumps({"code_commit": commit(), **screened}, indent=1, default=str))
    elif step == "final":
        screened = json.loads((HERE / "screen.json").read_text())
        checked = j.final(screened, processes=2)
        (HERE / "results.json").write_text(json.dumps({"code_commit": commit(), "screen": screened, "final": checked},
                                                      indent=1, default=str))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "")
