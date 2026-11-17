"""The lab's monthly re-selection (tech lead 2026-11-17: "每月一次，实验室读新数据").

    python -m data.downloads.binance_um_all --monthly-lab --start 2020-01-01 --end <first day of this month> \\
        --out data_files/binance_usdt_perps_2020_<the last day it holds>
    python -m lab.um_monthly plan   --snapshot <dir> --run research/lab/runs/<date>   # windows.md; commit it
    python -m lab.um_monthly screen --snapshot <dir> --run research/lab/runs/<date>   # selection and walk-forward
    python -m lab.um_monthly final  --snapshot <dir> --run research/lab/runs/<date>   # survivors: the holdout

- Periods (``monthly_periods``): the holdout is the 6 months before the snapshot's end. Before it, whole yearly folds,
  as far back as leave at least a year of selection after the 30 days the list rule needs; period 0 is the rest from
  2020-01-01. On the 2026-11-12 snapshot these are the 2026-11-17 run's periods.
- ``plan`` writes the run's windows.md once; ``screen`` and ``final`` run only when it is committed and unchanged, so
  the windows are registered before any result.
- The ideas are the registered ones (``lab.um_lab.lab_ideas``), the stop variants ``lab.um_judge.STOP_VARIANTS``,
  the rest as the 2026-11-17 run (runs/2026-11-17/windows.md). Every replay goes to research/lab/trials.jsonl.
- A holdout overlaps the previous months' holdouts. It is opened only for survivors, and each opening is recorded;
  the test on data nobody has seen is the 4 weeks of forward tracking a survivor gets before any money.
- The four registered forward tests stay frozen and are judged at their registered looks.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import date, timedelta
from pathlib import Path

from lab import um_judge as judge
from lab import um_lab as lab
from lab.causality import violations
from trading.domain.strategy.base import Window

FIRST_DAY = date(2020, 1, 1)
HOLDOUT_MONTHS = 6
MIN_SELECTION_DAYS = 365
H, DAY = lab.H, lab.DAY


def _ms(day: date) -> int:
    return lab.ms(day.isoformat())


def _months_before(day: date, months: int) -> date:
    index = day.year * 12 + day.month - 1 - months
    return date(index // 12, index % 12 + 1, day.day)


def monthly_periods(end: date) -> tuple[tuple[int, int], ...]:
    holdout = _months_before(end, HOLDOUT_MONTHS)
    first = FIRST_DAY + timedelta(days=30)                       # the list rule's 30 days
    starts = []
    year = 1
    while (start := date(holdout.year - year, holdout.month, holdout.day)) - first >= timedelta(MIN_SELECTION_DAYS):
        starts.append(start)
        year += 1
    if not starts:
        raise ValueError(f"no whole fold fits before the holdout {holdout}")
    starts.reverse()
    folds = tuple((_ms(s), _ms(date(s.year + 1, s.month, s.day))) for s in starts)
    return judge.periods(_ms(FIRST_DAY), folds, _ms(end))


def snapshot_end(snapshot: Path) -> date:
    first, end = json.loads((Path(snapshot) / "manifest.json").read_text())["window"]
    if date.fromisoformat(first) != FIRST_DAY:
        raise ValueError(f"a monthly snapshot starts at {FIRST_DAY}, not {first}")
    return date.fromisoformat(end)


def _configs() -> int:
    return sum(len(grid) for _, grid in lab.lab_ideas().values()) * len(judge.STOP_VARIANTS)


def plan(snapshot: Path, run: Path) -> Path:
    """The run's windows.md, written once; commit it before ``screen``."""
    path = Path(run) / "windows.md"
    if path.exists():
        raise FileExistsError(f"{path} exists; a run's windows are registered once")
    end = snapshot_end(snapshot)
    periods = monthly_periods(end)
    uses = ["只用来挑选"] + ["滚动前推"] * (len(periods) - 2) + ["验证段：只对过淘汰线的想法打开"]
    rows = "\n".join(f"| {k} | {judge._date(a)} | {judge._date(b)} | {use} |"
                     for k, ((a, b), use) in enumerate(zip(periods, uses)))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"""# {Path(run).name} 月度重选：窗口和做法（由 `lab/um_monthly.py plan` 在看到任何结果之前写出）

技术负责人 2026-11-17：实验室每月一次在新数据上重新挑选。本文件提交之后才跑回测，之后不改。

## 数据

- 快照 `{Path(snapshot).name}`：截止 {end}（不含），全部 USDT 保证金永续（含已下架），`manifest.json` 记来源和缺失。

## 时段（UTC 零点，前后相接）

| 段 | 起 | 止（不含） | 用途 |
|---|---|---|---|
{rows}

- 每一折用它之前所有段里夏普最高的配置（想法的参数和止损方案一起选）。
- 验证段和前几个月的验证段重叠；只对过淘汰线的想法打开，每次打开都记录。真正没见过的数据是之后 4 周的前向跟踪。

## 配置

- 想法：`trading/domain/strategy/ideas.py` 登记的 {len(lab.lab_ideas())} 个；止损方案：`lab/um_judge.STOP_VARIANTS` 的
  {len(judge.STOP_VARIANTS)} 种；共 {_configs()} 个配置。
- 其他照 `runs/2026-11-17/windows.md`：资金 300 USDT、10 份、每段开始时换清单、VIP0 手续费、实际资金费、
  市价 3 个基点和止损 10 个基点滑点、含已下架合约、只在策略窗口满了之后持仓；
  淘汰线和最终检查照 `runs/2026-11-16/ideas.md`。
- 每个回放过的配置和时段都登记在 `research/lab/trials.jsonl`。
""")
    return path


def registered(snapshot: Path, run: Path) -> None:
    """Refuse unless windows.md is committed, unchanged and written for this snapshot."""
    path = Path(run) / "windows.md"
    if not path.is_file():
        raise SystemExit(f"no {path}: run `plan` first and commit it")
    git = ["git", "-C", str(path.parent)]
    tracked = subprocess.run([*git, "ls-files", "--error-unmatch", path.name], capture_output=True).returncode == 0
    unchanged = subprocess.run([*git, "diff", "--quiet", "HEAD", "--", path.name], capture_output=True).returncode == 0
    if not (tracked and unchanged):
        raise SystemExit(f"{path} must be committed, unchanged, before any result")
    if f"快照 `{Path(snapshot).name}`" not in path.read_text():
        raise SystemExit(f"{path} registers another snapshot than {Path(snapshot).name}")


class MonthlyRun:
    def __init__(self, snapshot: Path, run: Path, cache: Path | None = None) -> None:
        self.snapshot, self.run = Path(snapshot), Path(run)
        self.periods = monthly_periods(snapshot_end(self.snapshot))
        self.cache = cache or Path.home() / "lab_cache" / self.run.name

    def lists(self) -> list[list[str]]:
        path = self.run / "universe.json"
        if path.is_file():
            return json.loads(path.read_text())["lists"]
        dates = [start for start, _ in self.periods]
        found = judge.universe(self.snapshot, dates, capital=lab.LIVE.capital, slots=10)
        record = judge.universe_record(found, dates, capital=lab.LIVE.capital, slots=10)
        path.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
        return found

    def causal(self, name: str, params: dict) -> list[str]:
        """lab/causality.py on real data through the strategy interface: the last fold's first five symbols (and what
        the idea reads besides), the two years before the holdout, two decision times inside them."""
        strategy = lab.lab_ideas()[name][0](params)
        last = list(self.lists()[-2][:5])
        symbols = last + [s for s in strategy.references() if s not in last]
        hi = self.periods[-1][0]
        lo = hi - 730 * DAY
        inputs = {}
        for s in symbols:
            w = lab.snapshot_window(s, self.snapshot)
            hourly = w.hourly[(w.hourly["open_time"] >= lo) & (w.hourly["open_time"] < hi)].reset_index(drop=True)
            inputs[s] = {"hourly": hourly, "funding": w.funding[(w.funding.index >= lo) & (w.funding.index < hi)]}

        def replay(inp: dict) -> dict:
            return strategy.history({s: Window(v["hourly"].reset_index(drop=True), v["funding"])
                                     for s, v in inp.items()})
        try:
            return violations(replay, inputs, [hi - 400 * DAY + 7 * H, hi - 150 * DAY + 23 * H])
        except Exception as exc:
            return [f"{type(exc).__name__}: {exc}"[:200]]

    def judge(self) -> judge.Judge:
        return judge.Judge(ideas=lab.lab_ideas(), lists=self.lists(), cache=self.cache, run_date=self.run.name,
                           snapshot=self.snapshot, periods=self.periods, lookahead=self.causal,
                           note=f"monthly re-selection {self.run.name} (lab/um_monthly.py, {self.run}/windows.md)")


def _commit() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=("plan", "screen", "final"))
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--processes", type=int, default=2)
    args = parser.parse_args(argv)
    if args.step == "plan":
        print(plan(args.snapshot, args.run))
        return
    registered(args.snapshot, args.run)
    j = MonthlyRun(args.snapshot, args.run).judge()
    if args.step == "screen":
        screened = j.screen(variants=judge.STOP_VARIANTS, processes=args.processes)
        (args.run / "screen.json").write_text(json.dumps({"code_commit": _commit(), **screened}, indent=1,
                                                         default=str))
    else:
        screened = json.loads((args.run / "screen.json").read_text())
        checked = j.final(screened, processes=args.processes)
        (args.run / "results.json").write_text(json.dumps({"code_commit": _commit(), "screen": screened,
                                                           "final": checked}, indent=1, default=str))


if __name__ == "__main__":
    main()


__all__ = ["MonthlyRun", "monthly_periods", "plan", "registered", "snapshot_end"]
