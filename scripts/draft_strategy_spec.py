#!/usr/bin/env python3
"""A lab champion's card -> a schema-2 spec draft (tech lead 2026-11-17: the monthly re-selection proposes a new spec;
it is tracked forward 4 weeks, then a person publishes it and marks it passed).

    python3 -m scripts.draft_strategy_spec --run research/lab/runs/<date> --snapshot data_files/<snapshot> \\
        [--idea NAME] [--allow-failed] [--out research/lab/runs/<date>/spec_draft]

- The draft: the judge's final choice (``results.json``), the run's latest symbol list (``universe.json``), the lab's
  execution settings, ``allow_short`` when the strategy's targets on that list go below 0, and ``passed: false``
  always. It must parse with the runtime's ``parse_spec``. Nothing is signed: ``scripts.publish_strategy_spec`` does
  that, by a person.
- README.md beside it: the runtime settings the backtest assumed and every stop rule the runtime lacks (a gap). Exit
  code 2 when there is a gap: the runtime cannot run the strategy as it was judged until the code changes.
"""
from __future__ import annotations

import argparse
import json
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any

from lab import um_lab as lab
from lab.strategy import StrategySpec
from trading.domain.issues import AFTER_STOP_PAUSE_THEN_RESET, STOP_PAUSE_SECONDS
from trading.domain.risk.liquidation import liquidation_distance
from trading.domain.strategy.spec import KIND_V2, parse_spec

MMR = 0.005                  # the maintenance margin rate check_stop_inside_liquidation assumes
PAUSE_HOURS = STOP_PAUSE_SECONDS // 3600


def largest_leverage(stop_pct: str) -> int:
    """The largest isolated leverage whose half liquidation distance still holds the stop (the runtime's rule)."""
    stop = float(Decimal(stop_pct))
    return max((lev for lev in range(1, 21) if stop <= 0.5 * liquidation_distance(lev, MMR)), default=0)


def _gaps(stop: dict) -> list[str]:
    if Decimal(stop["stop_pct"]) == 0:
        return ["no stop: the runtime always places an exchange stop"]
    if stop.get("vol_k") is not None:
        return [f"vol stop (k={stop['vol_k']}): the runtime's stop is a fixed STOP_PCT"]
    if stop.get("reenter") == "signal_reset":
        return ["signal reset after a stop without a pause: the runtime's after_stop always pauses first "
                "(pause_then_reset, tech lead 2026-11-17)"]
    if stop.get("pause_hours") != PAUSE_HOURS:
        return [f"a {stop.get('pause_hours')}-hour pause after a stop: the runtime's pause is {PAUSE_HOURS} hours"]
    return []


def _shorts(strategy: Any, symbols: list[str], snapshot: Path) -> bool:
    windows = {s: lab.snapshot_window(s, snapshot) for s in [*symbols, *strategy.references()]}
    if strategy.cross_sectional:
        frames = strategy.history(windows)
    else:
        frames = {s: strategy.history({s: windows[s], **{r: windows[r] for r in strategy.references()}})[s]
                  for s in symbols}
    return any((frames[s]["target"] < 0).any() for s in symbols)


def draft(results: dict, lists: list[list[str]], snapshot: Path, *, idea: str | None = None,
          allow_failed: bool = False) -> tuple[dict, dict, list[str]]:
    """(spec, runtime settings, gaps) for ``idea`` (default: the champion)."""
    final = results["final"]
    name = idea or final.get("champion")
    if not name:
        raise ValueError("the run has no champion; name an idea with --idea")
    card = final["final_check"].get(name)
    if card is None:
        raise ValueError(f"{name} is not in the final check of this run")
    if not card["passes"] and not allow_failed:
        raise ValueError(f"{name} did not pass the final check; --allow-failed drafts it anyway")
    choice, symbols = card["final_choice"], list(lists[-1])
    strategy = lab.lab_ideas()[name][0](choice["params"])
    outside = [s for s in strategy.references() if s not in symbols]
    if outside:
        raise ValueError(f"{name} reads {', '.join(outside)}, outside the list it trades: the runtime cannot run it")
    execution = StrategySpec(execution="market")
    spec = {"schema": 2, "kind": KIND_V2, "strategy": {"name": name, "params": choice["params"]},
            "execution": {"band": execution.band, "execution": "market", "ttl_bars": execution.ttl_bars},
            "symbols": symbols, "allow_short": _shorts(strategy, symbols, snapshot), "passed": False}
    if choice["stop"].get("reenter") == AFTER_STOP_PAUSE_THEN_RESET:       # the runtime's stricter rule (2026-11-17)
        spec["after_stop"] = AFTER_STOP_PAUSE_THEN_RESET
    parse_spec(spec)
    stop = choice["stop"]
    settings = {"STOP_PCT": stop["stop_pct"], "BINANCE_EXCHANGE_LEVERAGE (largest)": largest_leverage(stop["stop_pct"]),
                "BINANCE_MARGIN_TYPE": "ISOLATED", "STRATEGY_CAPITAL_USDT": lab.LIVE.capital, "STRATEGY_LEVERAGE": 1}
    return spec, settings, _gaps(stop)


def _notes(spec: dict, settings: dict, gaps: list[str], run: Path, passes: bool) -> str:
    lines = [f"# 规格草稿：{spec['strategy']['name']}（{run.name}）", "",
             "**未验证。不构成实盘批准。** `passed` 是 false；签名发布和标记通过只由技术负责人做。", "",
             "- 来源：`results.json` 的最终选择和 `universe.json` 最后一份清单。" + ("" if passes else
             "**这个想法没有通过最终检查**（用 --allow-failed 生成）。"),
             "- 上实盘之前：前向跟踪 4 周，每周一次登记的查看，4 次都还成立才进入小资金试运行。", "",
             "## 回测假设的实盘设置", ""]
    lines += [f"- `{k}={v}`" if "largest" not in k else f"- 交易所杠杆最多 {v} 倍（止损要在逐仓强平距离的一半以内）"
              for k, v in settings.items()]
    lines += ["", "## 实盘程序还不支持的（要先改程序）", ""] + ([f"- {g}" for g in gaps] or ["- 无"])
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--idea")
    parser.add_argument("--allow-failed", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    out = args.out or args.run / "spec_draft"
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} is not empty; a draft is written once")
    results = json.loads((args.run / "results.json").read_text())
    lists = json.loads((args.run / "universe.json").read_text())["lists"]
    spec, settings, gaps = draft(results, lists, args.snapshot, idea=args.idea, allow_failed=args.allow_failed)
    passes = results["final"]["final_check"][spec["strategy"]["name"]]["passes"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "final_spec.json").write_text(json.dumps(spec, indent=1) + "\n")
    (out / "README.md").write_text(_notes(spec, settings, gaps, args.run, passes))
    print(f"draft written to {out}; gaps: {len(gaps)}")
    if gaps:
        sys.exit(2)


if __name__ == "__main__":
    main()
