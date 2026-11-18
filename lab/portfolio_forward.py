"""Monthly forward looks of the frozen weak-signal portfolio (research/protocols/2026-11-18-weak-signal-portfolio.md;
tech lead 2026-11-18: "历史只作检查，前向序贯检验判定", "照做，序贯检验按夏普 1.5 设计").

    python -m lab.portfolio_forward --spec research/results/weak_portfolio_2026-11-18/forward_spec.json \\
        --end 2026-12-01 --data data_files/<the monthly re-selection's new snapshot> \\
        --out research/results/weak_portfolio_2026-11-18/looks/2026-12-01

- Refuses a look that is not registered in research/forward_looks.jsonl for this spec (by its SHA-256), registered
  on or after its end, or whose end date has not fully passed (``lab.forward.registered_look``); a spec whose forward
  start is not ``lab.forward.FORWARD_START``, or whose number of looks is not the registry's.
- Refuses data that is not whole: an hour missing in [forward start, end) for a symbol traded or read, or a member
  whose window is not full at the forward start (the snapshot's warm-up too short).
- Replays every frozen member (idea, parameters, stop) with the history check's code (``lab.um_judge.Judge``: the
  spec's capital split over its symbols, its leverage, the shared broker), flat at the forward start, to the end;
  combines them at equal capital, never rebalanced (``lab.um_portfolio.combine``); and applies Wald's sequential
  test (``lab.gates.sequential``) to the portfolio's daily returns from the forward start: H0 mean 0, H1 mean = the
  effect Sharpe / sqrt(365) x the frozen daily sd. Every look starts from the forward start again.
- The decision: "effect" or "no effect" once a bound is crossed (the tracking stops), "continue" while inside,
  "no conclusion" at the last registered look still inside.
- Writes ``result.json`` and ``README.md`` into a new directory: a look is evaluated once. Nothing here selects
  anything; the replays are not lab trials. Bars from the end on enter nothing: the replays end there and the
  members are causal (tests/unit/lab/test_lookahead.py).
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from lab import forward, gates
from lab import um_judge as judge
from lab import um_lab as lab
from lab import um_portfolio as portfolio
from trading.domain.strategy.base import Window

H, DAY = lab.H, lab.DAY
ROOT = Path(__file__).resolve().parents[1]
WORDS = {"effect": "有效：越过上界，停止跟踪", "no effect": "无效：越过下界，停止跟踪",
         "continue": "继续跟踪：还在两条界线之间", "no conclusion": "没有结论：最后一次查看还在两条界线之间，停止跟踪"}


def _date(t: int) -> str:
    return datetime.fromtimestamp(t / 1000, timezone.utc).date().isoformat()


def _spec(path: Path) -> dict[str, Any]:
    spec = json.loads(Path(path).read_text())
    if spec.get("passed") is not False or not spec.get("members") or "sequential" not in spec:
        raise ValueError(f"{path} is not a frozen portfolio spec")
    if forward.day_ms(spec["forward_start"]) != forward.FORWARD_START:
        raise ValueError(f"the spec's forward start {spec['forward_start']} is not the forward window's")
    return spec


def _number(looks: Path, study: str, digest: str, end: str, max_looks: int) -> int:
    """Which of the spec's registered looks this is (1-based); the registry must hold exactly ``max_looks``."""
    ends = sorted(r["end"] for r in forward.load_looks(looks) if (r["study"], r["spec_sha256"]) == (study, digest))
    if len(ends) != max_looks:
        raise ValueError(f"{len(ends)} looks are registered for this spec; it names {max_looks}")
    return ends.index(end) + 1


def _whole(snapshot: Path, symbols: list[str], start: int, end: int) -> None:
    hours = pd.RangeIndex(start, end, H)
    for symbol in symbols:
        if not (snapshot / f"{symbol}_1h.csv.gz").is_file():
            raise ValueError(f"{symbol} is not in {snapshot}: the data is not whole")
        missing = hours.difference(lab.frame(symbol, snapshot).index)
        if len(missing):
            raise ValueError(f"{symbol}: {len(missing)} hours missing from {_date(start)} to {_date(end)}: the data "
                             "is not whole")


def _ready(name: str, strategy: Any, symbols: list[str], snapshot: Path, start: int) -> None:
    """The member's window is full at its first forward decision (the close at the forward start)."""
    read = list(symbols) + [s for s in strategy.references() if s not in symbols]
    windows = {}
    for s in read:
        w = lab.snapshot_window(s, snapshot)
        windows[s] = Window(w.hourly[w.hourly["available_at"] <= start].reset_index(drop=True),
                            w.funding[w.funding.index < start])
    history = strategy.history(windows)
    for s in symbols:
        h = history[s]
        if not len(h) or int(h["available_at"].iloc[-1]) != start or not bool(h["ready"].iloc[-1]):
            raise ValueError(f"{name} is not ready on {s} at the forward start: the snapshot's warm-up is too short")


def _commit() -> str:
    return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


def look(spec_path: Path, *, end: str, data: Path, out: Path, looks: Path = forward.LOOKS, now_ms: int | None = None,
         ideas: Any = None) -> dict[str, Any]:
    spec_path, data, out = Path(spec_path), Path(data), Path(out)
    digest = forward.spec_digest(spec_path)
    study = json.loads(spec_path.read_text())["study"]
    registered = forward.registered_look(study, digest, end, path=looks, now_ms=now_ms)
    spec = _spec(spec_path)
    seq = spec["sequential"]
    number = _number(Path(looks), study, digest, end, int(seq["max_looks"]))
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} is not empty: a look is evaluated once")
    start, stop_at = forward.FORWARD_START, forward.day_ms(end)
    terms = lab.LiveTerms(capital=str(spec["terms"]["capital"]), leverage=str(spec["terms"]["leverage"]),
                          symbols=tuple(spec["terms"]["symbols"]))
    symbols = list(terms.symbols)
    ideas = lab.lab_ideas() if ideas is None else ideas
    j = judge.Judge(ideas=ideas, lists=[symbols], cache=out / "unused", run_date=f"forward {end}", snapshot=data,
                    terms=terms, slots=len(symbols), periods=((start, stop_at),), trials=out / "unused.jsonl")
    configs = [judge.Config(m["idea"], m["params"], judge.StopVariant(**m["stop"])) for m in spec["members"]]
    strategies = [ideas[c.idea][0](c.params) for c in configs]
    _whole(data, sorted(set(symbols) | {judge.BTC} | {r for s in strategies for r in s.references()}), start, stop_at)
    for config, strategy in zip(configs, strategies):
        _ready(config.idea, strategy, symbols, data, start)
    members, dailies = [], []
    for config in configs:
        r = j.run_config((config, (0,)))["periods"][0]
        dailies.append(r["daily"])
        members.append({"idea": config.idea, "params": config.params, "stop": config.label["stop"],
                        "total_return": float((1 + r["daily"]).prod() - 1), "round_trips": len(r["trips"]),
                        "fills": r["fills"], "exchange_stops": r["stops"]})
    daily = portfolio.combine(dailies)
    sd = float(seq["daily_sd"])
    effect = float(seq["effect_sharpe"]) / math.sqrt(365) * sd
    alpha, beta = float(seq["alpha"]), float(seq["beta"])
    test = gates.sequential(daily.to_numpy(), effect=effect, sd=sd, alpha=alpha, beta=beta)
    decision = test["decision"]
    if decision == "continue" and number == int(seq["max_looks"]):
        decision = "no conclusion"
    btc = lab.buy_and_hold(judge.BTC, start, stop_at, data)
    result = {"study": study, "end": end, "look": number, "of": int(seq["max_looks"]),
              "registered": registered["registered"], "spec": str(spec_path), "spec_sha256": digest,
              "data": str(data), "start": _date(start), "days": int(len(daily)), "code_commit": _commit(),
              "members": members, "portfolio": lab.summary(daily),
              "sequential": {**test, "effect_sharpe": float(seq["effect_sharpe"]), "effect_daily_mean": effect,
                             "daily_sd": sd, "alpha": alpha, "beta": beta,
                             "upper": math.log((1 - beta) / alpha), "lower": math.log(beta / (1 - alpha)),
                             "decided_on": None if test["at"] is None else _date(start + (test["at"] - 1) * DAY)},
              "decision": decision,
              "btc_hold_return": float((1 + btc).prod() - 1) if len(btc) else None}
    result = json.loads(json.dumps(result, default=float))
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    (out / "README.md").write_text(_notes(result))
    return result


def _notes(r: dict[str, Any]) -> str:
    s, p = r["sequential"], r["portfolio"]
    btc = "无数据" if r["btc_hold_return"] is None else f"{r['btc_hold_return']:+.2%}"
    crossed = f"，在 {s['decided_on']} 那天越过" if s["decided_on"] else ""
    lines = [
        f"# 弱信号组合前向查看：第 {r['look']}/{r['of']} 次，到 {r['end']}（不含）", "",
        "数字是【实际运行结果】，来自同目录的 `result.json`。**不构成实盘批准。**", "",
        f"- 判定：**{WORDS[r['decision']]}**{crossed}。",
        f"  - Wald 序贯检验，从 {r['start']} 起 {r['days']} 天的组合日收益；对数似然比 {s['llr']:.3f}，"
        f"上界 {s['upper']:.3f}，下界 {s['lower']:.3f}。",
        f"  - 有效 = 年化夏普 {s['effect_sharpe']}（日均 {s['effect_daily_mean']:.5f}，日标准差 {s['daily_sd']:.5f}，"
        f"冻结时定），α {s['alpha']}，β {s['beta']}。",
        f"- 组合（等资金，不再平衡）：收益 {p.get('total_return', 0.0):+.2%}，"
        f"最大回撤 {p.get('max_drawdown', 0.0):.2%}；同期持有 BTC：{btc}。",
        "- 成员："]
    lines += [f"  - {m['idea']} {json.dumps(m['params'], sort_keys=True)}：收益 {m['total_return']:+.2%}，往返 "
              f"{m['round_trips']} 笔，交易所止损 {m['exchange_stops']} 次" for m in r["members"]]
    lines += ["- 按协议，\"有效\"只说明可以准备小资金实盘：要不要上、上多少，由技术负责人决定。", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = look(args.spec, end=args.end, data=args.data, out=args.out)
    print(json.dumps({k: result[k] for k in ("look", "of", "days", "decision")}, ensure_ascii=False))


if __name__ == "__main__":
    main()


__all__ = ["look"]
