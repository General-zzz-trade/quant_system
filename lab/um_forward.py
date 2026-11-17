"""Weekly forward looks of a frozen lab champion (tech lead 2026-11-16 evening: frozen, tracked 4 weeks on new data, one
registered look a week, each on a new snapshot; "still holds" set 2026-11-17).

    python -m lab.um_forward download --frozen research/lab/runs/2026-11-17/forward_e3.json \\
        --study lab_2026-11-17:E3_vol_target_trend --end 2026-11-25 --out data_files/forward_e3_2026-11-25
    python -m lab.um_forward look --frozen research/lab/runs/2026-11-17/forward_e3.json \\
        --study lab_2026-11-17:E3_vol_target_trend --end 2026-11-25 --data data_files/forward_e3_2026-11-25 \\
        --out research/lab/forward/e3/2026-11-25

- Both refuse a look that is not registered in research/forward_looks.jsonl for this frozen file (by its SHA-256),
  registered on or after its end, or whose end date has not fully passed (``lab.forward.registered_look``).
- ``download``: a new snapshot of the frozen symbols and BTCUSDT, hourly bars and funding from ``--start`` (warm-up;
  default 2026-09-01) to the look's end.
- ``look``: replays the frozen configuration (strategy, parameters, symbols, stop, capital split 1/N) from the frozen
  start to the look's end through the shared broker, every look from the start again. The criterion: the lowest
  hourly portfolio equity since the start stays above 92.5 % of the capital (``HOLDS``). It writes ``result.json``
  and ``README.md`` into a new directory: a look is evaluated once.
- Nothing here selects anything; the replays are not lab trials.
"""
from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timezone
from decimal import Decimal as D
from pathlib import Path
from typing import Any

import pandas as pd

from lab import forward
from lab import um_judge as judge
from lab import um_lab as lab

HOLDS = 0.925                     # tech lead 2026-11-17: a cumulative loss of less than 7.5 %, half the loss stop
WARMUP_START = "2026-09-01"
H = lab.H


def _frozen(path: Path) -> dict[str, Any]:
    frozen = json.loads(Path(path).read_text())
    if frozen.get("kind") != "lab_champion_forward" or frozen.get("passed") is not False:
        raise ValueError(f"{path} is not a frozen lab champion")
    return frozen


def _check(frozen_path: Path, study: str, end: str, now_ms: int | None, looks: Path) -> dict[str, Any]:
    return forward.registered_look(study, forward.spec_digest(Path(frozen_path)), end, path=looks, now_ms=now_ms)


def download(frozen_path: Path, *, study: str, end: str, out: Path, start: str = WARMUP_START,
             looks: Path = forward.LOOKS, now_ms: int | None = None, fetch: Any = None) -> dict:
    from data.downloads import binance_um_all as um
    _check(frozen_path, study, end, now_ms, looks)
    frozen = _frozen(frozen_path)
    symbols = sorted(set(frozen["spec"]["symbols"]) | {judge.BTC})
    today = datetime.fromtimestamp((now_ms if now_ms is not None else lab.ms(date.today().isoformat())) / 1000,
                                   timezone.utc).date()
    kwargs = {"fetch": fetch} if fetch is not None else {}
    return um.download(Path(out), start=date.fromisoformat(start), end=date.fromisoformat(end), symbols=symbols,
                       forward_look=True, today=today, **kwargs)


def look(frozen_path: Path, *, study: str, end: str, data: Path, out: Path, looks: Path = forward.LOOKS,
         now_ms: int | None = None) -> dict[str, Any]:
    registered = _check(frozen_path, study, end, now_ms, looks)
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} is not empty: a look is evaluated once")
    frozen = _frozen(frozen_path)
    start, stop_at = lab.ms(frozen["start"]), lab.ms(end)
    name, params = frozen["spec"]["strategy"]["name"], frozen["spec"]["strategy"]["params"]
    symbols = list(frozen["spec"]["symbols"])
    capital = D(str(frozen["capital_usdt"]))
    slot = capital / int(frozen["slots"])
    stop = judge.StopVariant(**frozen["stop"])
    j = judge.Judge(ideas=lab.lab_ideas(), lists=[symbols], cache=out / "unused", run_date=f"forward {end}",
                    snapshot=Path(data), periods=((start, stop_at),), trials=out / "unused.jsonl")
    strategy = j.ideas[name][0](params)
    hours = pd.RangeIndex(start, stop_at, H)
    equity = pd.Series(0.0, index=hours)
    trips, fills, stops = 0, 0, 0
    targets: dict[tuple[str, tuple[str, ...]], pd.Series] = {}
    for symbol in symbols:
        given = tuple(symbols) if strategy.cross_sectional else (symbol,)
        if (symbol, given) not in targets:
            targets.update({(s, given): t for s, t in j._targets(strategy, given).items()})
        weights = targets[(symbol, given)].reindex(lab.frame(symbol, Path(data)).index).fillna(0.0)
        replayed = lab.replay(symbol, weights, start=start, end=stop_at, snapshot=Path(data),
                              starting_balance=str(slot), **stop.replay_terms())
        curve = replayed["curve"]
        # equity at each bar's close, by the bar's open time; before the first close and when idle: the slot's capital
        slot_equity = curve.set_axis(curve.index - H).reindex(hours).ffill().fillna(float(slot))
        equity = equity.add(slot_equity, fill_value=0.0)
        trips += len(lab.round_trips(replayed["fills"]))
        fills += len(replayed["fills"])
        stops += sum(1 for f in replayed["fills"] if f["order"] == "protective_stop")
    lowest = float(equity.min() / float(capital))
    btc = lab.buy_and_hold(judge.BTC, start, stop_at, Path(data))
    result = {"study": study, "end": end, "registered": registered["registered"], "frozen": str(frozen_path),
              "frozen_sha256": forward.spec_digest(Path(frozen_path)), "data": str(data), "start": frozen["start"],
              "capital": str(capital), "final_equity": float(equity.iloc[-1]),
              "total_return": float(equity.iloc[-1] / float(capital) - 1), "lowest_equity_share": lowest,
              "holds": lowest > HOLDS, "round_trips": trips, "fills": fills, "exchange_stops": stops,
              "btc_hold_return": float((1 + btc).prod() - 1) if len(btc) else None}
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    (out / "README.md").write_text(_notes(result))
    return result


def _notes(r: dict[str, Any]) -> str:
    verdict = "还成立" if r["holds"] else "不成立：跟踪结束，不上实盘"
    btc = "无数据" if r["btc_hold_return"] is None else f"{r['btc_hold_return']:+.2%}"
    lines = [
        f"# 前向查看：{r['study']}，到 {r['end']}（不含）", "",
        "数字是【实际运行结果】，来自同目录的 `result.json`。**不构成实盘批准。**", "",
        f"- 结论：**{verdict}**。",
        f"  标准是从 {r['start']} 起每小时权益的最低点高于起始资金的 92.5%（技术负责人 2026-11-17）。",
        f"- 最低点：起始资金的 {r['lowest_equity_share']:.2%}。",
        f"- 期末权益 {r['final_equity']:.2f} USDT，收益 {r['total_return']:+.2%}；同期持有 BTC：{btc}。",
        f"- 往返 {r['round_trips']} 笔，成交 {r['fills']} 笔，交易所止损 {r['exchange_stops']} 次。",
        "- 这么短的时间，收益说明不了有没有优势；这条标准只挡住明显坏掉的策略。", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=("download", "look"))
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--study", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--start", default=WARMUP_START)
    args = parser.parse_args(argv)
    if args.step == "download":
        manifest = download(args.frozen, study=args.study, end=args.end, out=args.out, start=args.start)
        print(f"{manifest['symbols_with_bars']} symbols written to {args.out}; failed {manifest['failed']}")
    else:
        if args.data is None:
            raise SystemExit("look needs --data")
        result = look(args.frozen, study=args.study, end=args.end, data=args.data, out=args.out)
        print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()


__all__ = ["HOLDS", "download", "look"]
