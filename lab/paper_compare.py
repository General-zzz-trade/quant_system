"""Compare a paper record with research on the same bars, fill by fill (S11, design 2026-11 M5, G6).

    python -m lab.paper_compare --paper-dir data/runtime/paper --data-dir data_files/binance_public_<date>

Paper runs the execution path on the simulated exchange (ADR-0013 plan B); research runs ``lab.simulate`` (the
shared research broker) with the same spec. The strategy is the one the paper record names (the overlay, or from R7d
any registered strategy): every symbol's targets come from one call of it on the bars paper saw, as the runtime
decides them, so a cross-sectional strategy is compared too. Over the same bars their fills must be the same, except
where paper left research on purpose (a ``paper_divergence`` event). This tool says which, and never says "equal"
when unsure.

It reads, read-only: the paper directory's ``state.sqlite`` in one read transaction (the saved paper state and the
order journal's fills and intents), its event log (the record's first warm-up, the ``paper_divergence`` events) and
a dated data snapshot (``python -m data.downloads.binance_public``) for the bars and funding settlements, which paper
read from Binance REST and records nowhere. A snapshot is immutable, so a comparison can be repeated; it must hold
what paper saw (every close an order was sent at, every funding settlement applied and the last close are checked).
Paper's fills are the trades the order journal booked, each matched with the simulated exchange's fill row for its
time. Research runs over the record's first warm-up window and trades through each symbol's last decided bar.

Verdicts and exit codes:
- 0 "一致": each symbol's fills are equal, or equal up to a difference an intended divergence explains;
- 1 "不一致": a symbol's first differing fill, which no intended divergence explains (both sides are printed);
- 2 "无法比较": something the comparison needs is missing or unusable. Never "一致".

An intended divergence explains a symbol's first differing fill only through a fact of the paper record, and only
with its event; from there on the symbol is not compared:
- a paper fill research does not have, of a protective stop (``stop_triggered``) or of a flatten (``flatten`` with
  the kill switch or drawdown reason, ``stop_breached``);
- a research fill whose order paper's gates refused (``entry_refused`` naming that order), or whose paper order the
  execution path cancelled (``cancel_entries`` or ``flatten``).

The output holds fills only, no equity or return: comparing paper's fills with a replay is the operational check of
the forward protocol (research/protocols/2026-11-14-forward-tests.md, "纸面运行"), not a forward look.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal as D
from decimal import InvalidOperation
from pathlib import Path
from typing import Any

from data.quality.causal_bars import load_bars
from lab.data import HOUR, frame_from_bars, funding_series, load_funding
from lab.simulate import simulate
from lab.strategy import StrategySpec
from trading.adapters.codec import decode
from trading.adapters.simulated.bar_broker import SYMBOL_RULES, broker_config
from trading.application.runtime import PAPER_SCHEMA
from trading.domain.strategy.base import STRATEGIES, OverlayStrategy, Strategy, Window
from trading.domain.strategy.spec import PARAMS

EQUAL, INCONSISTENT, CANNOT = "一致", "不一致", "无法比较"
EXIT_CODES = {EQUAL: 0, INCONSISTENT: 1, CANNOT: 2}
SETTLEMENT = 8 * HOUR
STRATEGY = "rebalance"                      # the reason in a session's own order ids
FLATTENS = {"kill_switch": "flatten", "drawdown_latched": "flatten", "stop_breached": "stop_breached"}
CANCELLED = "canceled by request"           # the research broker's mark on an order someone cancelled
CANCELLING = ("cancel_entries", "flatten")  # the risk exits that cancel resting entries
BAD_DATA = (OSError, KeyError, ValueError, InvalidOperation)


class CannotCompare(Exception):
    """Something the comparison needs is missing or unusable."""


@dataclass(frozen=True)
class Fill:
    at: int
    side: str
    qty: D
    price: D
    fee: D
    order: str            # paper: the order journal's intent id; research: the research order id
    kind: str             # the reason in the order id: "rebalance" (strategy), a flatten reason, "stop" or "other"
    bar: int | None       # the close of the bar the order was decided at, when the order id names it
    k: int | None         # the order's number within that decision

    def key(self) -> tuple[Any, ...]:
        return self.at, self.side, self.qty, self.price, self.fee

    def text(self) -> str:
        return f"{self.side} {self.qty} @ {self.price}，手续费 {self.fee}，订单 {self.order}"


@dataclass
class SymbolResult:
    symbol: str
    first: int                  # the open of the first bar compared
    last: int                   # the open of the last bar compared: the symbol's last decided bar
    matched: int                # equal fills before the first difference
    differs: bool = False
    paper: Fill | None = None   # the first differing fills; None: that side has no more fills
    research: Fill | None = None
    cause: str | None = None    # the intended divergence that explains the difference
    since: int | None = None


@dataclass
class Report:
    verdict: str
    paper_dir: Path
    data_dir: Path
    reason: str = ""
    symbols: list[SymbolResult] = field(default_factory=list)
    divergences: list[dict[str, Any]] = field(default_factory=list)


def when(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


# -- the paper record ---------------------------------------------------------------------------------------------
def _read_record(paper_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, str]]:
    """The saved paper state, the journal's fills and each intent's client order id, from one read transaction."""
    path = paper_dir / "state.sqlite"
    if not path.is_file():
        raise CannotCompare(f"{path} 不存在：没有模拟盘记录")
    try:
        db = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
        try:
            db.execute("BEGIN")
            if db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='paper_state'").fetchone() is None:
                raise CannotCompare(f"{path} 不是模拟盘记录的库（没有模拟盘状态表）")
            row = db.execute("SELECT document FROM paper_state WHERE id = 1").fetchone()
            fills = [json.loads(payload) for (payload,) in db.execute("SELECT payload FROM execution_fills")]
            clients = {str(i): str(c) for i, c in db.execute("SELECT intent_id, client_id FROM submissions")}
        finally:
            db.close()
    except sqlite3.Error as exc:
        raise CannotCompare(f"{path} 读不出（{exc}）") from None
    if row is None:
        raise CannotCompare(f"{path} 里没有保存过的模拟盘状态")
    document = json.loads(row[0], object_hook=decode)
    if document.get("schema") != PAPER_SCHEMA:
        raise CannotCompare(f"模拟盘状态的格式是 {document.get('schema')!r}，本工具读的是 {PAPER_SCHEMA}")
    return document, fills, clients


def _read_events(paper_dir: Path) -> list[dict[str, Any]]:
    path = paper_dir / "target_events.jsonl"
    if not path.is_file():
        raise CannotCompare(f"事件日志 {path} 不存在：记录从哪根 K 线开始、哪些是有意的分叉，都要从事件日志读")
    events = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                events.append(json.loads(line))
            except ValueError:
                raise CannotCompare(f"事件日志 {path} 第 {n} 行读不出") from None
    return events


def _record_start(events: list[dict[str, Any]], symbols: list[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """This record's events and each symbol's first warm-up. A new record warms up without ``resumed_from``, one
    ``warmup`` event per symbol written one after another; a later restart resumes from the saved state."""
    fresh = [i for i, e in enumerate(events) if e.get("kind") == "warmup" and e.get("resumed_from") is None]
    if not fresh:
        raise CannotCompare("事件日志里没有这个模拟盘记录的开头（新记录第一次预热的 warmup 事件）")
    first, members = fresh[-1], set(fresh)
    while first - 1 in members:
        first -= 1
    warm = {str(e.get("symbol")): e for e in events[first:fresh[-1] + 1]}
    if sorted(warm) != symbols:
        raise CannotCompare(f"事件日志里这个记录的第一次预热有 {sorted(warm)}，模拟盘状态有 {symbols}")
    return events[first:], warm


def _intent(intent_id: str) -> tuple[str, int | None, int | None]:
    """(kind, decision bar close, number) from an intent id: ``SYMBOL:close:reason:k`` for the session's and the
    flattener's orders, ``stop-<id>`` for the order a protective stop became."""
    if intent_id.startswith("stop-"):
        return "stop", None, None
    parts = intent_id.split(":")
    if len(parts) == 4 and parts[1].isdigit() and parts[3].isdigit():
        return parts[2], int(parts[1]), int(parts[3])
    return "other", None, None


def _paper_fills(symbol: str, broker: dict[str, Any], journal: list[dict[str, Any]]) -> list[Fill]:
    """The simulated exchange's fills in order, each with the journal's booking of it; the two must agree."""
    booked = {str(row["trade_id"]): row for row in journal if row.get("symbol") == symbol}
    fills = []
    for f in broker["fills"]:
        row = booked.pop(str(f["id"]), None)
        if row is None:
            raise CannotCompare(f"{symbol}：模拟交易所 {when(int(f['at']))} 的成交 {f['id']} 不在订单库里"
                                "（模拟盘记录不完整）")
        venue = (f["side"], D(str(f["qty"])), D(str(f["price"])), D(str(f["fee"])))
        if (row["side"], D(str(row["qty"])), D(str(row["price"])), D(str(row["fee"]))) != venue:
            raise CannotCompare(f"{symbol}：订单库与模拟交易所对成交 {f['id']} 的记录不同")
        kind, bar, k = _intent(str(row["intent_id"]))
        fills.append(Fill(int(f["at"]), *venue, str(row["intent_id"]), kind, bar, k))
    if booked:
        raise CannotCompare(f"{symbol}：订单库里有 {len(booked)} 笔模拟交易所没有的成交")
    return fills


# -- the snapshot -------------------------------------------------------------------------------------------------
def _bars(data_dir: Path, symbol: str, window: int, last: int) -> list[Any]:
    path = data_dir / f"{symbol}_1h.csv"
    try:
        bars = load_bars(path, start_ms=window, extras=True)          # R7 audit F4: the kline fields too
    except BAD_DATA as exc:
        raise CannotCompare(f"K 线快照 {path} 不能用：{exc}") from None
    bars = [b for b in bars if b.open_time <= last]
    if not bars or bars[0].open_time != window or bars[-1].open_time != last:
        raise CannotCompare(f"K 线快照 {path.name} 没有覆盖模拟盘记录的 K 线（{when(window)} 到 {when(last)}）")
    return bars


def _funding(data_dir: Path, symbol: str, window: int, last: int) -> dict[int, tuple[D, D]]:
    path = data_dir / f"{symbol}_funding.csv"
    try:
        funding = load_funding(path)
    except BAD_DATA as exc:
        raise CannotCompare(f"资金费快照 {path} 不能用：{exc}") from None
    first = -(-window // SETTLEMENT) * SETTLEMENT
    missing = [t for t in range(first, last + 1, SETTLEMENT) if t not in funding]
    if missing:
        raise CannotCompare(f"资金费快照 {path.name} 缺 {len(missing)} 次结算，第一次在 {when(missing[0])}")
    return {t: v for t, v in funding.items() if t <= last}


def _same_data(symbol: str, broker: dict[str, Any], bars: list[Any], funding: dict[int, tuple[D, D]],
               first: int) -> None:
    """The snapshot holds what paper saw: the close each order was sent at, each funding settlement and the bar
    open it was charged at, and the last close."""
    closes = {b.available_at: b.close for b in bars}
    opens = {b.open_time: b.open for b in bars}
    for row in broker["orders"]:
        at = int(row["submitted_at"])
        if at not in closes:
            raise CannotCompare(f"{symbol}：模拟盘在 {when(at)} 下的单不在记录的 K 线范围里"
                                "（记录的开头与事件日志不符）")
        if D(str(row["reference"])) != closes[at]:
            raise CannotCompare(f"{symbol}：{when(at - HOUR)} 这根 K 线的收盘价与模拟盘看到的不同"
                                f"（快照 {closes[at]}，模拟盘下单时 {row['reference']}）")
    applied = {}
    for entry in broker["funding_log"]:
        at = int(entry["at"])
        applied[at] = entry
        if at not in opens or at not in funding or D(str(entry["rate"])) != funding[at][0] \
                or D(str(entry["mark"])) != opens[at]:
            raise CannotCompare(f"{symbol}：{when(at)} 的资金费结算与模拟盘看到的不同")
    last = bars[-1].open_time
    unapplied = [t for t in range(-(-first // SETTLEMENT) * SETTLEMENT, last + 1, SETTLEMENT) if t not in applied]
    if unapplied:
        raise CannotCompare(f"{symbol}：模拟盘没有结算 {when(unapplied[0])} 的资金费，快照里有")
    if int(broker["now"]) and D(str(broker["mark"])) != closes.get(int(broker["now"])):
        raise CannotCompare(f"{symbol}：最后一根 K 线的收盘价与模拟盘看到的不同")


# -- research -----------------------------------------------------------------------------------------------------
def _strategy(identity: dict[str, Any]) -> tuple[Strategy, StrategySpec]:
    """The strategy a paper record names (``TargetSpec.identity``) and its execution settings: the overlay's record,
    or a registered strategy's name, parameters and settings (R7d)."""
    if "strategy" not in identity:
        spec = StrategySpec.from_record({k: v for k, v in identity.items() if k not in ("allow_short", "after_stop")})
        return OverlayStrategy(spec), spec
    name = identity["strategy"]
    if name not in STRATEGIES or name not in PARAMS:
        raise CannotCompare(f"不认识模拟盘记录里的策略 {name}")
    try:
        strategy = STRATEGIES[name](PARAMS[name](identity["params"]))
        spec = StrategySpec(band=float(identity["band"]), execution=str(identity["execution"]),
                            ttl_bars=int(identity["ttl_bars"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise CannotCompare(f"模拟盘记录里的策略 {name} 不能重建：{exc}") from None
    return strategy, spec


def _research(symbol: str, spec: StrategySpec, config: dict[str, Any], bars: list[Any],
              funding: dict[int, tuple[D, D]], first: int, last: int, targets: Any) -> list[Fill]:
    rules = SYMBOL_RULES.get(symbol)
    if rules is None:
        raise CannotCompare(f"研究没有 {symbol} 的交易规则")
    balance = str(config.get("starting_balance"))
    if broker_config(spec, rules, starting_balance=balance).record() != config:
        raise CannotCompare(f"{symbol}：模拟交易所的撮合配置与研究的不同")
    result = simulate(symbol, bars, targets, funding, spec, start_ms=first, end_ms=last + HOUR,
                      starting_balance=balance)
    orders = {row["id"]: str(row["internal_id"]) for row in result.order_rows}
    fills = []
    for f in result.fills:
        order = orders[f["order"]]
        _, bar, k = order.split(":")                     # lab.simulate's ids: SYMBOL:close:k
        fills.append(Fill(int(f["at"]), f["side"], D(str(f["qty"])), D(str(f["price"])), D(str(f["fee"])), order,
                          STRATEGY, int(bar), int(k)))
    return fills


# -- comparison ---------------------------------------------------------------------------------------------------
def _explained(symbol: str, paper: Fill | None, research: Fill | None, events: list[dict[str, Any]],
               broker: dict[str, Any], clients: dict[str, str]) -> tuple[str, int] | None:
    """The intended divergence behind a symbol's first differing fill, tied to a fact of the paper record."""
    mine = [e for e in events if e.get("kind") == "paper_divergence" and e.get("symbol") == symbol]
    if paper is not None and paper.kind != STRATEGY and (research is None or paper.at <= research.at):
        if paper.kind == "stop":
            causes = [e for e in mine if e.get("cause") == "stop_triggered"]
        elif paper.kind in FLATTENS:
            cause = FLATTENS[paper.kind]
            causes = [e for e in mine
                      if e.get("cause") == cause and (cause != "flatten" or e.get("reason") == paper.kind)]
        else:
            causes = []
        if causes:
            return str(causes[0]["cause"]), paper.at
    if research is not None and (paper is None or research.at <= paper.at):
        intent = f"{symbol}:{research.bar}:{STRATEGY}:{research.k}"     # paper's order for research's order
        if any(e.get("cause") == "entry_refused" and e.get("order_id") == intent for e in mine):
            return "entry_refused", research.at
        client = clients.get(intent)
        row = next((o for o in broker["orders"] if o.get("internal_id") == client), None) if client else None
        if row is not None and row.get("rejection") == CANCELLED:
            cause = next((str(e["cause"]) for e in mine if e.get("cause") in CANCELLING), None)
            if cause is not None:
                return cause, research.at
    return None


@dataclass
class _Inputs:
    """What paper saw for one symbol: its bars from the warm-up window on, the settlements, the first live bar."""
    broker: dict[str, Any]
    config: dict[str, Any]
    bars: list[Any]
    funding: dict[int, tuple[D, D]]
    first: int
    last: int


def _inputs(symbol: str, document: dict[str, Any], warm: dict[str, Any], data_dir: Path) -> _Inputs:
    saved = document["venue"]["brokers"][symbol]
    broker, config = saved["broker"], saved["config"]
    last_open, count = int(warm["last_open"]), int(warm["bars"])
    window, first = last_open - (count - 1) * HOUR, last_open + HOUR
    last = int(document["symbols"][symbol]["cursor"])
    if int(broker["now"]) > last + HOUR:
        raise CannotCompare(f"{symbol}：模拟交易所已撮合 {when(last + HOUR)} 开始的 K 线，还没有决策；"
                            "等模拟盘再跑一轮后再比较")
    bars = _bars(data_dir, symbol, window, last)
    funding = _funding(data_dir, symbol, window, last)
    _same_data(symbol, broker, bars, funding, first)
    return _Inputs(broker, config, bars, funding, first, last)


def _compare_symbol(symbol: str, inputs: _Inputs, journal: list[dict[str, Any]], clients: dict[str, str],
                    events: list[dict[str, Any]], spec: StrategySpec, targets: Any) -> SymbolResult:
    broker, first, last = inputs.broker, inputs.first, inputs.last
    paper = _paper_fills(symbol, broker, journal)
    research = _research(symbol, spec, inputs.config, inputs.bars, inputs.funding, first, last, targets)
    n = 0
    while n < len(paper) and n < len(research) and paper[n].key() == research[n].key():
        n += 1
    result = SymbolResult(symbol, first, last, n)
    if n == len(paper) == len(research):
        return result
    result.differs = True
    result.paper = paper[n] if n < len(paper) else None
    result.research = research[n] if n < len(research) else None
    explained = _explained(symbol, result.paper, result.research, events, broker, clients)
    if explained is not None:
        result.cause, result.since = explained
    return result


def compare(paper_dir: Path, data_dir: Path) -> Report:
    paper_dir, data_dir = Path(paper_dir), Path(data_dir)
    try:
        document, journal, clients = _read_record(paper_dir)
        events = _read_events(paper_dir)
        symbols = sorted(document["symbols"])
        if symbols != sorted(document["venue"]["brokers"]):
            raise CannotCompare("模拟盘状态里的品种与模拟交易所的品种不同")
        record, warm = _record_start(events, symbols)
        strategy, spec = _strategy(document["spec"])
        inputs = {s: _inputs(s, document, warm[s], data_dir) for s in symbols}
        # R7d: every symbol's targets in one call of the strategy, on what paper saw - as the runtime decides
        windows = {s: Window(frame_from_bars(i.bars, extras=True), funding_series(i.funding))
                   for s, i in inputs.items()}
        targets = strategy.history(windows)
        if not document["spec"].get("allow_short", True):        # the session trades a negative target as 0 (R6)
            targets = {s: t.assign(target=t["target"].clip(lower=0.0)) for s, t in targets.items()}
        results = [_compare_symbol(s, inputs[s], journal, clients, record, spec, targets[s]) for s in symbols]
    except CannotCompare as exc:
        return Report(CANNOT, paper_dir, data_dir, reason=str(exc))
    except Exception as exc:                 # whatever else cannot be read is never "equal" either
        return Report(CANNOT, paper_dir, data_dir, reason=f"比较时出错：{type(exc).__name__}: {exc}")
    verdict = INCONSISTENT if any(r.differs and r.cause is None for r in results) else EQUAL
    return Report(verdict, paper_dir, data_dir, symbols=results,
                  divergences=[e for e in record if e.get("kind") == "paper_divergence"])


def render(report: Report) -> str:
    lines = [f"结论：{report.verdict}", f"模拟盘目录：{report.paper_dir}", f"K 线快照：{report.data_dir}"]
    if report.verdict == CANNOT:
        lines.append(f"原因：{report.reason}")
        return "\n".join(lines) + "\n"
    for r in report.symbols:
        span = f"（{when(r.first)} 到 {when(r.last)} 开始的 K 线）"
        if not r.differs:
            lines.append(f"{r.symbol}：逐笔一致 {r.matched} 笔{span}")
        elif r.cause is not None and r.since is not None:
            lines.append(f"{r.symbol}：前 {r.matched} 笔一致；从 {when(r.since)} 起与研究分叉（{r.cause}），之后不再"
                         f"逐笔比较{span}")
        else:
            at = min(f.at for f in (r.paper, r.research) if f is not None)
            lines.append(f"{r.symbol}：前 {r.matched} 笔一致，第 {r.matched + 1} 笔不同（{when(at)}）{span}")
            lines.append(f"  模拟盘：{r.paper.text() if r.paper else '没有这一笔'}")
            lines.append(f"  研究：{r.research.text() if r.research else '没有这一笔'}")
    groups: dict[tuple[str, str, str], int] = {}
    for e in report.divergences:
        key = (str(e.get("symbol")), str(e.get("cause")), str(e.get("reason") or e.get("issue") or ""))
        groups[key] = groups.get(key, 0) + 1
    lines.append("已解释的分叉（paper_divergence 事件，不算不一致）：" + ("" if groups else "没有"))
    for (symbol, cause, detail), count in groups.items():
        lines.append(f"  - {symbol} {cause}：{count} 次" + (f"（{detail}）" if detail else ""))
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare a paper record with research on the same bars (G6).")
    parser.add_argument("--paper-dir", type=Path, required=True, help="the paper state directory")
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="a binance_public data snapshot that covers the paper record")
    args = parser.parse_args(argv)
    report = compare(args.paper_dir, args.data_dir)
    print(render(report), end="")
    return EXIT_CODES[report.verdict]


if __name__ == "__main__":
    raise SystemExit(main())
