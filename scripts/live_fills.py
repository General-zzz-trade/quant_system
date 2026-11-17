"""The pilot's real fills against the backtest's assumptions (tech lead 2026-11-16: the pilot exists to calibrate the
backtest against reality).

    python3 -m scripts.live_fills --state-db /var/lib/quant/state/mainnet/state.sqlite --history <dir>

``--history`` is the read-only export of the account's own records (``python3 -m scripts.account_check --history
<dir>``). The state database is opened read-only. For every order the runtime sent and the exchange filled:

- the decision: the intent id is ``SYMBOL:<decision bar close, ms>:<reason>:<k>``; a triggered exchange stop is
  ``stop-<client algo id>``, priced against its trigger in ``protective_stops``;
- the fill: the account's trades of the order (average price, fees, the first trade's time);
- the reference the backtest uses: the open of the bar after the decision (market orders, assumed 3 bp worse) or the
  stop's trigger (assumed 10 bp worse); the taker fee is assumed 5 bp.

Slippage is in basis points against the order: positive means worse than the reference. An order with no trade in
the export is listed as missing and never guessed. The output holds fills and costs only, no returns.

The daily calibration report (task 3, 2026-11-17) adds, per UTC day, with ``--json-out`` / ``--markdown-out``:

- entries and stops: slippage against the reference above (by the day of the order's first trade);
- order outcomes: every order the runtime reserved that day (by its reservation time) by final status - the rejection
  rate is ``rejected / orders``; ``not_found`` is an unknown outcome that ended absent after the grace. The venue's
  error codes are not recorded in the state database or the event log, so no code breakdown is possible here;
- refusals: the runtime's own ``blocked`` events by reason (``--events``, the execution event log);
- fees in basis points and USDT against the assumed 5 bp taker fee;
- funding: the account's funding income on the strategy's symbols (the export's ``income.json``) against what the
  backtest's broker charges for the same position, ``-position x mark x rate`` at each settlement (Binance's public
  funding history), the position being the sum of the strategy's own trades before the settlement.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal as D
from pathlib import Path
from typing import Any

ASSUMED = {"market": 3.0, "stop": 10.0, "fee": 5.0}      # trading/adapters/simulated/bar_broker.py broker_config
H = 3_600_000


@dataclass(frozen=True)
class Row:
    order_id: str
    symbol: str
    kind: str                 # rebalance, stop, or a flatten's reason (kill_switch, drawdown_latched, ...)
    side: str
    decided_at: int | None    # the decision bar's close (ms); None for a stop
    qty: D
    avg_price: D
    reference: D
    slip_bp: float
    fee_bp: float
    delay_ms: int | None      # first trade time - decision bar close
    filled_at: int | None = None   # the first trade's time (ms)


@dataclass
class Report:
    rows: list[Row] = field(default_factory=list)
    missing: list[dict[str, Any]] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for kind in sorted({r.kind for r in self.rows}):
            part = [r for r in self.rows if r.kind == kind]
            slips = [r.slip_bp for r in part]
            delays = [r.delay_ms for r in part if r.delay_ms is not None]
            out[kind] = {"orders": len(part), "assumed_slip_bp": ASSUMED["stop" if kind == "stop" else "market"],
                         "slip_bp_mean": statistics.fmean(slips), "slip_bp_median": statistics.median(slips),
                         "slip_bp_max": max(slips),
                         "delay_ms_median": statistics.median(delays) if delays else None}
        fees = [r.fee_bp for r in self.rows]
        out["fee_bp_median"] = statistics.median(fees) if fees else None
        out["assumed_fee_bp"] = ASSUMED["fee"]
        out["missing_orders"] = len(self.missing)
        return out


def connect_read_only(path: Path) -> sqlite3.Connection:
    db = sqlite3.connect(f"file:{Path(path)}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    return db


def _trades(history: Path) -> dict[str, list[dict[str, Any]]]:
    by_order: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(Path(history).glob("trades_*.json")):
        for trade in json.loads(path.read_text()):
            by_order.setdefault(str(trade["orderId"]), []).append(trade)
    return by_order


def _decision(intent_id: str) -> tuple[str, int | None]:
    """(kind, decision bar close) from an intent id."""
    if intent_id.startswith("stop-"):
        return "stop", None
    parts = intent_id.split(":")
    if len(parts) == 4 and parts[1].isdigit():
        return parts[2], int(parts[1])
    return "other", None


def compare(state_db: Path, history: Path, *, open_price: Callable[[str, int], D]) -> Report:
    trades = _trades(history)
    report = Report()
    with connect_read_only(state_db) as db:
        orders = db.execute("SELECT intent_id, symbol, order_id, side FROM submissions "
                            "WHERE order_id IS NOT NULL AND order_id != '' ORDER BY created_at").fetchall()
        triggers = {f"stop-{r['client_algo_id']}": D(r["trigger_price"])
                    for r in db.execute("SELECT client_algo_id, trigger_price FROM protective_stops")}
    for order in orders:
        intent, order_id, side = str(order["intent_id"]), str(order["order_id"]), str(order["side"]).lower()
        if intent.endswith(":foreign"):
            continue
        fills = sorted(trades.get(order_id, []), key=lambda t: int(t["time"]))
        if not fills:
            report.missing.append({"order_id": order_id, "intent_id": intent, "symbol": order["symbol"]})
            continue
        kind, decided = _decision(intent)
        qty = sum((D(str(t["qty"])) for t in fills), D(0))
        quote = sum((D(str(t["price"])) * D(str(t["qty"])) for t in fills), D(0))
        fee = sum((D(str(t["commission"])) for t in fills if t.get("commissionAsset", "USDT") == "USDT"), D(0))
        avg = quote / qty
        if kind == "stop":
            reference = triggers.get(intent)
            if reference is None:
                report.missing.append({"order_id": order_id, "intent_id": intent, "symbol": order["symbol"],
                                       "why": "no trigger price recorded"})
                continue
        elif decided is not None:
            reference = D(open_price(str(order["symbol"]), decided))
        else:
            report.missing.append({"order_id": order_id, "intent_id": intent, "symbol": order["symbol"],
                                   "why": "the intent id names no decision bar"})
            continue
        sign = 1 if side == "buy" else -1
        report.rows.append(Row(order_id=order_id, symbol=str(order["symbol"]), kind=kind, side=side,
                               decided_at=decided, qty=qty, avg_price=avg, reference=reference,
                               slip_bp=float(sign * (avg / reference - 1) * 10_000), fee_bp=float(fee / quote * 10_000),
                               delay_ms=int(fills[0]["time"]) - decided if decided is not None else None,
                               filled_at=int(fills[0]["time"])))
    return report


def binance_open(symbol: str, open_time_ms: int) -> D:
    """The 1h bar's open from Binance's public klines (no account, no key)."""
    url = (f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval=1h&startTime={open_time_ms}"
           f"&endTime={open_time_ms + H - 1}&limit=1")
    with urllib.request.urlopen(url, timeout=15) as response:
        rows = json.loads(response.read())
    if not rows or int(rows[0][0]) != open_time_ms:
        raise RuntimeError(f"{symbol}: no 1h bar opening at {open_time_ms}")
    return D(str(rows[0][1]))


DAY = 86_400_000
FundingRates = Callable[[str, int, int], list[tuple[int, D, D]]]    # symbol, from, to -> [(time ms, rate, mark)]


def day_of(ms: float) -> str:
    """The UTC date of a time in milliseconds."""
    return datetime.fromtimestamp(ms / 1000, timezone.utc).strftime("%Y-%m-%d")


def _strategy_orders(db: sqlite3.Connection) -> list[sqlite3.Row]:
    return [r for r in db.execute("SELECT intent_id, symbol, order_id, side, status, settled, created_at "
                                  "FROM submissions ORDER BY created_at").fetchall()
            if not str(r["intent_id"]).endswith(":foreign")]


def outcomes(state_db: Path) -> dict[str, dict[str, Any]]:
    """Per day of reservation: every order the runtime reserved, by final status, and the rejection rate."""
    days: dict[str, dict[str, Any]] = {}
    with connect_read_only(state_db) as db:
        rows = _strategy_orders(db)
    for row in rows:
        if row["created_at"] is None:
            continue
        day = days.setdefault(day_of(float(row["created_at"]) * 1000), {"orders": 0, "statuses": {}, "unsettled": 0})
        day["orders"] += 1
        status = str(row["status"])
        day["statuses"][status] = day["statuses"].get(status, 0) + 1
        day["unsettled"] += 0 if row["settled"] else 1
    for day in days.values():
        day["rejection_rate"] = day["statuses"].get("rejected", 0) / day["orders"]
    return days


def refusals(events: Path | None) -> dict[str, dict[str, int]]:
    """Per day: the runtime's own ``blocked`` events (entries it refused) by reason."""
    days: dict[str, dict[str, int]] = {}
    if events is None:
        return days
    for line in Path(events).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("kind") != "blocked":
            continue
        reason = str(row.get("reason") or row.get("refusal") or "unknown")
        day = days.setdefault(day_of(float(row["ts"]) * 1000), {})
        day[reason] = day.get(reason, 0) + 1
    return days


def funding(state_db: Path, history: Path, *, rates: FundingRates) -> dict[str, dict[str, float]]:
    """Per day: the account's funding income on the strategy's symbols against the backtest broker's charge for the
    strategy's own position, ``-position x mark x rate`` at each settlement."""
    with connect_read_only(state_db) as db:
        orders = {str(r["order_id"]): str(r["symbol"]) for r in _strategy_orders(db) if r["order_id"]}
    symbols = set(orders.values())
    trades = [t for order_id, rows in _trades(history).items() if order_id in orders for t in rows]
    income_path = Path(history) / "income.json"
    income = json.loads(income_path.read_text()) if income_path.exists() else []
    days: dict[str, dict[str, float]] = {}

    def add(day: str, key: str, amount: D) -> None:
        entry = days.setdefault(day, {"actual": 0.0, "model": 0.0})
        entry[key] += float(amount)

    for row in income:
        if (row.get("incomeType") == "FUNDING_FEE" and row.get("asset", "USDT") == "USDT"
                and row.get("symbol") in symbols):
            add(day_of(int(row["time"])), "actual", D(str(row["income"])))
    summary_path = Path(history) / "summary.json"
    end = json.loads(summary_path.read_text()).get("to_ms") if summary_path.exists() else None
    for symbol in sorted(symbols):
        moves = sorted((int(t["time"]), D(str(t["qty"])) * (1 if str(t["side"]).upper() == "BUY" else -1))
                       for t in trades if orders[str(t["orderId"])] == symbol)
        if not moves:
            continue
        last = int(end) if end is not None else max(int(r["time"]) for r in income) if income else moves[-1][0]
        for at, rate, mark in rates(symbol, moves[0][0], last):
            position = sum((qty for when, qty in moves if when < at), D(0))
            if position:
                add(day_of(at), "model", -position * mark * rate)
    for entry in days.values():
        entry["difference"] = entry["actual"] - entry["model"]
    return days


def daily(report: Report, *, outcome: dict[str, dict[str, Any]] | None = None,
          refused: dict[str, dict[str, int]] | None = None,
          paid: dict[str, dict[str, float]] | None = None) -> dict[str, dict[str, Any]]:
    """The calibration report by UTC day: fills, fees, order outcomes, refusals and funding."""
    days: dict[str, dict[str, Any]] = {}
    for row in report.rows:
        if row.filled_at is None:
            continue
        days.setdefault(day_of(row.filled_at), {}).setdefault("_rows", []).append(row)
    for day in days.values():
        rows = day.pop("_rows")
        for name, part, assumed in (("entries", [r for r in rows if r.kind != "stop"], ASSUMED["market"]),
                                    ("stops", [r for r in rows if r.kind == "stop"], ASSUMED["stop"])):
            if part:
                slips = [r.slip_bp for r in part]
                day[name] = {"orders": len(part), "assumed_slip_bp": assumed,
                             "slip_bp_median": statistics.median(slips), "slip_bp_mean": statistics.fmean(slips),
                             "slip_bp_max": max(slips)}
        day["fees"] = {"fee_bp_median": statistics.median([r.fee_bp for r in rows]), "assumed_fee_bp": ASSUMED["fee"],
                       "fee_usdt": float(sum((D(str(r.fee_bp)) / 10_000 * r.qty * r.avg_price for r in rows), D(0)))}
    for key, source in (("outcomes", outcome), ("refusals", refused), ("funding", paid)):
        for date, value in (source or {}).items():
            days.setdefault(date, {})[key] = value
    return dict(sorted(days.items()))


def render_markdown(days: dict[str, dict[str, Any]], missing: int) -> str:
    """A one-page summary of the daily report (numbers only from ``days``)."""
    lines = ["# 校准试运行：实盘对回测的每日对照", "",
             "基点为单位；滑点为正表示比回测参考价差。回测假设：市价单 3 bp、止损 10 bp、吃单费 5 bp；资金费按"
             "\"仓位 × 标记价 × 费率\"。交易所的错误码没有记录，拒单只按最终状态统计。", "",
             "| 日期 (UTC) | 入场笔数 | 入场滑点中位数 | 止损笔数 | 止损滑点中位数 | 手续费中位数 bp | 订单数 "
             "| 拒单率 | 宽限后查无 | 自己拒绝的入场 | 资金费 实际 / 模型 (USDT) |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for date, d in days.items():
        e, s, f, o, r, p = (d.get(k, {}) for k in ("entries", "stops", "fees", "outcomes", "refusals", "funding"))

        def num(x: Any, fmt: str = ".2f") -> str:
            return "-" if x is None else format(x, fmt)
        lines.append(f"| {date} | {e.get('orders', 0)} | {num(e.get('slip_bp_median'))} | {s.get('orders', 0)} | "
                     f"{num(s.get('slip_bp_median'))} | {num(f.get('fee_bp_median'))} | {o.get('orders', 0)} | "
                     f"{num(o.get('rejection_rate'), '.0%')} | {o.get('statuses', {}).get('not_found', 0)} | "
                     f"{sum(r.values()) if r else 0} | "
                     f"{num(p.get('actual'), '.4f')} / {num(p.get('model'), '.4f')} |")
    reasons: dict[str, int] = {}
    for d in days.values():
        for reason, count in d.get("refusals", {}).items():
            reasons[reason] = reasons.get(reason, 0) + count
    if reasons:
        lines += ["", "自己拒绝的入场，按原因：" + "；".join(f"{k} {v}" for k, v in sorted(reasons.items()))]
    lines += ["", f"导出里找不到成交的订单：{missing} 笔（不猜）。"]
    return "\n".join(lines) + "\n"


def binance_funding_rates(symbol: str, start_ms: int, end_ms: int) -> list[tuple[int, D, D]]:
    """Settled funding (time, rate, mark price) from Binance's public funding history (no account, no key)."""
    out: list[tuple[int, D, D]] = []
    cursor = start_ms
    while cursor <= end_ms:
        url = (f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={symbol}&startTime={cursor}&endTime={end_ms}"
               f"&limit=1000")
        with urllib.request.urlopen(url, timeout=15) as response:
            rows = json.loads(response.read())
        out += [(int(r["fundingTime"]), D(str(r["fundingRate"])), D(str(r["markPrice"]))) for r in rows]
        if len(rows) < 1000:
            break
        cursor = int(rows[-1]["fundingTime"]) + 1
    return out


def render(report: Report) -> str:
    summary = report.summary()
    lines = ["Pilot fills against the backtest's assumptions (basis points, positive = worse than the reference)"]
    for kind, s in summary.items():
        if isinstance(s, dict):
            lines.append(f"  {kind}: {s['orders']} orders; slippage median {s['slip_bp_median']:.2f}, mean "
                         f"{s['slip_bp_mean']:.2f}, max {s['slip_bp_max']:.2f} (assumed {s['assumed_slip_bp']:g})"
                         + (f"; decision to fill median {s['delay_ms_median'] / 1000:.1f} s"
                            if s["delay_ms_median"] is not None else ""))
    if summary["fee_bp_median"] is not None:
        lines.append(f"  fees: median {summary['fee_bp_median']:.2f} (assumed {summary['assumed_fee_bp']:g})")
    lines.append(f"  missing: {summary['missing_orders']} orders with no trade in the export")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--state-db", type=Path, required=True)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--events", type=Path, default=None, help="the execution event log (JSONL), for refusals")
    parser.add_argument("--json-out", type=Path, default=None, help="write the daily report as JSON")
    parser.add_argument("--markdown-out", type=Path, default=None, help="write the daily report's one-page summary")
    args = parser.parse_args(argv)
    report = compare(args.state_db, args.history, open_price=binance_open)
    print(render(report))
    if args.json_out or args.markdown_out:
        days = daily(report, outcome=outcomes(args.state_db), refused=refusals(args.events),
                     paid=funding(args.state_db, args.history, rates=binance_funding_rates))
        if args.json_out:
            args.json_out.write_text(json.dumps({"assumed": ASSUMED, "summary": report.summary(), "days": days,
                                                 "missing": report.missing}, indent=2, default=str) + "\n")
        if args.markdown_out:
            args.markdown_out.write_text(render_markdown(days, len(report.missing)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["ASSUMED", "Report", "Row", "binance_funding_rates", "binance_open", "compare", "connect_read_only",
           "daily", "day_of", "funding", "main", "outcomes", "refusals", "render", "render_markdown"]
