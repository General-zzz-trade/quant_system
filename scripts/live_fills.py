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
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
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
                               delay_ms=int(fills[0]["time"]) - decided if decided is not None else None))
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
    args = parser.parse_args(argv)
    print(render(compare(args.state_db, args.history, open_price=binance_open)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["ASSUMED", "Report", "Row", "binance_open", "compare", "connect_read_only", "main", "render"]
