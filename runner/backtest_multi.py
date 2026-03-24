"""Multi-asset backtest runner.

Extracted from runner/backtest_runner.py to keep file sizes manageable.
"""
from __future__ import annotations

import csv
import json
import logging
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from engine.coordinator import CoordinatorConfig, EngineCoordinator
from engine.decision_bridge import DecisionBridge
from engine.execution_bridge import ExecutionBridge
from event.header import EventHeader
from event.types import EventType, MarketEvent

from runner.backtest.csv_io import iter_ohlcv_csv
from runner.backtest.adapter import BacktestExecutionAdapter
from runner.backtest.metrics import (
    EquityPoint,
    _build_trades_from_fills,
    _build_summary,
    _json_safe,
)

logger = logging.getLogger(__name__)


def _ensure_utc(ts):
    if ts is None:
        return None
    if not isinstance(ts, datetime):
        raise TypeError(f"ts must be datetime, got {type(ts).__name__}")
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def run_multi_backtest(
    *,
    csv_paths: Dict[str, Path],
    starting_balance: Decimal,
    fee_bps: Decimal,
    slippage_bps: Decimal = Decimal("0"),
    out_dir: Optional[Path] = None,
    embargo_bars: int = 1,
    decision_modules: Optional[List[Any]] = None,
) -> Tuple[List[EquityPoint], List[Dict[str, Any]]]:
    """Run a multi-asset backtest across multiple CSV files."""
    symbols = sorted(csv_paths.keys())
    first_symbol = symbols[0]

    tagged_bars: List[Tuple[str, Any]] = []
    for sym, path in csv_paths.items():
        for bar in iter_ohlcv_csv(path):
            tagged_bars.append((sym.upper(), bar))
    tagged_bars.sort(key=lambda x: x[1].ts)

    equity: List[EquityPoint] = []
    fills: List[Dict[str, Any]] = []

    def _on_fill(ev: Any) -> None:
        ts_val = ""
        ev_ts = getattr(ev, "ts", None)
        if isinstance(ev_ts, datetime):
            ts_val = ev_ts.isoformat()
        else:
            ts_ns = getattr(getattr(ev, "header", None), "ts_ns", None)
            ts_val = str(ts_ns) if ts_ns is not None else ""

        fills.append({
            "ts": ts_val,
            "symbol": getattr(ev, "symbol", None),
            "side": getattr(ev, "side", None),
            "qty": getattr(ev, "qty", None),
            "price": getattr(ev, "price", None),
            "fee": getattr(ev, "fee", None),
            "realized_pnl": getattr(ev, "realized_pnl", None),
            "event_id": getattr(getattr(ev, "header", None), "event_id", None),
            "root_event_id": getattr(getattr(ev, "header", None), "root_event_id", None),
        })

    def _on_pipeline(out: Any) -> None:
        m = out.market
        a = out.account
        positions = out.positions
        ts = getattr(m, "last_ts", None)
        if ts is None:
            return
        if isinstance(ts, str):
            raw = ts.strip()
            if raw.endswith("Z"):
                raw = raw[:-1] + "+00:00"
            ts = _ensure_utc(datetime.fromisoformat(raw))

        close_f = getattr(m, "close_f", None)
        if close_f is not None:
            close_d = Decimal(str(close_f))
        else:
            close = getattr(m, "close", None) or getattr(m, "last_price", None)
            if close is None:
                return
            close_d = Decimal(str(close))

        total_unreal = Decimal("0")
        for sym in symbols:
            pos = positions.get(sym.upper())
            if pos is None:
                continue
            qf = getattr(pos, "qty_f", None)
            qty = Decimal(str(qf)) if qf is not None else Decimal(str(getattr(pos, "qty", 0)))
            af = getattr(pos, "avg_price_f", None)
            avg = Decimal(str(af)) if af is not None else getattr(pos, "avg_price", None)
            if qty != 0 and avg is not None:
                total_unreal += (close_d - avg) * qty

        bf = getattr(a, "balance_f", None)
        bal = Decimal(str(bf)) if bf is not None else getattr(a, "balance", Decimal("0"))
        rf = getattr(a, "realized_pnl_f", None)
        realized = Decimal(str(rf)) if rf is not None else getattr(a, "realized_pnl", Decimal("0"))
        eq = bal + total_unreal

        pos = positions.get(first_symbol.upper())
        if pos is None:
            qty = Decimal("0")
            avg = None
        else:
            qf = getattr(pos, "qty_f", None)
            qty = Decimal(str(qf)) if qf is not None else Decimal(str(getattr(pos, "qty", 0)))
            af = getattr(pos, "avg_price_f", None)
            avg = Decimal(str(af)) if af is not None else getattr(pos, "avg_price", None)

        assert isinstance(ts, datetime)
        equity.append(EquityPoint(
            ts=ts, close=close_d, position_qty=qty, avg_price=avg,
            balance=bal, realized=realized, unrealized=total_unreal, equity=eq,
        ))

    coordinator = EngineCoordinator(
        cfg=CoordinatorConfig(
            symbol_default=first_symbol.upper(),
            symbols=tuple(s.upper() for s in symbols),
            currency="USDT",
            starting_balance=float(starting_balance),
            on_pipeline_output=_on_pipeline,
        )
    )

    def _emit(ev: Any) -> None:
        coordinator.emit(ev, actor="backtest")

    _latest_prices: Dict[str, Decimal] = {}

    def _price(sym: str) -> Optional[Decimal]:
        return _latest_prices.get(sym.upper())

    def _ts() -> Optional[datetime]:
        view = coordinator.get_state_view()
        m = view.get("market")
        return getattr(m, "last_ts", None) if m is not None else None

    from execution.sim.embargo import EmbargoExecutionAdapter

    base_adapter = BacktestExecutionAdapter(
        price_source=_price, ts_source=_ts,
        fee_bps=fee_bps, slippage_bps=slippage_bps,
        source="paper", on_fill=_on_fill,
    )

    embargo_adapter = EmbargoExecutionAdapter(inner=base_adapter, embargo_bars=embargo_bars)
    exec_bridge = ExecutionBridge(adapter=embargo_adapter, dispatcher_emit=_emit)

    modules = decision_modules if decision_modules is not None else []
    decision_bridge = DecisionBridge(dispatcher_emit=_emit, modules=modules)

    coordinator.attach_execution_bridge(exec_bridge)
    coordinator.attach_decision_bridge(decision_bridge)
    coordinator.start()

    for i, (sym, bar) in enumerate(tagged_bars):
        if embargo_bars > 0:
            for fill_ev in embargo_adapter.on_bar(i, open_price=bar.o):
                coordinator.emit(fill_ev, actor="backtest")
        embargo_adapter.set_bar(i)

        _latest_prices[sym.upper()] = bar.c

        h = EventHeader.new_root(event_type=EventType.MARKET, version=MarketEvent.VERSION, source="csv")
        ev = MarketEvent(
            header=h, ts=bar.ts, symbol=sym.upper(),
            open=bar.o, high=bar.h, low=bar.l, close=bar.c,
            volume=bar.v if bar.v is not None else Decimal("0"),
        )
        coordinator.emit(ev, actor="replay")

    if embargo_bars > 0 and tagged_bars:
        last_bar = tagged_bars[-1][1]
        for fill_ev in embargo_adapter.on_bar(len(tagged_bars), open_price=last_bar.c):
            coordinator.emit(fill_ev, actor="backtest")

    coordinator.stop()

    if out_dir is not None:
        _write_multi_results(out_dir, equity, fills, csv_paths, symbols)

    return equity, fills


def _write_multi_results(out_dir, equity, fills, csv_paths, symbols):
    """Write multi-backtest results to CSV and JSON files."""
    out_dir.mkdir(parents=True, exist_ok=True)

    eq_path = out_dir / "equity_curve.csv"
    with eq_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ts", "close", "qty", "avg_price", "balance", "realized", "unrealized", "equity"])
        for r in equity:
            w.writerow([
                r.ts.isoformat(), str(r.close), str(r.position_qty),
                str(r.avg_price) if r.avg_price is not None else "",
                str(r.balance), str(r.realized), str(r.unrealized), str(r.equity),
            ])

    fills_path = out_dir / "fills.csv"
    with fills_path.open("w", newline="") as f:
        dw = csv.DictWriter(
            f,
            fieldnames=["ts", "symbol", "side", "qty", "price", "fee", "realized_pnl", "event_id", "root_event_id"],
        )
        dw.writeheader()
        for row in fills:
            dw.writerow(row)

    trades = _build_trades_from_fills(fills)

    trades_path = out_dir / "trades.csv"
    with trades_path.open("w", newline="") as f:
        fieldnames = [
            "trade_id", "symbol", "side", "entry_ts", "exit_ts", "qty",
            "entry_price", "exit_price", "gross_pnl", "fees", "net_pnl",
            "return", "duration_sec",
        ]
        dw2 = csv.DictWriter(f, fieldnames=fieldnames)
        dw2.writeheader()
        for t in trades:
            dw2.writerow(t)

    summary = _build_summary(
        equity=equity, trades=trades,
        csv_path=list(csv_paths.values())[0],
        symbol=",".join(symbols),
    )
    summary_path = out_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(_json_safe(summary), f, ensure_ascii=False, indent=2)
