"""One symbol of the target runtime: bars in, target weight, rebalance intents, orders out.

Moved from runner/binance_target.py (migration step 10). Each closed 1h bar: advance the sleeve (paper) or
consume confirmed fills (execution), compute the target with the same function research uses, rebalance,
and send intents through the entry gate to the executor. Every collaborator is injected; the paper sleeve
is the research broker, supplied by the composition root.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Protocol

from trading.application.equity import EquitySource
from trading.application.ports import EventLog
from trading.domain.bars import HOUR, Bar
from trading.domain.market import frame_from_bars, funding_window
from trading.domain.orders import BLOCKED, SETTLED
from trading.domain.position import StrategyPosition
from trading.domain.rebalance import Intent, plan
from trading.domain.strategy.overlay import StrategySpec, latest_target

FUNDING_INTERVAL = 8 * HOUR
MAX_ENTRY_DELAY_MS = 10 * 60_000
# BarBroker attributes that change while trading (config and callbacks are rebuilt from the spec).
BROKER_STATE = ("fees", "funding", "mark", "now", "orders", "fills", "funding_log", "margin", "stop_price",
                "liquidations")


class LotRules(Protocol):
    step: Decimal
    min_qty: Decimal
    min_notional: Decimal


class OrderFactory(Protocol):
    def __call__(self, *, order_id: str, source: str, symbol: str, side: str, qty: Decimal, price: Decimal,
                 reduce_only: bool, reason: str) -> Any: ...


@dataclass
class ExecutionParts:
    """What a session needs to trade for real: the executor, its fill consumer, stops, entry gate and equity."""
    executor: Any
    consumer: Any
    stops: Any
    gate: Any
    equity: EquitySource


class TargetSession:
    def __init__(self, symbol: str, spec: StrategySpec, *, journal: EventLog, rules: LotRules,
                 make_order: OrderFactory, allocation: Decimal = Decimal("0.5"), sleeve: object = None,
                 execution: ExecutionParts | None = None) -> None:
        if (sleeve is None) == (execution is None):
            raise ValueError("a session is either paper (a sleeve) or execution (execution parts)")
        self.symbol, self.spec, self.journal, self.rules = symbol, spec, journal, rules
        self.mode = "paper" if sleeve is not None else "execution"
        self.allocation, self._make_order = allocation, make_order
        self.bars: deque[Bar] = deque(maxlen=spec.window_hours() + 48)
        self.funding: dict[int, float] = {}
        self.cursor: int | None = None
        self.entry_ready, self.entry_reason = False, "no live bar yet"
        self.last_target: tuple[float, bool] = (0.0, False)
        self.sleeve: Any = sleeve
        self.executor: Any = execution.executor if execution else None
        self.consumer: Any = execution.consumer if execution else None
        self.stops: Any = execution.stops if execution else None
        self.gate: Any = execution.gate if execution else None
        self.equity: Any = execution.equity if execution else None
        self._logged = 0
        self.pending: dict[str, bool] = {}   # working order id -> reduce_only

    # -- data ---------------------------------------------------------------
    def funding_series(self, upto: int | None = None) -> Any:
        """Settlements known at ``upto`` (all when None); a catch-up must not see later ones."""
        return funding_window(self.funding, upto, self.spec.funding_window())

    def _target(self) -> tuple[float, bool]:
        return latest_target(frame_from_bars(list(self.bars)), self.funding_series(self.bars[-1].available_at),
                             self.spec)

    # -- paper persistence --------------------------------------------------
    def paper_snapshot(self) -> dict[str, Any]:
        b = self.sleeve.broker
        return {"cursor": self.cursor, "config": b.config.record(), "orders_sent": self.sleeve.orders,
                "position": b.position.to_dict(), "pending": [row["id"] for row in b.pending],
                "broker": {key: getattr(b, key) for key in BROKER_STATE}}

    def restore_paper(self, snap: dict[str, Any]) -> None:
        b = self.sleeve.broker
        if snap["config"] != b.config.record():
            raise RuntimeError(f"{self.symbol}: saved paper state used another broker configuration")
        for key in BROKER_STATE:
            setattr(b, key, snap["broker"][key])
        b.position = StrategyPosition.from_dict(snap["position"])
        by_id = {row["id"]: row for row in b.orders}
        b.pending = [by_id[i] for i in snap["pending"]]   # pending rows are the same objects as order rows
        self.sleeve.orders = snap["orders_sent"]
        self._logged = len(b.fills)

    # -- execution callbacks ------------------------------------------------
    def _on_status(self, order_id: str, status: str) -> None:
        if status == BLOCKED or status in SETTLED:
            self.pending.pop(order_id, None)

    def position_qty(self) -> Decimal:
        if self.sleeve is not None:
            qty: Decimal = self.sleeve.broker.position.qty
            return qty
        owned: Decimal = self.consumer.position.qty
        return owned

    def sleeve_equity(self) -> Decimal:
        if self.sleeve is not None:
            equity: Decimal = self.sleeve.broker.equity
            return equity
        current: Decimal = self.equity.current() * self.allocation
        return current

    def consume_fills(self) -> None:
        for event in self.executor.reconcile_pending():
            if self.consumer.consume(event):
                self.journal.write("fill", symbol=self.symbol, order_id=event.order_id, side=event.side,
                                   qty=str(event.qty), price=str(event.price))

    def submit(self, intents: list[Intent], bar: Bar, reason: str) -> None:
        for k, intent in enumerate(intents):
            order = self._make_order(order_id=f"{self.symbol}:{bar.available_at}:{reason}:{k}",
                                     source=f"target.{self.symbol}", symbol=self.symbol, side=intent.side,
                                     qty=intent.qty, price=bar.close, reduce_only=intent.reduce_only, reason=reason)
            self.journal.write("intent", symbol=self.symbol, order_id=order.order_id, side=intent.side,
                               qty=str(intent.qty), reduce_only=intent.reduce_only, reason=reason)
            if self.sleeve is not None:
                self.sleeve.broker.send_order(order)
                continue
            check = self.gate.check(order)
            if not check.allowed:
                self.journal.write("blocked", symbol=self.symbol, order_id=order.order_id, reason=check.reason)
                continue
            self.pending[order.order_id] = intent.reduce_only
            self.executor.send_order(order)
            self.executor.notify_status()
            self.consume_fills()

    # -- the hourly step ----------------------------------------------------
    def process_bar(self, bar: Bar, *, live: bool, now_ms: int) -> None:
        if self.cursor is not None and bar.open_time != self.cursor + HOUR:
            raise RuntimeError(f"{self.symbol}: non-contiguous bar {bar.open_time} after {self.cursor}")
        self.bars.append(bar)
        self.cursor = bar.open_time
        if not live:
            return
        if self.sleeve is not None:
            rate = self.funding.get(bar.open_time)
            self.sleeve.on_bar(bar, Decimal(str(rate)) if rate is not None else None)
        reasons = []
        if now_ms - bar.available_at > MAX_ENTRY_DELAY_MS:
            reasons.append("bar processed too late")
        latest_settlement = bar.available_at // FUNDING_INTERVAL * FUNDING_INTERVAL
        if latest_settlement not in self.funding:
            reasons.append("funding settlement at or before this close not published yet")
        self.entry_ready, self.entry_reason = not reasons, "; ".join(reasons)
        target, ready = self._target()
        self.last_target = (target, ready)
        if self.sleeve is not None:
            # Paper: exactly the research sleeve step (no readiness gate in research).
            self.sleeve.decide(bar, target, ready)
            for fill in self.sleeve.broker.fills[self._logged:]:
                self.journal.write("paper_fill", symbol=self.symbol, **fill)
            self._logged = len(self.sleeve.broker.fills)
            self.journal.write("paper_equity", symbol=self.symbol, bar_open=bar.open_time,
                               equity=str(self.sleeve.broker.equity), qty=str(self.sleeve.broker.position.qty),
                               target=target)
            return
        if not ready:
            return
        intents = plan(target, equity=self.sleeve_equity(), price=bar.close, current_qty=self.position_qty(),
                       entry_pending=any(not reduce_only for reduce_only in self.pending.values()),
                       band=self.spec.band, step=self.rules.step, min_qty=self.rules.min_qty,
                       min_notional=self.rules.min_notional)
        self.submit(intents, bar, "rebalance")

    def flatten(self, reason: str) -> None:
        qty = self.position_qty()
        if qty == 0 or self.pending or not self.bars:
            return
        self.submit([Intent("sell" if qty > 0 else "buy", abs(qty), True)], self.bars[-1], reason)


__all__ = ["BROKER_STATE", "ExecutionParts", "FUNDING_INTERVAL", "MAX_ENTRY_DELAY_MS", "TargetSession"]
