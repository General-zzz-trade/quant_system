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
from trading.domain.rebalance import Intent, close_without_equity, plan
from trading.domain.strategy.overlay import StrategySpec, latest_target
from trading.domain.events import EventKind

FUNDING_INTERVAL = 8 * HOUR
MAX_ENTRY_DELAY_MS = 10 * 60_000
# BarBroker attributes that change while trading (config and callbacks are rebuilt from the spec).
BROKER_STATE = ("fees", "funding", "mark", "now", "orders", "fills", "funding_log", "margin", "stop_price",
                "liquidations")


class LotRules(Protocol):
    @property
    def step(self) -> Decimal: ...

    @property
    def min_qty(self) -> Decimal: ...

    @property
    def min_notional(self) -> Decimal: ...


class RoutedOrder(Protocol):
    """What the session reads of the order it built: its durable intent id."""

    @property
    def order_id(self) -> str: ...


class OrderFactory(Protocol):
    def __call__(self, *, order_id: str, source: str, symbol: str, side: str, qty: Decimal, price: Decimal,
                 reduce_only: bool, reason: str) -> Any: ...


@dataclass
class Deferred:
    """Entries of one bar that only the orders-per-second limit refused; asked again on later cycles."""
    bar: Bar
    reason: str
    items: list[tuple[int, Intent]]


@dataclass
class ExecutionParts:
    """What a session needs to trade for real: the executor, its fill consumer, stops, entry gate and equity."""
    executor: Any
    consumer: Any
    stops: Any
    gate: Any
    equity: EquitySource
    flattener: Any = None       # trading.application.execution.flatten.Flattener


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
        self.flattener: Any = execution.flattener if execution else None
        self._logged = 0
        self.pending: dict[str, bool] = {}   # working order id -> reduce_only
        self.deferred: Deferred | None = None

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
                self.journal.write(EventKind.FILL, symbol=self.symbol, order_id=event.order_id, side=event.side,
                                   qty=str(event.qty), price=str(event.price))

    def _order(self, k: int, intent: Intent, bar: Bar, reason: str) -> Any:
        return self._make_order(order_id=f"{self.symbol}:{bar.available_at}:{reason}:{k}",
                                source=f"target.{self.symbol}", symbol=self.symbol, side=intent.side,
                                qty=intent.qty, price=bar.close, reduce_only=intent.reduce_only, reason=reason)

    def _blocked(self, order_id: str, reason: str, **extra: Any) -> None:
        self.journal.write(EventKind.BLOCKED, symbol=self.symbol, order_id=order_id, reason=reason, **extra)

    def _send(self, order: RoutedOrder, intent: Intent) -> None:
        self.pending[order.order_id] = intent.reduce_only
        self.executor.send_order(order)
        refusal = self.executor.take_refusal(order.order_id)
        if refusal is not None:                  # ADR-0005 7b: say why the order journal refused it
            self._blocked(order.order_id, f"journal: {refusal}")
        self.executor.notify_status()
        self.consume_fills()

    def submit(self, intents: list[Intent], bar: Bar, reason: str) -> None:
        for k, intent in enumerate(intents):
            order = self._order(k, intent, bar, reason)
            self.journal.write(EventKind.INTENT, symbol=self.symbol, order_id=order.order_id, side=intent.side,
                               qty=str(intent.qty), reduce_only=intent.reduce_only, reason=reason)
            if self.sleeve is not None:
                self.sleeve.broker.send_order(order)
                continue
            check = self.gate.check(order)
            if check.allowed:
                self._send(order, intent)
            elif check.retry and not intent.reduce_only:
                # Only the orders-per-second limit refused it (the other symbol has just traded): keep the
                # entry and ask again on a later cycle of the same bar (decision 2026-11-14, defect R23).
                if self.deferred is None or self.deferred.bar is not bar or self.deferred.reason != reason:
                    self.deferred = Deferred(bar, reason, [])
                self.deferred.items.append((k, intent))
                self._blocked(order.order_id, check.reason, retry=True)
            else:
                self._blocked(order.order_id, check.reason)

    def retry_deferred(self, now_ms: int) -> None:
        """Ask again for the entries the rate limit refused; drop them once the bar is too old to enter."""
        deferred = self.deferred
        if deferred is None:
            return
        self.deferred = None
        late = now_ms - deferred.bar.available_at > MAX_ENTRY_DELAY_MS
        waiting: list[tuple[int, Intent]] = []
        for k, intent in deferred.items:
            order = self._order(k, intent, deferred.bar, deferred.reason)
            if late:
                self._blocked(order.order_id, "inputs: bar processed too late")
                continue
            check = self.gate.check(order)
            if check.allowed:
                self._send(order, intent)
            elif check.retry:
                waiting.append((k, intent))
            else:
                self._blocked(order.order_id, check.reason)
        if waiting:
            self.deferred = Deferred(deferred.bar, deferred.reason, waiting)

    def drop_deferred(self, why: str) -> None:
        if self.deferred is not None:
            for k, intent in self.deferred.items:
                self._blocked(self._order(k, intent, self.deferred.bar, self.deferred.reason).order_id, why)
            self.deferred = None

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
                self.journal.write(EventKind.PAPER_FILL, symbol=self.symbol, **fill)
            self._logged = len(self.sleeve.broker.fills)
            self.journal.write(EventKind.PAPER_EQUITY, symbol=self.symbol, bar_open=bar.open_time,
                               equity=str(self.sleeve.broker.equity), qty=str(self.sleeve.broker.position.qty),
                               target=target)
            return
        self.drop_deferred("superseded by a newer bar")
        if not ready:
            return
        equity = self.sleeve_equity()
        if equity <= 0:      # unknown: only a whole-position close can be planned (R13, decided 2026-11-15)
            self.submit(close_without_equity(target, self.position_qty()), bar, "rebalance")
            return
        intents = plan(target, equity=equity, price=bar.close, current_qty=self.position_qty(),
                       entry_pending=any(not reduce_only for reduce_only in self.pending.values()),
                       band=self.spec.band, step=self.rules.step, min_qty=self.rules.min_qty,
                       min_notional=self.rules.min_notional)
        self.submit(intents, bar, "rebalance")

    def cancel_entries(self) -> None:
        """Cancel resting entries without closing anything (a kill switch without flatten, a paused symbol)."""
        if self.executor is None:
            return
        self.drop_deferred("entries blocked: deferred entry dropped")
        self.executor.cancel_entries(self.symbol)
        self.consume_fills()

    def flatten(self, reason: str, now_ms: int | None = None) -> None:
        """Exit for a kill switch, a latched drawdown or a breached stop; called every cycle while it holds.
        Resting entries are canceled first and failed attempts are retried (step 11, ``Flattener``). Without
        bars (a warm-up that market data did not allow, step 14) the hour of ``now_ms`` names the attempts."""
        self.drop_deferred(f"{reason}: deferred entry dropped")
        if self.flattener is None:
            return
        if self.bars:
            bar = self.bars[-1]
        elif now_ms is not None:
            hour = now_ms // HOUR * HOUR
            bar = Bar(hour - HOUR, Decimal(0), Decimal(0), Decimal(0), Decimal(0), Decimal(0))
        else:
            return
        self.flattener.flatten(reason, bar=bar, ledger=self.position_qty, consume=self.consume_fills)


__all__ = ["BROKER_STATE", "Deferred", "ExecutionParts", "FUNDING_INTERVAL", "MAX_ENTRY_DELAY_MS", "TargetSession"]
