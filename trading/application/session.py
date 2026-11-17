"""One symbol of the target runtime: bars in, target weight, rebalance intents, orders out.

Moved from runner/binance_target.py (migration step 10). Each closed 1h bar: advance the window, compute the
target with the same function research uses, rebalance, and send intents through the entry gate to the executor.
Paper and execution take this one path (ADR-0013 point 3, S10): in paper the executor's venue is the simulated
exchange, and the market side has it match the bar and the exit side settle it before ``decide``. Every
collaborator is injected by the composition root.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Protocol

from trading.application.ports import EventLog
from trading.domain.bars import HOUR, Bar
from trading.domain.market import frame_from_bars, funding_window
from trading.domain.orders import BLOCKED, SETTLED
from trading.domain.rebalance import Intent, close_without_equity, plan
from trading.domain.strategy.base import OverlayStrategy, Strategy, Window
from trading.domain.strategy.overlay import StrategySpec
from trading.domain.events import EventKind
from trading.domain.issues import AFTER_STOP_PAUSE
from trading.domain.symbol_state import SymbolState

MAX_ENTRY_DELAY_MS = 10 * 60_000


class LotRules(Protocol):
    @property
    def step(self) -> Decimal: ...

    @property
    def min_qty(self) -> Decimal: ...

    @property
    def min_notional(self) -> Decimal: ...


class SizingEquity(Protocol):
    """The equity a symbol sizes from: zero, never a guess, when it is unknown. In execution the account's
    (``EquitySource``), in paper the symbol's own simulated account (``ModePolicy.sizing_equity``)."""

    def current(self) -> Decimal: ...


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
    equity: SizingEquity
    flattener: Any = None       # trading.application.execution.flatten.Flattener


class TargetSession:
    def __init__(self, symbol: str, spec: StrategySpec, *, journal: EventLog, rules: LotRules,
                 make_order: OrderFactory, execution: ExecutionParts, allocation: Decimal = Decimal("0.5"),
                 strategy: Strategy | None = None, allow_short: bool | None = None,
                 after_stop: str = AFTER_STOP_PAUSE) -> None:
        self.symbol, self.spec, self.journal, self.rules = symbol, spec, journal, rules
        self.allocation, self._make_order = allocation, make_order
        # R1 (runtime extension plan): the target comes from a strategy behind the shared interface; the spec still
        # carries the execution settings (band, execution mode, maker lifetime). Without one, the spec's overlay.
        self.strategy: Strategy = strategy if strategy is not None else OverlayStrategy(spec)
        # R6: whether the signed spec allows shorts; without one, as the spec's own rule says (``long_only``).
        self.allow_short = (not spec.long_only) if allow_short is None else bool(allow_short)
        self.bars: deque[Bar] = deque(maxlen=self.strategy.window_hours() + 48)
        self.funding: dict[int, float] = {}
        # Plan R4: ms between this symbol's funding settlements, from the exchange (the market side sets it); unknown,
        # the symbol takes no entries.
        self.funding_interval: int | None = None
        self.cursor: int | None = None
        self.entry_ready, self.entry_reason = False, "no live bar yet"
        self.state = SymbolState.WARMING      # set by the exit side every cycle (ADR-0014 point 3)
        self.last_target: tuple[float, bool] = (0.0, False)
        # Tech lead 2026-11-17: the spec's rule after a stop, and when a ready decision last wanted exactly 0 (the
        # strategy's reset; in memory only, so after a restart the symbol waits for the next one: fail closed).
        self.after_stop = after_stop
        self.flat_target_at: int | None = None
        self.executor: Any = execution.executor
        self.consumer: Any = execution.consumer
        self.stops: Any = execution.stops
        self.gate: Any = execution.gate
        self.equity: Any = execution.equity
        self.flattener: Any = execution.flattener
        # Paper only, set by the composition root: a read-only view of this symbol's simulated broker, for people
        # and the tests that compare it with research. The session never reads it.
        self.sleeve: Any = None
        self.pending: dict[str, bool] = {}   # working order id -> reduce_only
        self.deferred: Deferred | None = None

    # -- data ---------------------------------------------------------------
    def funding_series(self, upto: int | None = None) -> Any:
        """Settlements known at ``upto`` (all when None); a catch-up must not see later ones."""
        return funding_window(self.funding, upto, self.strategy.funding_window())

    def window(self) -> Window:
        """The closed bars and the funding settlements known at the newest close: what the strategy decides on."""
        return Window(frame_from_bars(list(self.bars), extras=True), self.funding_series(self.bars[-1].available_at))

    def _target(self) -> tuple[float, bool]:
        return self.strategy.latest({self.symbol: self.window()})[self.symbol]

    # -- execution callbacks ------------------------------------------------
    def restore_pending(self) -> None:
        """The working orders of this symbol from the durable journal, so that after a restart an entry that is
        still working counts as pending in the plan, as research sees its resting entry (S10)."""
        for row in self.executor.journal.orders(self.executor.owner, unresolved=True):
            if row["symbol"] == self.symbol:
                self.pending[row["intent_id"]] = bool(row["reducing"])

    def _on_status(self, order_id: str, status: str) -> None:
        if status == BLOCKED or status in SETTLED:
            self.pending.pop(order_id, None)

    def position_qty(self) -> Decimal:
        owned: Decimal = self.consumer.position.qty
        return owned

    def sleeve_equity(self) -> Decimal:
        """The equity this symbol sizes from: its share of the account in execution, its own simulated account in
        paper (``ModePolicy.sizing_equity``; the composition root supplies the source and the share)."""
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
            self._blocked(order.order_id, f"journal: {refusal}", refusal=refusal)
        self.executor.notify_status()
        self.consume_fills()

    def submit(self, intents: list[Intent], bar: Bar, reason: str) -> None:
        for k, intent in enumerate(intents):
            order = self._order(k, intent, bar, reason)
            self.journal.write(EventKind.INTENT, symbol=self.symbol, order_id=order.order_id, side=intent.side,
                               qty=str(intent.qty), reduce_only=intent.reduce_only, reason=reason)
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
        """Advance the window by ``bar`` and, for a live bar, decide on it."""
        self.advance(bar)
        if live:
            self.decide(bar, now_ms=now_ms)

    def advance(self, bar: Bar) -> None:
        if self.cursor is not None and bar.open_time != self.cursor + HOUR:
            raise RuntimeError(f"{self.symbol}: non-contiguous bar {bar.open_time} after {self.cursor}")
        self.bars.append(bar)
        self.cursor = bar.open_time

    def funding_rate(self, bar: Bar) -> Decimal | None:
        """The funding settlement at ``bar``'s open, as the venue charges it (paper: the simulated exchange)."""
        rate = self.funding.get(bar.open_time)
        return Decimal(str(rate)) if rate is not None else None

    def decide(self, bar: Bar, *, now_ms: int, target: tuple[float, bool] | None = None) -> None:
        """Rebalance toward the target at ``bar``'s close; ``now_ms`` is the entry gates' clock. ``target`` is the
        (weight, ready) a cross-sectional strategy gave for every symbol at once (R3); without it the session asks
        its own strategy."""
        reasons = []
        if now_ms - bar.available_at > MAX_ENTRY_DELAY_MS:
            reasons.append("bar processed too late")
        interval = self.funding_interval
        if interval is None:
            reasons.append("funding interval not known")
        elif bar.available_at // interval * interval not in self.funding:
            reasons.append("funding settlement at or before this close not published yet")
        self.entry_ready, self.entry_reason = not reasons, "; ".join(reasons)
        weight, ready = target if target is not None else self._target()
        if weight < 0 and not self.allow_short:
            weight = 0.0             # R6: no short unless the spec allows it; one already held is closed
        self.last_target = (weight, ready)
        if ready and weight == 0.0:
            self.flat_target_at = bar.available_at
        self.drop_deferred("superseded by a newer bar")
        if not ready:
            return
        equity = self.sleeve_equity()
        if equity <= 0:      # unknown: only a whole-position close can be planned (R13, decided 2026-11-15)
            self.submit(close_without_equity(weight, self.position_qty()), bar, "rebalance")
            return
        intents = plan(weight, equity=equity, price=bar.close, current_qty=self.position_qty(),
                       entry_pending=any(not reduce_only for reduce_only in self.pending.values()),
                       band=self.spec.band, step=self.rules.step, min_qty=self.rules.min_qty,
                       min_notional=self.rules.min_notional)
        self.submit(intents, bar, "rebalance")

    def cancel_entries(self) -> None:
        """Cancel resting entries without closing anything (a kill switch without flatten, a paused symbol)."""
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


__all__ = ["Deferred", "ExecutionParts", "MAX_ENTRY_DELAY_MS", "TargetSession"]
