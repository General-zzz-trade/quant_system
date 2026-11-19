"""Reliable flattening: the exit of a kill switch, a latched drawdown or a breached stop (migration step 11).

ADR-0005, approved 2026-11-14:

- 2b: every call first cancels the symbol's resting entries and consumes the fills that brings. A
  cancellation whose outcome is unknown does not stop the flatten; the next call cancels again.
- 3b: a kill switch or a latched drawdown closes the symbol's whole venue position; since ADR-0012 point 6 a
  breached stop does too. The ledger is never corrected here: a fill beyond it is booked as a foreign fill
  (ADR-0012 point 5), and if the ledger otherwise differs from the venue, reconciliation keeps reporting it and
  entries stay blocked until a person resolves it.
- 1b: every attempt is a new intent with its own id; nothing is ever resent. The next attempt goes out only
  when no reduction of the symbol is unresolved and RETRY_SECONDS have passed since the last one. After
  MAX_FAILURES attempts in a row without a fill the automatic retries stop, an alert is written, and one
  attempt is made per new bar.
- 7b: an attempt the order journal refuses is logged with the reason, and counts as a failure.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from decimal import Decimal
from typing import Protocol

from trading.application.execution.stops import PositionView
from trading.application.ports import Clock, EventLog, ExchangeAccount, OrderJournal
from trading.domain.bars import Bar
from trading.domain.orders import OrderIntent
from trading.domain.events import EventKind
from trading.domain.issues import IssueKind

logger = logging.getLogger(__name__)

# These close the whole venue position: the kill switch and a latched drawdown (ADR-0005 3b), and since ADR-0012
# point 6 a breached stop. A fill beyond the ledger is booked as a foreign fill (TargetFillConsumer).
EMERGENCY = frozenset({"kill_switch", "drawdown_latched", "stop_breached"})
RETRY_SECONDS = 30.0
MAX_FAILURES = 10


class FlattenExecutor(Protocol):
    @property
    def journal(self) -> OrderJournal: ...

    @property
    def owner(self) -> str: ...

    def send_order(self, intent: OrderIntent) -> tuple[object, ...]: ...

    def cancel_entries(self, symbol: str) -> tuple[object, ...]: ...

    def take_refusal(self, intent_id: str) -> str | None: ...


class FlattenAccount(ExchangeAccount, Protocol):
    """The account, with one symbol's venue position (flat included; None when the answer has no row for it)."""

    def get_position(self, symbol: str) -> PositionView | None: ...


class Flattener:
    def __init__(self, symbol: str, executor: FlattenExecutor, account: FlattenAccount, events: EventLog,
                 clock: Clock) -> None:
        self.symbol, self.executor, self.account, self.events, self.clock = symbol, executor, account, events, clock
        self.failures = 0                     # attempts in a row that ended without a fill
        self._last: str | None = None         # the last attempt, until its outcome has been counted
        self._sent_at: float | None = None
        self._waiting_bar: int | None = None  # after MAX_FAILURES: the bar whose attempt has been used

    def flatten(self, reason: str, *, bar: Bar, ledger: Callable[[], Decimal], consume: Callable[[], None],
                position: PositionView | None = None, may_hold: bool = True) -> None:
        """``position``: the symbol's row in the exit side's account snapshot, if it has one; ``may_hold``: without a
        row, whether the symbol could hold something (plan D2, ``_venue_qty``)."""
        try:
            self.executor.cancel_entries(self.symbol)
            consume()
        except Exception as exc:              # the flatten still goes ahead; the journal refuses it if it must
            self._issue(f"flatten: entry cancellation or fill consumption failed ({type(exc).__name__}: {exc})")
        if self._reduction_unresolved():
            return                            # never a second reduction while one is unresolved
        self._count_last(bar)
        qty = self._venue_qty(ledger, position, may_hold) if reason in EMERGENCY else ledger()
        if qty is None:
            return
        if qty == 0:
            self.failures, self._sent_at, self._waiting_bar = 0, None, None
            return
        now = self.clock()
        if self._sent_at is not None and now - self._sent_at < RETRY_SECONDS:
            return
        if self.failures >= MAX_FAILURES and self._waiting_bar == bar.open_time:
            return
        self._attempt(reason, qty, bar, now, consume)

    # ------------------------------------------------------------------------------------------------------
    def _rows(self) -> list[dict[str, object]]:
        return [r for r in self.executor.journal.orders(self.executor.owner) if r["symbol"] == self.symbol]

    def _reduction_unresolved(self) -> bool:
        return any(r["reducing"] and not r["settled"] for r in self._rows())

    def _count_last(self, bar: Bar) -> None:
        if self._last is None:
            return
        filled = bool(self.executor.journal.intent_fills(self._last))
        self._last = None
        if filled:
            self.failures, self._waiting_bar = 0, None
            return
        self.failures += 1
        if self.failures >= MAX_FAILURES:
            self._waiting_bar = bar.open_time
            reason = f"flatten failed {self.failures} times in a row; automatic retries stopped, once per bar now"
            self.events.write(EventKind.ALERT, symbol=self.symbol, reason=reason, failures=self.failures)
            logger.error("%s: %s", self.symbol, reason)

    def _venue_qty(self, ledger: Callable[[], Decimal], position: PositionView | None,
                   may_hold: bool) -> Decimal | None:
        """The venue position to close; None (nothing sent, the issue blocks entries) when it cannot be read or the
        answer has no row for the symbol: unknown is never flat (S4-S5 audit, finding 7).

        Plan D2: from the snapshot row when there is one, so a flat symbol sends and reads nothing - unless the ledger
        holds what the row does not show (a fill since the snapshot): then the venue is asked. Without a row, only a
        symbol that could hold something reads its own; any other sends nothing and reports nothing, because what
        it holds would not be the strategy's (the whole-account check finds it and blocks every entry). The close
        itself is sized from a fresh read when it is sent (``size_reduction``)."""
        if position is not None:
            qty = Decimal(str(position.qty))
            if qty != 0 or ledger() == 0:
                return qty
        elif not may_hold:
            return None
        try:
            position = self.account.get_position(self.symbol)
        except Exception as exc:
            self._issue(f"flatten: venue position unavailable ({type(exc).__name__})")
            return None
        if position is None:
            self._issue("flatten: venue position unavailable (no row for the symbol)")
            return None
        return Decimal(str(position.qty))

    def _attempt(self, reason: str, qty: Decimal, bar: Bar, now: float, consume: Callable[[], None]) -> None:
        prefix = f"{self.symbol}:{bar.available_at}:{reason}:"
        intent_id = prefix + str(sum(1 for r in self._rows() if str(r["intent_id"]).startswith(prefix)))
        side = "sell" if qty > 0 else "buy"
        emergency = reason in EMERGENCY
        intent = OrderIntent(order_id=intent_id, source=self.executor.owner, symbol=self.symbol, side=side,
                             qty=Decimal(0) if emergency else abs(qty), price=bar.close, reduce_only=True,
                             reason=reason, close_venue_position=emergency)
        self.events.write(EventKind.INTENT, symbol=self.symbol, order_id=intent_id, side=side, qty=str(abs(qty)),
                          reduce_only=True, reason=reason)
        self._last, self._sent_at = intent_id, now
        self.executor.send_order(intent)
        refusal = self.executor.take_refusal(intent_id)
        if refusal is not None:
            self.events.write(EventKind.BLOCKED, symbol=self.symbol, order_id=intent_id,
                              reason=f"journal: {refusal}")
        elif not any(r["intent_id"] == intent_id for r in self._rows()):
            self.events.write(EventKind.BLOCKED, symbol=self.symbol, order_id=intent_id,
                              reason="not placed; see the log")
        try:
            consume()
        except Exception as exc:              # the order is out; its fill is consumed on a later call
            self._issue(f"flatten: fill consumption failed ({type(exc).__name__}: {exc})")

    def _issue(self, issue: str) -> None:
        self.events.write(EventKind.ISSUE, symbol=self.symbol, issue=issue, issue_kind=IssueKind.FLATTEN_FAILED.value)
        logger.warning("%s: %s", self.symbol, issue)


__all__ = ["EMERGENCY", "MAX_FAILURES", "RETRY_SECONDS", "Flattener"]
