"""The strategy loss stop (owner, 2026-11-16; it replaced "3 losses in a row").

Once the strategy has lost ``limit`` of its capital - the ledger's realized PnL after fees, the funding income and
the open positions' unrealized PnL at the venue's mark - it sets the operator kill switch with flatten. The audited
kill path then closes every position and refuses entries, and the kill file is the durable latch: the runtime only
ever sets it, a person clears it. The loss stays in the ledger, so clearing alone sets it again; going on means a
new pilot in a new state directory.

Once reached, the stop stays reached for the life of the process (audit 2026-11-16): every check reads the kill
switch, sets it again if it is not active with flatten, and returns a finding that blocks every entry, so a person's
clear lets nothing through. A restart reads the loss again in its warm-up, before any entry can be decided.

A loss that cannot be known closes nothing and blocks every entry (an issue, until it is known again). So does a
kill switch that could not be set, which is tried again at every check.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from decimal import Decimal
from typing import Protocol

from trading.application.issues import Issue
from trading.application.ports import Clock, KillLatch
from trading.domain.issues import IssueKind as K

LOSS_STOP_SECONDS = 60.0     # how often the loss is read while it is known (one whole-account position request)


class MarkView(Protocol):
    @property
    def mark_price(self) -> Decimal | None: ...


def strategy_unrealized(ledger: Iterable[tuple[str, Decimal, Decimal | None]],
                        venue: Callable[[str], MarkView | None]) -> Decimal:
    """The strategy's own unrealized PnL: for each ledger position (symbol, quantity, entry price), ledger quantity x
    (the venue's mark for the symbol - ledger entry). The venue's quantity and entry price are not the strategy's on a
    shared account: a person's position on the same symbol is netted into them (audit 2026-11-16, findings 1 and 2).
    A ledger quantity of 0 adds nothing whatever the venue shows. A held ledger position whose symbol the venue
    reports no row or no usable mark for, or whose own quantity or entry price cannot be used, raises: unknown,
    never 0."""
    total = Decimal(0)
    for symbol, qty, entry in ledger:
        if not qty.is_finite():
            raise RuntimeError(f"{symbol}: ledger quantity {qty} cannot be used")
        if qty == 0:
            continue
        if entry is None or not (entry.is_finite() and entry > 0):
            raise RuntimeError(f"{symbol}: ledger entry price {entry} cannot be used")
        position = venue(symbol)
        if position is None:
            raise RuntimeError(f"{symbol}: the ledger holds {qty} and the venue reports no row for the symbol")
        mark = position.mark_price
        if mark is None or not (mark.is_finite() and mark > 0):
            raise RuntimeError(f"{symbol}: the ledger holds {qty} and the venue reports no usable mark price")
        total += qty * (mark - entry)
    return total


class StrategyLossStop:
    def __init__(self, capital: Decimal, limit: Decimal, pnl: Callable[[], Decimal | None], kill: KillLatch, *,
                 clock: Clock) -> None:
        self.capital, self.limit, self.pnl, self.kill, self.clock = Decimal(capital), Decimal(limit), pnl, kill, clock
        self._checked_at: float | None = None
        self._findings: list[Issue] = []
        self._trip: str | None = None
        self._reached: str | None = None            # the trip reason, from the first read at or beyond the limit

    def check(self) -> list[Issue]:
        """Read the loss when due and set the kill switch at the limit. Never raises; returns what blocks every entry
        meanwhile: an unknown loss, a switch that could not be set, and - once the limit was reached, for the life
        of the process - the stop itself. From then on every call reads the kill switch and sets it again if a
        person cleared it; the loss is not read again."""
        now = self.clock()
        if self._reached is None:
            if self._checked_at is not None and not self._findings and now - self._checked_at < LOSS_STOP_SECONDS:
                return []
            self._checked_at = now
            self._findings = self._evaluate()
        if self._reached is not None:
            self._findings = self._hold(self._reached)
        return list(self._findings)

    def take_trip(self) -> str | None:
        """The reason of a trip not reported yet (for the alert), once."""
        trip, self._trip = self._trip, None
        return trip

    def _evaluate(self) -> list[Issue]:
        try:
            raw = self.pnl()
            if raw is None or not Decimal(raw).is_finite():
                raise ValueError(f"PnL {raw!r}")
            pnl = Decimal(raw)
        except Exception as exc:
            return [Issue(K.STRATEGY_LOSS_STOP, None, f"strategy loss unknown ({type(exc).__name__}: {exc}); "
                                                      "every entry is blocked, nothing is closed")]
        if pnl <= -self.limit * self.capital:
            self._reached = (f"strategy loss stop: the strategy has lost {-pnl:.2f} USDT of its {self.capital} USDT "
                             f"capital (limit {(self.limit * 100).normalize():f} %); every position is closed and "
                             "entries are refused. Clearing the kill switch alone sets it again: going on means a "
                             "new pilot in a new state directory")
        return []

    def _hold(self, reason: str) -> list[Issue]:
        """Reached: the switch is kept active with flatten, and this process refuses every entry until it ends - a
        person's clear cannot let one through before the switch is set again (audit 2026-11-16, finding 6)."""
        try:
            state = self.kill.state()
            if not (state.active and state.flatten):
                self.kill.trip(reason)
                self._trip = reason
        except Exception as exc:
            return [Issue(K.STRATEGY_LOSS_STOP, None, "strategy loss stop reached but the kill switch could not be "
                                                      f"set ({type(exc).__name__}: {exc}); every entry is blocked")]
        return [Issue(K.STRATEGY_LOSS_STOP, None, "strategy loss stop reached: this process refuses every entry "
                                                  "until it ends and keeps the kill switch set with flatten")]


__all__ = ["LOSS_STOP_SECONDS", "StrategyLossStop", "strategy_unrealized"]
