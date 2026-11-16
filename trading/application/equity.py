"""Account equity used for sizing: a value older than two minutes, or not positive, sizes to zero.

``CappedEquity`` (owner, 2026-11-16): the strategy shares the main account and sizes from
min(account equity, cap + the strategy's own cumulative realized PnL after fees). The owner chose that the base
grows with the strategy's profits and shrinks with its losses; it never exceeds the account's equity, the 3x
exposure cap still holds, and anything unknown (stale equity, an unreadable ledger) sizes to zero.
"""
from __future__ import annotations

from collections.abc import Callable
from decimal import Decimal

from trading.application.ports import Clock

EQUITY_MAX_AGE = 120.0


class EquitySource:
    def __init__(self, clock: Clock) -> None:
        self.value, self.at, self.clock = Decimal(0), 0.0, clock

    def update(self, equity: Decimal) -> None:
        self.value, self.at = Decimal(str(equity)), self.clock()

    def current(self) -> Decimal:
        if self.value <= 0 or self.clock() - self.at > EQUITY_MAX_AGE:
            return Decimal(0)  # stale equity sizes to zero
        return self.value


class CappedEquity:
    def __init__(self, account: EquitySource, cap: Decimal, realized: Callable[[], Decimal | None]) -> None:
        self.account, self.cap, self.realized = account, Decimal(cap), realized

    def current(self) -> Decimal:
        account = self.account.current()
        if account <= 0:
            return Decimal(0)
        try:
            pnl = self.realized()
        except Exception:                 # an unreadable ledger is unknown: size to zero, never guess
            return Decimal(0)
        if pnl is None or not Decimal(pnl).is_finite():
            return Decimal(0)
        return max(Decimal(0), min(account, self.cap + Decimal(pnl)))


__all__ = ["CappedEquity", "EQUITY_MAX_AGE", "EquitySource"]
