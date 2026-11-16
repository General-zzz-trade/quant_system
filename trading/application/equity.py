"""Account equity used for sizing: a value older than two minutes, or not positive, sizes to zero.

``CappedEquity`` (owner, 2026-11-16): the strategy shares the main account and sizes from
min(account equity, cap + the strategy's own cumulative realized PnL after fees). The owner chose that the base
grows with the strategy's profits and shrinks with its losses; it never exceeds the account's equity, the 3x
exposure cap still holds, and anything unknown (stale equity, an unreadable ledger) sizes to zero.

``StrategyFunding`` (tech lead, 2026-11-16): the realized PnL of the base includes the strategy's funding income on
its symbols, each counted from the strategy's first fill on it. It is asked of the venue at most every ten minutes;
a total older than two hours, or none yet, is unknown, so the capped base sizes to zero. A failed refresh never
raises: the last total is kept, and only the base sees it age.
"""
from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from decimal import Decimal

from trading.application.ports import Clock, FundingIncome

logger = logging.getLogger("binance_target")

EQUITY_MAX_AGE = 120.0
FUNDING_REFRESH_SECONDS = 600.0     # at most one request per strategy symbol every ten minutes
FUNDING_MAX_AGE = 7200.0            # a funding total older than two hours is unknown


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


class StrategyFunding:
    """The strategy's funding income: per symbol, what the venue reports from the strategy's first fill on it
    (``first_fill``: seconds, None for a symbol never traded, which adds nothing). The precondition that the account
    holds nothing outside the strategy ledger is what makes this funding the strategy's. Called on the runtime's
    cycle; ``clock`` is the runtime's."""

    def __init__(self, venue: FundingIncome, symbols: Sequence[str], first_fill: Callable[[str], float | None],
                 clock: Clock) -> None:
        self.venue, self.symbols, self.first_fill, self.clock = venue, tuple(symbols), first_fill, clock
        self.total = Decimal(0)
        self.refreshed_at: float | None = None      # the last successful refresh
        self.attempted_at: float | None = None

    def refresh_if_due(self) -> None:
        now = self.clock()
        if self.attempted_at is not None and now - self.attempted_at < FUNDING_REFRESH_SECONDS:
            return
        self.attempted_at = now
        try:
            total = Decimal(0)
            for symbol in self.symbols:
                since = self.first_fill(symbol)
                if since is not None:
                    total += Decimal(self.venue.get_funding_income(symbol, since_ms=int(since * 1000),
                                                                   now_ms=int(now * 1000)))
        except Exception:
            logger.warning("Strategy funding not refreshed; the last total is kept for at most two hours",
                           exc_info=True)
            return
        self.total, self.refreshed_at = total, now

    def current(self) -> Decimal:
        """The funding total; raises when it is older than two hours or was never read (the base sizes to zero)."""
        self.refresh_if_due()
        if self.refreshed_at is None or self.clock() - self.refreshed_at > FUNDING_MAX_AGE:
            raise RuntimeError("strategy funding unknown: no successful refresh in the last two hours")
        return self.total


__all__ = ["CappedEquity", "EQUITY_MAX_AGE", "EquitySource", "FUNDING_MAX_AGE", "FUNDING_REFRESH_SECONDS",
           "StrategyFunding"]
