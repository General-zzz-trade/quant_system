"""Account equity used for sizing: a value older than two minutes, or not positive, sizes to zero.

``CappedEquity`` (owner, 2026-11-16): the strategy shares the main account and sizes from
min(account equity, cap + the strategy's own cumulative realized PnL after fees). The owner chose that the base
grows with the strategy's profits and shrinks with its losses; it never exceeds the account's equity, the 3x
exposure cap still holds, and anything unknown (stale equity, an unreadable ledger) sizes to zero.

``StrategyFunding`` (tech lead, 2026-11-16): the realized PnL of the base includes the strategy's funding income on
its symbols, each counted from the strategy's first fill on it. It is asked of the venue at most every ten minutes;
a total older than two hours, or none yet, is unknown, so the capped base sizes to zero. A failed refresh never
raises: the last total is kept, and only the base sees it age. The loss stop counts the last total read in this
process whatever its age (``last_known``); only a total never read is unknown to it.
"""
from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from decimal import Decimal

from trading.application.ports import Clock, FundingIncome, FundingStore

logger = logging.getLogger("binance_target")

EQUITY_MAX_AGE = 120.0
FUNDING_REFRESH_SECONDS = 600.0     # at most one request per strategy symbol every ten minutes
FUNDING_MAX_AGE = 7200.0            # a funding total older than two hours is unknown
FUNDING_OVERLAP_MS = 24 * 3600 * 1000  # each read starts this long before the kept span ends (late-posted rows)
# Binance keeps the income history for three months; the adapter refuses a read that starts before this (88 days).
# Kept rows that end before it can never be completed: their total is partial, so it is unknown (audit of #147, 1).
FUNDING_HISTORY_MS = 88 * 86_400_000


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
    (``first_fill``: seconds, None for a symbol never traded, which adds nothing). ``symbols`` is a fixed list, or a
    function asked at every refresh (the ledger's owners: an earlier spec's symbol still counts; one that cannot be
    listed fails the refresh). The precondition that the account holds nothing outside the strategy ledger is what
    makes this funding the strategy's. Called on the runtime's cycle; ``clock`` is the runtime's."""

    def __init__(self, venue: FundingIncome, symbols: Sequence[str] | Callable[[], Sequence[str]],
                 first_fill: Callable[[str], float | None], clock: Clock, *, store: FundingStore | None = None) -> None:
        self.venue, self.first_fill, self.clock, self.store = venue, first_fill, clock, store
        fixed = () if callable(symbols) else tuple(symbols)
        self.symbols: Callable[[], Sequence[str]] = symbols if callable(symbols) else (lambda: fixed)
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
            for symbol in self.symbols():
                since = self.first_fill(symbol)
                if since is None:
                    continue
                if self.store is None:
                    total += Decimal(self.venue.get_funding_income(symbol, since_ms=int(since * 1000),
                                                                   now_ms=int(now * 1000)))
                else:
                    self._read(symbol, int(since * 1000), int(now * 1000))
            if self.store is not None:
                total = self._stored()
        except Exception:
            logger.warning("Strategy funding not refreshed; the last total is kept for at most two hours",
                           exc_info=True)
            return
        self.total, self.refreshed_at = total, now

    def _read(self, symbol: str, since_ms: int, now_ms: int) -> None:
        """Tech lead 2026-11-17: read from the first fill when the kept rows do not reach back to it, otherwise from
        ``FUNDING_OVERLAP_MS`` before their span ends, and keep what was read. A span that ended more than Binance's
        three months ago cannot be continued: the read raises, and the total is unknown from then on."""
        assert self.store is not None
        start = self._start(self.store.coverage(symbol), since_ms)
        rows = self.venue.get_funding_rows(symbol, since_ms=start, now_ms=now_ms)
        self.store.add(symbol, rows, from_ms=start, through_ms=now_ms)

    @staticmethod
    def _start(covered: tuple[int, int] | None, since_ms: int) -> int:
        """Where a symbol's next read starts: its first fill when the kept rows do not reach back to it, otherwise
        ``FUNDING_OVERLAP_MS`` before they end (never before the first fill)."""
        if covered is None or covered[0] > since_ms:
            return since_ms
        return max(since_ms, covered[1] - FUNDING_OVERLAP_MS)

    def _stored(self) -> Decimal:
        """The kept total since each symbol's first fill; raises when a symbol's kept rows do not reach back to it.
        (Rows that end more than 88 days ago are refused before: ``last_known`` checks ``_continuable`` first, and a
        refresh fails to read across such a gap.)"""
        assert self.store is not None
        total = Decimal(0)
        for symbol in self.symbols():
            since = self.first_fill(symbol)
            if since is None:
                continue
            covered = self.store.coverage(symbol)
            if covered is None or covered[0] > int(since * 1000):
                raise RuntimeError(f"strategy funding unknown: {symbol}'s kept rows do not reach its first fill")
            total += self.store.total(symbol, since_ms=int(since * 1000))
        return total

    def _continuable(self) -> None:
        """Raise when a traded symbol's next read would start more than Binance's history ago: what was paid after
        its kept rows can never be read, so no total is whole (audit of PR #147, finding 1: unknown, never a partial
        total; its re-audit, item 1: the read starts 24 h before the kept rows end, so this is checked at the read's
        start, not at their end)."""
        assert self.store is not None
        oldest = int(self.clock() * 1000) - FUNDING_HISTORY_MS
        for symbol in self.symbols():
            since = self.first_fill(symbol)
            if since is None:
                continue
            # No kept rows yet: the read starts at the first fill (audit of the re-audit fixes, S1).
            if self._start(self.store.coverage(symbol), int(since * 1000)) < oldest:
                raise RuntimeError(f"strategy funding unknown: {symbol}'s next read would start more than 88 days "
                                   "ago, beyond Binance's history")

    def current(self) -> Decimal:
        """The funding total; raises when it is older than two hours or was never read (the base sizes to zero)."""
        self.refresh_if_due()
        if self.refreshed_at is None or self.clock() - self.refreshed_at > FUNDING_MAX_AGE:
            raise RuntimeError("strategy funding unknown: no successful refresh in the last two hours")
        return self.total

    def last_known(self) -> Decimal:
        """The last funding total read in this process, whatever its age: what the loss stop counts (audit
        2026-11-16, finding 4: a refresh that keeps failing must not blind the stop, and an old total is nearer the
        truth than none). Raises only when no total was ever read in this process. With the store (tech lead
        2026-11-17), before the first read of this process (a restart): the kept total; raises only when the kept
        rows do not reach back to a first fill."""
        self.refresh_if_due()
        if self.store is not None:
            self._continuable()              # a gap that can never be read: unknown, also in a running process
        if self.refreshed_at is None and self.store is not None:
            return self._stored()            # not read in this process yet (a restart): what the store holds
        if self.refreshed_at is None:
            raise RuntimeError("strategy funding unknown: not read once in this process")
        return self.total


__all__ = ["CappedEquity", "EQUITY_MAX_AGE", "EquitySource", "FUNDING_HISTORY_MS", "FUNDING_MAX_AGE",
           "FUNDING_OVERLAP_MS", "FUNDING_REFRESH_SECONDS", "StrategyFunding"]
