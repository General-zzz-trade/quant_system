"""The market side of a cycle (ADR-0013 point 4, S3c): take in newly closed bars, back off on market-data
failures, and hand each new bar to its symbol's session.

A market-data failure is retried with backoff instead of ending the process. With ``ModePolicy.isolate_failures``
(execution) one symbol's bar that raises blocks only that symbol (DECISION_FAILED) and the next bar decides again;
paper raises, because a paper record must not diverge from research silently.

What differs between the modes is read from ``ModePolicy`` (ADR-0013, S10 part 3):

- ``latest_only`` (execution): the bars missed during an outage only extend the window; the newest decides.
- ``replay_all`` (paper): every new bar is replayed in time order across the symbols. At each bar time the windows
  advance, the bar listeners (the simulated exchange) match the bar, the exit side settles it when
  ``settle_new_bar_before_decision``, and the sessions decide - on the gate clock at the bar's close when the
  entry gates run on bar time (``limiter_clock``), where an entry the per-second limit deferred is asked again one
  poll (5 s) later in simulated time.
"""
from __future__ import annotations

import logging
from collections.abc import Callable, Sequence

from trading.application.issues import Issue, write_issue
from trading.application.ports import BarListener, Clock, EventLog, GateClock, MarketData
from trading.application.session import FUNDING_INTERVAL, TargetSession
from trading.application.mode_policy import ModePolicy
from trading.domain.bars import HOUR, Bar, bar_from_kline
from trading.domain.events import EventKind
from trading.domain.issues import IssueKind as K

logger = logging.getLogger("binance_target")

POLL_SECONDS = 5.0
FUNDING_WAIT_MS = 2 * 60_000
MARKET_BACKOFF_MAX = 300.0
MARKET_ALERT_AFTER = 5            # failures in a row before an alert (about 75 s with the 5/10/20/40 s backoff)


class MarketPoller:
    def __init__(self, market: MarketData, sessions: dict[str, TargetSession], *, policy: ModePolicy,
                 journal: EventLog, clock: Clock, market_errors: tuple[type[BaseException], ...],
                 final_market_errors: tuple[type[BaseException], ...] = (), listeners: Sequence[BarListener] = (),
                 settle: Callable[[], None] | None = None, gate_clock: GateClock | None = None,
                 on_step: Callable[[list[tuple[str, Bar]]], None] | None = None) -> None:
        self.market, self.sessions, self.policy, self.journal, self.clock = market, sessions, policy, journal, clock
        self.listeners, self.settle, self.gate_clock, self.on_step = tuple(listeners), settle, gate_clock, on_step
        self.errors = market_errors
        self.final_errors = final_market_errors           # the exchange says retrying will not help (step 14)
        self.failures, self.retry_at = 0, 0.0
        self.refused: str | None = None                   # set by such an error, cleared by a success
        self.decision_errors: dict[str, Issue] = {}       # symbol -> its last failed bar, until a bar succeeds

    def due(self) -> bool:
        return self.clock() >= self.retry_at

    def recovered(self) -> None:
        self.failures, self.refused = 0, None

    def poll(self) -> int:
        """Process newly closed bars. A market-data failure is logged and retried with backoff instead of
        ending the process, so reconciliation, stops and risk actions keep running in the same cycle."""
        if not self.due():
            return 0
        try:
            processed = self._poll_market()
        except self.errors as exc:
            self.failed(exc)
            return 0
        self.recovered()
        return processed

    def failed(self, exc: BaseException) -> None:
        """Count, back off, and tell a person: after MARKET_ALERT_AFTER failures in a row, or at once when the
        exchange calls the error final - entries then stay blocked until market data works again."""
        self.failures += 1
        final = isinstance(exc, self.final_errors)
        delay = MARKET_BACKOFF_MAX if final else min(POLL_SECONDS * 2 ** (self.failures - 1), MARKET_BACKOFF_MAX)
        self.retry_at = self.clock() + delay
        text = f"{type(exc).__name__}: {exc}"
        self.journal.write(EventKind.ISSUE, issue=f"market data unavailable: {text}", issue_kind=K.MARKET_DATA.value,
                           failures=self.failures, retry_in=delay)
        logger.warning("Market data unavailable (%d in a row); retry in %.0fs: %s", self.failures, delay, exc)
        if final and self.refused is None:
            self.refused = f"market data refused by the exchange ({text}); entries blocked"
            self.journal.write(EventKind.ALERT, reason=f"{self.refused} until it works again")
        elif not final and self.failures == MARKET_ALERT_AFTER:
            self.journal.write(EventKind.ALERT,
                               reason=f"market data unavailable {MARKET_ALERT_AFTER} times in a row: {text}")

    def decision_ms(self) -> int:
        """The entry gates' clock now: the gate clock when they run on bar time, the runtime clock otherwise."""
        return self.gate_clock.ms() if self.gate_clock is not None else int(self.clock() * 1000)

    def _poll_market(self) -> int:
        processed = 0
        now_ms = int(self.clock() * 1000)
        replay: dict[str, list[Bar]] = {}
        for symbol, session in self.sessions.items():
            try:
                new = self._new_bars(symbol, session, now_ms)
                if self.policy.catch_up == "replay_all":
                    replay[symbol] = new
                    continue
                for k, bar in enumerate(new):
                    # Execution decides on the newest bar only: older bars missed during an outage just extend the
                    # window. A late newest bar may still reduce; entries are refused as "processed too late"
                    # (approved 2026-11-14, ADR-0007).
                    session.process_bar(bar, live=k == len(new) - 1, now_ms=now_ms)
                    self.decision_errors.pop(symbol, None)
                processed += len(new)
            except self.errors:
                raise                               # market data: the backoff in poll(), as before S3
            except Exception as exc:
                if not self.policy.isolate_failures:
                    raise                           # paper must not diverge from research silently
                # One symbol's malformed data or failing bar stops only that symbol (S3 audit finding 4).
                self._decision_failed(symbol, session, exc)
        return processed + self.replay(replay)

    def replay(self, bars: dict[str, list[Bar]]) -> int:
        """Every bar, in time order across the symbols (``catch_up == "replay_all"``), as research replays them."""
        processed = 0
        for at in sorted({bar.open_time for new in bars.values() for bar in new}):
            group = [(symbol, bar) for symbol, new in bars.items() for bar in new if bar.open_time == at]
            self._step(group)
            processed += len(group)
        return processed

    def _step(self, group: list[tuple[str, Bar]]) -> None:
        """One bar time: advance the windows, let the listeners match the bar, settle it, then decide on it."""
        for symbol, bar in group:
            self.sessions[symbol].advance(bar)
        for symbol, bar in group:
            for listener in self.listeners:
                listener.on_closed_bar(symbol, bar, self.sessions[symbol].funding_rate(bar))
        if self.policy.settle_new_bar_before_decision and self.settle is not None:
            self.settle()
        gate = self.gate_clock
        if gate is not None:
            gate.at(max(bar.available_at for _, bar in group))
        decide_ms = self.decision_ms()
        for symbol, bar in group:
            self._guarded(symbol, lambda: self.sessions[symbol].decide(bar, now_ms=decide_ms))
        if gate is not None:
            # An entry the orders-per-second limit refused goes again one poll later, before the next bar, as the
            # next cycle sends it in real time (S10 decision 2; R23).
            gate.advance(POLL_SECONDS)
            for symbol, _ in group:
                self._guarded(symbol, lambda: self.sessions[symbol].retry_deferred(gate.ms()), clears=False)
        if self.on_step is not None:
            self.on_step(group)

    def _guarded(self, symbol: str, run: Callable[[], None], *, clears: bool = True) -> None:
        try:
            run()
            if clears:
                self.decision_errors.pop(symbol, None)
        except Exception as exc:
            if not self.policy.isolate_failures:
                raise                               # paper must not diverge from research silently
            self._decision_failed(symbol, self.sessions[symbol], exc)

    def _new_bars(self, symbol: str, session: TargetSession, now_ms: int) -> list[Bar]:
        assert session.cursor is not None
        close = session.cursor + 2 * HOUR
        if now_ms < close:
            return []
        missing = (now_ms // HOUR * HOUR - (session.cursor + HOUR)) // HOUR
        for at, rate in self.market.get_funding_history(symbol, limit=3 + max(missing, 0) // 8):
            session.funding[round(at / HOUR) * HOUR] = rate
        settlement = close // FUNDING_INTERVAL * FUNDING_INTERVAL
        if settlement not in session.funding and now_ms - close < FUNDING_WAIT_MS:
            return []
        rows = self.market.get_klines(symbol, "60", int(min(max(missing, 1) + 2, 1500)), now_ms=now_ms)
        return [bar for bar in map(bar_from_kline, rows) if bar.open_time > session.cursor]

    def _decision_failed(self, symbol: str, session: TargetSession, exc: Exception) -> None:
        """One symbol's bar raised (S3c): only its entries are blocked, the other symbols and the exit side go
        on, and the next bar decides again. Until a bar succeeds the issue is re-applied every reconciliation."""
        issue = Issue(K.DECISION_FAILED, symbol, f"{symbol}: bar not processed: {type(exc).__name__}: {exc}")
        if self.decision_errors.get(symbol) != issue:
            write_issue(self.journal, symbol, issue)
            logger.exception("%s bar not processed; its entries are blocked", symbol)
        self.decision_errors[symbol] = issue
        session.executor.block_entries()


__all__ = ["FUNDING_WAIT_MS", "MARKET_ALERT_AFTER", "MARKET_BACKOFF_MAX", "MarketPoller", "POLL_SECONDS"]
