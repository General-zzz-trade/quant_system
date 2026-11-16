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

Deciding together (plan R3, owner 2026-11-16): a per-symbol strategy decides each symbol as its bar arrives, so a
late or failing symbol holds back nobody else. A cross-sectional strategy (``Strategy.cross_sectional``) decides every
symbol at one close with one call, once every symbol has that close's bar; while one is behind nobody decides (the
decision depends on the missing bar: fail closed), and once that close is too late to enter (``MAX_ENTRY_DELAY_MS``)
the symbols still behind have their entries blocked (DECISION_FAILED, once per close; paper raises).
"""
from __future__ import annotations

import logging
from collections.abc import Callable, Sequence

from trading.application.issues import Issue, write_issue
from trading.application.ports import BarListener, Clock, EventLog, GateClock, MarketData
from trading.application.session import MAX_ENTRY_DELAY_MS, TargetSession
from trading.application.mode_policy import ModePolicy
from trading.domain.bars import HOUR, Bar, bar_from_kline
from trading.domain.events import EventKind
from trading.domain.issues import IssueKind as K
from trading.domain.strategy.base import Strategy

logger = logging.getLogger("binance_target")

POLL_SECONDS = 5.0
FUNDING_WAIT_MS = 2 * 60_000
MARKET_BACKOFF_MAX = 300.0
MARKET_ALERT_AFTER = 5            # failures in a row before an alert (about 75 s with the 5/10/20/40 s backoff)
INTERVALS_RETRY_MS = 60_000       # a failed read of the funding intervals is tried again once a minute (plan R4)
INTERVALS_KEPT_MS = 2 * HOUR      # the last intervals read are used this long while reads fail; then unknown


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
        self.behind_at: int | None = None                 # the last close whose missing bars were reported (R3)
        self.intervals_at: int | None = None              # when the funding intervals were last read (R4)
        self.intervals_tried: int | None = None
        self.intervals_lost = False                       # every interval was set unknown after failed reads
        self.interval_issues: dict[str, Issue] = {}       # symbol -> why its funding interval is unknown (R4 audit)

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

    def together(self) -> Strategy | None:
        """The strategy every session shares when its symbols decide together, or None when each symbol decides on
        its own (R3): a cross-sectional strategy, and from R7d one that reads other symbols (``references``, e.g.
        BTCUSDT): a symbol's decision then depends on their bars too."""
        strategies = {id(session.strategy): session.strategy for session in self.sessions.values()}
        if not any(s.cross_sectional or s.references() for s in strategies.values()):
            return None
        if len(strategies) != 1:
            raise ValueError("a cross-sectional strategy must be the one every symbol uses")
        return next(iter(strategies.values()))

    def refresh_intervals(self, now_ms: int) -> None:
        """Each symbol's funding interval from the exchange (plan R4): read at warm-up and in the first poll of every
        hour (bars close on the hour). A failed read is tried again once a minute and the last intervals are kept for
        ``INTERVALS_KEPT_MS``; after that, and at once for a symbol the exchange does not list, the interval is
        unknown and the symbol takes no entries (``TargetSession.decide``)."""
        read_this_hour = self.intervals_at is not None and now_ms // HOUR == self.intervals_at // HOUR
        tried_lately = self.intervals_tried is not None and now_ms - self.intervals_tried < INTERVALS_RETRY_MS
        if not read_this_hour and not tried_lately:
            self.intervals_tried = now_ms
            try:
                hours = self.market.get_funding_intervals()
            except self.errors as exc:
                logger.warning("Funding intervals not read; retry in %ds: %s", INTERVALS_RETRY_MS // 1000, exc)
            else:
                self.intervals_at, self.intervals_lost = now_ms, False
                for symbol, session in self.sessions.items():
                    session.funding_interval = hours[symbol] * HOUR if symbol in hours else None
                    self._interval_issue(symbol, None if symbol in hours else
                                         f"{symbol}: the exchange lists no funding interval; entries refused")
                return
        if self.intervals_lost or (self.intervals_at is not None and now_ms - self.intervals_at <= INTERVALS_KEPT_MS):
            return
        self.intervals_lost = True
        since = "since the start" if self.intervals_at is None else f"for {INTERVALS_KEPT_MS // HOUR} hours"
        for symbol, session in self.sessions.items():
            session.funding_interval = None
            self._interval_issue(symbol, f"{symbol}: funding interval unknown: not read {since}; entries refused")

    def _interval_issue(self, symbol: str, text: str | None) -> None:
        """An unknown interval is the symbol's issue (blocks its entries, shows in its state, alerted when it lasts);
        written once while it lasts, cleared by a read that lists the symbol."""
        if text is None:
            self.interval_issues.pop(symbol, None)
            return
        issue = Issue(K.FUNDING_INTERVAL_UNKNOWN, symbol, text)
        if self.interval_issues.get(symbol) != issue:
            write_issue(self.journal, symbol, issue)
        self.interval_issues[symbol] = issue

    def symbol_issue(self, symbol: str) -> Issue | None:
        """The market side's issue for one symbol, for the exit side: a failed decision, else an unknown interval."""
        return self.decision_errors.get(symbol) or self.interval_issues.get(symbol)

    def _poll_market(self) -> int:
        processed = 0
        now_ms = int(self.clock() * 1000)
        self.refresh_intervals(now_ms)
        together = self.together()
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
                    # (approved 2026-11-14, ADR-0007). A cross-sectional strategy decides below, for every symbol.
                    session.process_bar(bar, live=together is None and k == len(new) - 1, now_ms=now_ms)
                    if together is None:
                        self.decision_errors.pop(symbol, None)      # together: the decision below clears it
                processed += len(new)
            except self.errors:
                raise                               # market data: the backoff in poll(), as before S3
            except Exception as exc:
                if not self.policy.isolate_failures:
                    raise                           # paper must not diverge from research silently
                # One symbol's malformed data or failing bar stops only that symbol (S3 audit finding 4).
                self._decision_failed(symbol, session, exc)
        if together is not None and processed:
            self._decide_together(together, now_ms)
        processed += self.replay(replay)
        if together is not None:
            # On the runtime clock in both modes: paper's gate clock moves only with new bars, so a symbol that stays
            # behind would never be late on it (R3 audit, finding 1).
            self._report_behind(together, int(self.clock() * 1000))
        return processed

    def replay(self, bars: dict[str, list[Bar]], *, matched: Callable[[str, Bar], bool] | None = None) -> int:
        """Every bar, in time order across the symbols (``catch_up == "replay_all"``), as research replays them. A bar
        ``matched`` says the venue has matched already (a restart after a crash between the matching and the
        decision) is not handed to the listeners again."""
        processed = 0
        for at in sorted({bar.open_time for new in bars.values() for bar in new}):
            group = [(symbol, bar) for symbol, new in bars.items() for bar in new if bar.open_time == at]
            self._step(group, matched)
            processed += len(group)
        return processed

    def _step(self, group: list[tuple[str, Bar]], matched: Callable[[str, Bar], bool] | None = None) -> None:
        """One bar time: advance the windows, let the listeners match the bar, settle it, then decide on it."""
        for symbol, bar in group:
            self.sessions[symbol].advance(bar)
        for symbol, bar in group:
            if matched is not None and matched(symbol, bar):
                continue
            for listener in self.listeners:
                listener.on_closed_bar(symbol, bar, self.sessions[symbol].funding_rate(bar))
        if self.policy.settle_new_bar_before_decision and self.settle is not None:
            self.settle()
        gate = self.gate_clock
        if gate is not None:
            gate.at(max(bar.available_at for _, bar in group))
        decide_ms = self.decision_ms()
        together = self.together()
        if together is None:
            for symbol, bar in group:
                self._guarded(symbol, lambda: self.sessions[symbol].decide(bar, now_ms=decide_ms))
        else:
            self._decide_together(together, decide_ms)
        if gate is not None:
            # An entry the orders-per-second limit refused goes again one poll later, before the next bar, as the
            # next cycle sends it in real time (S10 decision 2; R23). Deciding together, every symbol may have one.
            gate.advance(POLL_SECONDS)
            for symbol in (s for s, _ in group) if together is None else self.sessions:
                self._guarded(symbol, lambda: self.sessions[symbol].retry_deferred(gate.ms()), clears=False)
        if self.on_step is not None:
            self.on_step(group)

    def _decide_together(self, strategy: Strategy, decide_ms: int) -> None:
        """Every symbol at one close, with one call of the strategy - only when a new bar has just completed the close
        (the callers' condition), so a close is decided once. While a symbol is behind nobody decides."""
        if len({session.cursor for session in self.sessions.values()}) != 1:
            return
        try:
            targets = strategy.latest({symbol: session.window() for symbol, session in self.sessions.items()})
        except Exception as exc:
            if not self.policy.isolate_failures:
                raise                               # paper must not diverge from research silently
            for symbol, session in self.sessions.items():
                self._decision_failed(symbol, session, exc)
            return
        for symbol, session in self.sessions.items():
            bar = session.bars[-1]
            self._guarded(symbol, lambda: session.decide(bar, now_ms=decide_ms, target=targets[symbol]))

    def _report_behind(self, strategy: Strategy, decide_ms: int) -> None:
        """Once the newest close is too late to enter and a symbol still has no bar there, block that symbol's
        entries (once per close): the cross-sectional decision waits for it."""
        lead = max(self.sessions.values(), key=lambda session: session.cursor or 0)
        behind = [symbol for symbol, session in self.sessions.items() if session.cursor != lead.cursor]
        if not behind or not lead.bars or self.behind_at == lead.cursor:
            return
        if decide_ms - lead.bars[-1].available_at <= MAX_ENTRY_DELAY_MS:
            return
        self.behind_at = lead.cursor
        for symbol in behind:
            text = (f"{symbol}: no bar at the close {lead.cursor}; the {strategy.name} strategy decides every symbol "
                    "together, so no symbol decided there")
            if not self.policy.isolate_failures:
                raise RuntimeError(text)
            self._block(symbol, self.sessions[symbol], Issue(K.DECISION_FAILED, symbol, text), traceback=False)

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
        interval = session.funding_interval
        per = interval or HOUR                  # unknown: ask for as many settlements as an hourly symbol has
        for at, rate in self.market.get_funding_history(symbol, limit=3 + max(missing, 0) * HOUR // per):
            session.funding[round(at / HOUR) * HOUR] = rate
        # Wait for the symbol's settlement at or before the close (R4: its own interval). Unknown, nothing to wait
        # for: the session refuses entries anyway.
        if interval is not None and close // interval * interval not in session.funding \
                and now_ms - close < FUNDING_WAIT_MS:
            return []
        rows = self.market.get_klines(symbol, "60", int(min(max(missing, 1) + 2, 1500)), now_ms=now_ms)
        return [bar for bar in map(bar_from_kline, rows) if bar.open_time > session.cursor]

    def _decision_failed(self, symbol: str, session: TargetSession, exc: Exception) -> None:
        """One symbol's bar raised (S3c): only its entries are blocked, the other symbols and the exit side go
        on, and the next bar decides again. Until a bar succeeds the issue is re-applied every reconciliation."""
        issue = Issue(K.DECISION_FAILED, symbol, f"{symbol}: bar not processed: {type(exc).__name__}: {exc}")
        self._block(symbol, session, issue, traceback=True)

    def _block(self, symbol: str, session: TargetSession, issue: Issue, *, traceback: bool) -> None:
        if self.decision_errors.get(symbol) != issue:
            write_issue(self.journal, symbol, issue)
            if traceback:
                logger.exception("%s bar not processed; its entries are blocked", symbol)
            else:
                logger.warning("%s; its entries are blocked", issue)
        self.decision_errors[symbol] = issue
        session.executor.block_entries()


__all__ = ["FUNDING_WAIT_MS", "MARKET_ALERT_AFTER", "MARKET_BACKOFF_MAX", "MarketPoller", "POLL_SECONDS"]
