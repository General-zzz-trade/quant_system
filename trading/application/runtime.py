"""The target runtime's main loop: warm up, then every cycle the market, exit and status sides.

Moved from runner/binance_target.py (migration step 10); split into three parts in S3c (ADR-0013 point 4):
``MarketPoller`` (market_poller.py), ``ExitSupervisor`` (exit_supervisor.py) and ``StatusReporter`` (status.py).
This class wires them, keeps the paper state, and runs one cycle. With ``ModePolicy.isolate_failures`` (execution)
the three sides are isolated: one that raises does not stop the others in that cycle, and blocks every entry until
it works again; paper raises.

Paper and execution run the same cycle (ADR-0013 plan B, S10 part 3); what differs is the ``ModePolicy``. In paper
the account is the simulated exchange, which is also the market side's bar listener, and the paper state is its
snapshot plus the cursors; the order journal keeps the rest.

Everything it touches is injected: market data, account, sessions, event log, kill switch, drawdown guard, paper
store and venue, status sink and the clocks.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from trading.application.equity import EquitySource
from trading.application.exit_supervisor import EQUITY_SECONDS, RECONCILE_SECONDS, RELEASE_COMMAND, ExitSupervisor
from trading.application.issues import Issue, write_issue
from trading.application.market_poller import (FUNDING_WAIT_MS, MARKET_ALERT_AFTER, MARKET_BACKOFF_MAX,
                                               POLL_SECONDS, MarketPoller)
from trading.application.mode_policy import ModePolicy
from trading.application.ports import (Clock, DrawdownGuard, EntryPauses, EventLog, ExchangeAccount, GateClock,
                                       KillSwitch, MarketData, Notifier, PaperStore, PaperVenue, StatusSink)
from trading.application.risk.loss_stop import StrategyLossStop
from trading.application.session import TargetSession
from trading.application.status import StatusReporter
from trading.domain.bars import HOUR, Bar, bar_from_kline
from trading.domain.events import EventKind
from trading.domain.issues import IssueKind as K
from trading.domain.strategy.overlay import StrategySpec

logger = logging.getLogger("binance_target")

PAPER_SCHEMA = 2        # 2: the simulated exchange's snapshot and the cursors (S10); 1 held the research sleeves


class TargetRuntime:
    def __init__(self, *, spec: StrategySpec, policy: ModePolicy, market: MarketData,
                 sessions: dict[str, TargetSession],
                 journal: EventLog, equity: EquitySource, clock: Clock, status_sink: StatusSink,
                 market_errors: tuple[type[BaseException], ...], account: ExchangeAccount | None = None,
                 final_market_errors: tuple[type[BaseException], ...] = (),
                 kill: KillSwitch | None = None, drawdown: DrawdownGuard | None = None,
                 paper_store: PaperStore | None = None,
                 spec_passed: object = None, pauses: EntryPauses | None = None,
                 notifier: Notifier | None = None, alert_prefix: str = "",
                 event_log_failure: Callable[[], Issue | None] = lambda: None,
                 config_digest: str | None = None,
                 account_check: Callable[[], list[Issue]] | None = None, venue: PaperVenue | None = None,
                 gate_clock: GateClock | None = None,
                 exit_observer: Callable[[str, str, str], None] | None = None,
                 loss_stop: StrategyLossStop | None = None) -> None:
        self.policy, self.venue = policy, venue
        self.spec, self.mode, self.market, self.account, self.clock = spec, policy.name, market, account, clock
        self.sessions, self.journal, self.equity = sessions, journal, equity
        self.kill: Any = kill
        self.drawdown: Any = drawdown
        self._status_sink, self._paper_store = status_sink, paper_store
        self.pauses = pauses
        self.notifier = notifier
        self._cycle_errors: dict[str, Issue] = {}     # side -> its last failure, until the side succeeds
        self._event_log_failure = event_log_failure    # S3 audit finding 3: blocks entries while it lasts
        self._warmed = False                          # the whole warm-up succeeded (S3 audit finding 6)
        # Paper: the last bar each symbol has decided on, and whether the saved state has been loaded (before that
        # nothing may be saved over it).
        self._decided: dict[str, int | None] = {}
        self._paper_loaded = False
        self.market_side = MarketPoller(market, sessions, policy=policy, journal=journal, clock=clock,
                                        market_errors=market_errors, final_market_errors=final_market_errors,
                                        listeners=(venue,) if venue is not None else (),
                                        settle=lambda: self.exits.settle(), gate_clock=gate_clock,
                                        on_step=self._decided_on)
        self.exits = ExitSupervisor(sessions, spec=spec, journal=journal, clock=clock, account=account, equity=equity,
                                    kill=kill, drawdown=drawdown, pauses=pauses, notifier=notifier,
                                    alert_prefix=alert_prefix, market_refused=lambda: self.market_side.refused,
                                    decision_error=self.market_side.decision_errors.get,
                                    cycle_errors=self._runtime_issues, warmed=lambda: self._warmed,
                                    account_check=account_check, exit_observer=exit_observer, loss_stop=loss_stop)
        self.reporter = StatusReporter(sessions, mode=policy.name, clock=clock, spec_passed=spec_passed,
                                       notifier=notifier, kill=kill, drawdown=drawdown, equity=equity,
                                       market_failures=lambda: self.market_side.failures, config_digest=config_digest,
                                       pause_status=self.exits.pause_status)

    def _runtime_issues(self) -> list[Issue]:
        """Problems of the runtime as a whole: a failing side, an event log that cannot be written."""
        log_failure = self._event_log_failure()
        return list(self._cycle_errors.values()) + ([log_failure] if log_failure is not None else [])

    # -- the parts, as the entry point and the tests have always called them --------------------------------------
    def refresh_equity(self) -> None:
        self.exits.refresh_equity()

    def reconcile(self) -> None:
        self.exits.reconcile()

    def risk_actions(self) -> None:
        self.exits.risk_actions()

    def poll(self) -> int:
        return self.market_side.poll()

    def status(self) -> dict[str, Any]:
        return self.reporter.status()

    def _alert_wait(self, issue: Issue) -> float:
        return self.exits.alert_wait(issue)

    @property
    def _market_retry_at(self) -> float:
        return self.market_side.retry_at

    # -- lifecycle ------------------------------------------------------------
    def warmup(self) -> None:
        """Load history. A paper run resumes from its saved state: the simulated exchange is restored, the exit side
        sees it and the order journal, and then the bars missed meanwhile are replayed in time order across the
        symbols, as the market side replays them (``ModePolicy.catch_up``)."""
        now_ms = int(self.clock() * 1000)
        saved = self._load_paper_state() if self._paper_store is not None else None
        if saved is not None:
            assert self.venue is not None
            self.venue.restore(saved["venue"])
        cursors: dict[str, Any] = saved["symbols"] if saved is not None else {}
        self.market_side.refresh_intervals(now_ms)      # R4: each symbol's funding interval, before its history
        missed: dict[str, list[Bar]] = {}
        resumed: dict[str, int | None] = {}
        for symbol, session in self.sessions.items():
            snap = cursors.get(symbol)
            resume = resumed[symbol] = snap["cursor"] if snap else None
            gap = 0 if resume is None else max(0, (now_ms // HOUR * HOUR - HOUR - resume) // HOUR)
            per = (session.funding_interval or HOUR) // HOUR      # R4: the symbol's interval; unknown, hourly
            limit = session.strategy.funding_window() + 30 + gap // per + 1   # R2: the strategy's windows
            for at, rate in self.market.get_funding_history(symbol, limit=limit):
                session.funding[round(at / HOUR) * HOUR] = rate
            rows = self.market.get_klines(symbol, "60", (session.bars.maxlen or 0) + gap, now_ms=now_ms)
            if snap is not None and not any(row["start"] == resume for row in rows):
                raise RuntimeError(f"{symbol}: saved paper state at {resume} cannot be resumed from the available "
                                   "market data; start a new paper record with --reset-paper-state")
            missed[symbol] = []
            for row in rows:
                bar = bar_from_kline(row)
                if resume is not None and bar.open_time > resume:
                    missed[symbol].append(bar)             # replayed below, in time order across the symbols
                else:
                    session.advance(bar)
            have = min(len(session.bars) + len(missed[symbol]), session.bars.maxlen or 0)
            if have < session.strategy.window_hours():
                raise RuntimeError(f"{symbol}: incomplete warmup ({have}/{session.strategy.window_hours()})")
            self._decided[symbol] = session.cursor
        self._warmed = True
        self._paper_loaded = self._paper_store is not None
        if any(missed.values()):
            # The exit side sees the restored venue and journal before any missed bar is decided. A bar the venue
            # had matched before a crash, but not yet decided on, is decided without being matched again.
            self.exits.refresh_equity()
            self.exits.reconcile()
            self.exits.update_states()
            self.market_side.replay(missed, matched=self.venue.has_matched if self.venue is not None else None)
        for symbol, session in self.sessions.items():
            self.journal.write(EventKind.WARMUP, symbol=symbol, bars=len(session.bars), last_open=session.cursor,
                               resumed_from=resumed[symbol])
        if self._paper_store is not None:
            self.save_paper_state()
        self.exits.refresh_equity()
        self.exits.reconcile()
        self.exits.update_states()

    def warmup_or_hold(self) -> bool:
        """Warm up. In execution mode a market-data failure does not end the process: the exit side - equity,
        reconciliation, stop upkeep, risk actions - runs, and False asks the caller to try again later
        (step 14). Without ``ModePolicy.isolate_failures`` (paper) a failure raises as before."""
        if not self.policy.isolate_failures:
            self.warmup()
            return True
        if self.market_side.due():
            for session in self.sessions.values():        # a partial earlier attempt must not leave bars behind
                session.bars.clear()
                session.funding.clear()
                session.cursor = None
            try:
                self.warmup()
                self.market_side.recovered()
                return True
            except self.market_side.errors as exc:
                self.market_side.failed(exc)
        # The exit and status sides of the hold are isolated like a cycle's (S3 audit finding 5). An error of the
        # warm-up itself other than a market-data failure still ends the process: a person looks before it runs.
        self._run_side("exit", self._hold_exit)
        self._run_side("status", lambda: self._status_sink.write(self.status()))
        return False

    def _hold_exit(self) -> None:
        self.exits.refresh_equity_if_due()
        self.exits.reconcile()
        self.exits.risk_actions()
        self.exits.update_states()

    # -- paper state -----------------------------------------------------------
    def _load_paper_state(self) -> dict[str, Any] | None:
        """The saved paper state, checked against this spec and symbol set. A new paper record (a reset) is set up by
        the composition root before the sessions are built, because it also starts a new order journal."""
        assert self._paper_store is not None
        data = self._paper_store.load()
        if data is None:
            return None
        if (data.get("schema") != PAPER_SCHEMA or data.get("spec") != self.spec.record()
                or set(data["symbols"]) != set(self.sessions)):
            raise RuntimeError("saved paper state belongs to another spec or symbol set; "
                               "start a new paper record with --reset-paper-state")
        return data

    def _decided_on(self, group: list[tuple[str, Bar]]) -> None:
        for symbol, bar in group:
            self._decided[symbol] = bar.open_time

    def paper_revision(self) -> object | None:
        """Changes whenever the paper document would; None until the saved state is loaded (S10 decision 4)."""
        if not self._paper_loaded or self.venue is None:
            return None
        return self.venue.revision, tuple(sorted(self._decided.items()))

    def paper_document(self) -> dict[str, Any]:
        """The paper state: the simulated exchange's snapshot and the last bar each symbol has decided on."""
        assert self.venue is not None
        return {"schema": PAPER_SCHEMA, "spec": self.spec.record(),
                "symbols": {symbol: {"cursor": cursor} for symbol, cursor in self._decided.items()},
                "venue": self.venue.snapshot()}

    def save_paper_state(self) -> None:
        """Persist the paper state after the bars processed and the exits taken."""
        assert self._paper_store is not None
        self._paper_store.save(self.paper_document())

    # -- the cycle ------------------------------------------------------------------------------------------------
    def cycle(self) -> None:
        # Three sides (S3c), isolated with ModePolicy.isolate_failures: one that raises does not stop the others.
        self._run_side("market", self._market)
        self._run_side("exit", self.exits.run)
        if self._paper_store is not None:
            self.save_paper_state()
        self._run_side("status", lambda: self._status_sink.write(self.status()))

    def _market(self) -> None:
        self.exits.refresh_equity_if_due()
        # The gate's symbol state must be this cycle's: a pause released or a kill cleared since the last exit side
        # must not refuse the next entry as PAUSED or HALTED (S3 audit finding 1). If it cannot be read, this
        # cycle's entries are blocked and the bars are still processed, so strategy reductions go on.
        try:
            self.exits.update_states()
        except Exception:
            logger.exception("Symbol states not refreshed; every entry is blocked this cycle")
            for session in self.sessions.values():
                session.executor.block_entries()
        for session in self.sessions.values():           # entries the rate limit refused on an earlier cycle
            session.retry_deferred(self.market_side.decision_ms())
        self.market_side.poll()

    def _run_side(self, name: str, run: Callable[[], None]) -> None:
        """A failing side blocks every entry at once, and at every reconciliation until it succeeds again. Without
        ``ModePolicy.isolate_failures`` (paper) it raises."""
        if not self.policy.isolate_failures:
            run()
            return
        try:
            run()
        except Exception as exc:
            issue = Issue(K.CYCLE_FAILED, None, f"{name} side failed: {type(exc).__name__}: {exc}")
            if self._cycle_errors.get(name) != issue:
                logger.exception("%s side failed; every entry is blocked", name)
                try:
                    write_issue(self.journal, None, issue)
                except Exception:               # the handler itself never raises out of the cycle
                    logger.exception("%s side failure not written to the event log", name)
            self._cycle_errors[name] = issue
            for session in self.sessions.values():
                session.executor.block_entries()
        else:
            self._cycle_errors.pop(name, None)


__all__ = ["EQUITY_SECONDS", "FUNDING_WAIT_MS", "MARKET_ALERT_AFTER", "MARKET_BACKOFF_MAX", "PAPER_SCHEMA",
           "POLL_SECONDS", "RECONCILE_SECONDS", "RELEASE_COMMAND", "TargetRuntime"]
