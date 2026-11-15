"""The target runtime's main loop: warm up, then every cycle the market, exit and status sides.

Moved from runner/binance_target.py (migration step 10); split into three parts in S3c (ADR-0013 point 4):
``MarketPoller`` (market_poller.py), ``ExitSupervisor`` (exit_supervisor.py) and ``StatusReporter`` (status.py).
This class wires them, keeps the paper state, and runs one cycle. In execution mode the three sides are isolated:
one that raises does not stop the others in that cycle, and blocks every entry until it works again.

Everything it touches is injected: market data, account, sessions, event log, kill switch, drawdown guard, paper
store, status sink and the clock.
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
from trading.application.ports import (Clock, DrawdownGuard, EntryPauses, EventLog, ExchangeAccount, KillSwitch,
                                       MarketData, Notifier, PaperStore, StatusSink)
from trading.application.session import TargetSession
from trading.application.status import StatusReporter
from trading.domain.bars import HOUR, bar_from_kline
from trading.domain.events import EventKind
from trading.domain.issues import IssueKind as K
from trading.domain.strategy.overlay import StrategySpec

logger = logging.getLogger("binance_target")

PAPER_SCHEMA = 1


class TargetRuntime:
    def __init__(self, *, spec: StrategySpec, mode: str, market: MarketData, sessions: dict[str, TargetSession],
                 journal: EventLog, equity: EquitySource, clock: Clock, status_sink: StatusSink,
                 market_errors: tuple[type[BaseException], ...], account: ExchangeAccount | None = None,
                 final_market_errors: tuple[type[BaseException], ...] = (),
                 kill: KillSwitch | None = None, drawdown: DrawdownGuard | None = None,
                 paper_store: PaperStore | None = None, reset_paper: bool = False,
                 spec_passed: object = None, pauses: EntryPauses | None = None,
                 notifier: Notifier | None = None, alert_prefix: str = "",
                 event_log_failure: Callable[[], Issue | None] = lambda: None) -> None:
        if mode not in ("paper", "execution"):
            raise ValueError("mode must be paper or execution")
        self.spec, self.mode, self.market, self.account, self.clock = spec, mode, market, account, clock
        self.sessions, self.journal, self.equity = sessions, journal, equity
        self.kill: Any = kill
        self.drawdown: Any = drawdown
        self._status_sink, self._paper_store = status_sink, paper_store
        self.reset_paper = reset_paper
        self._saved_cursors: dict[str, int | None] | None = None
        self.pauses = pauses
        self.notifier = notifier
        self._cycle_errors: dict[str, Issue] = {}     # side -> its last failure, until the side succeeds
        self._event_log_failure = event_log_failure    # S3 audit finding 3: blocks entries while it lasts
        self._warmed = False                          # the whole warm-up succeeded (S3 audit finding 6)
        self.market_side = MarketPoller(market, sessions, mode=mode, journal=journal, clock=clock,
                                        market_errors=market_errors, final_market_errors=final_market_errors)
        self.exits = ExitSupervisor(sessions, spec=spec, journal=journal, clock=clock, account=account, equity=equity,
                                    kill=kill, drawdown=drawdown, pauses=pauses, notifier=notifier,
                                    alert_prefix=alert_prefix, market_refused=lambda: self.market_side.refused,
                                    decision_error=self.market_side.decision_errors.get,
                                    cycle_errors=self._runtime_issues, warmed=lambda: self._warmed)
        self.reporter = StatusReporter(sessions, mode=mode, clock=clock, spec_passed=spec_passed, notifier=notifier,
                                       kill=kill, drawdown=drawdown, equity=equity,
                                       market_failures=lambda: self.market_side.failures,
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
        """Load history; a paper run resumes from its saved state and replays the bars it missed."""
        now_ms = int(self.clock() * 1000)
        saved = self._load_paper_state() if self.mode == "paper" else {}
        for symbol, session in self.sessions.items():
            snap = saved.get(symbol)
            resume = snap["cursor"] if snap else None
            gap = 0 if resume is None else max(0, (now_ms // HOUR * HOUR - HOUR - resume) // HOUR)
            limit = self.spec.funding_window() + 30 + gap // 8 + 1
            for at, rate in self.market.get_funding_history(symbol, limit=limit):
                session.funding[round(at / HOUR) * HOUR] = rate
            rows = self.market.get_klines(symbol, "60", (session.bars.maxlen or 0) + gap, now_ms=now_ms)
            if snap is not None:
                if not any(row["start"] == resume for row in rows):
                    raise RuntimeError(f"{symbol}: saved paper state at {resume} cannot be resumed from the available "
                                       "market data; start a new paper record with --reset-paper-state")
                session.restore_paper(snap)
            for row in rows:
                bar = bar_from_kline(row)
                session.process_bar(bar, live=resume is not None and bar.open_time > resume, now_ms=now_ms)
            if len(session.bars) < self.spec.window_hours():
                raise RuntimeError(f"{symbol}: incomplete warmup ({len(session.bars)}/{self.spec.window_hours()})")
            self.journal.write(EventKind.WARMUP, symbol=symbol, bars=len(session.bars), last_open=session.cursor,
                               resumed_from=resume)
        self._warmed = True
        if self.mode == "paper":
            self.save_paper_state()
        if self.mode == "execution":
            self.exits.refresh_equity()
            self.exits.reconcile()
            self.exits.update_states()

    def warmup_or_hold(self) -> bool:
        """Warm up. In execution mode a market-data failure does not end the process: the exit side - equity,
        reconciliation, stop upkeep, risk actions - runs, and False asks the caller to try again later
        (step 14). Paper mode raises as before."""
        if self.mode != "execution":
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
    def _load_paper_state(self) -> dict[str, Any]:
        assert self._paper_store is not None
        if self.reset_paper:
            archived = self._paper_store.archive()
            if archived is not None:
                self.journal.write(EventKind.PAPER_RESET, archived=archived)
        data = self._paper_store.load()
        if data is None:
            return {}
        if (data.get("schema") != PAPER_SCHEMA or data.get("spec") != self.spec.record()
                or set(data["symbols"]) != set(self.sessions)):
            raise RuntimeError("saved paper state belongs to another spec or symbol set; "
                               "start a new paper record with --reset-paper-state")
        symbols: dict[str, Any] = data["symbols"]
        return symbols

    def save_paper_state(self) -> None:
        """Atomically persist every paper sleeve after the bars it has processed."""
        assert self._paper_store is not None
        cursors = {symbol: s.cursor for symbol, s in self.sessions.items()}
        if cursors == self._saved_cursors:
            return
        self._paper_store.save({"schema": PAPER_SCHEMA, "spec": self.spec.record(),
                                "symbols": {symbol: s.paper_snapshot() for symbol, s in self.sessions.items()}})
        self._saved_cursors = cursors

    # -- the cycle ------------------------------------------------------------------------------------------------
    def cycle(self) -> None:
        if self.mode == "paper":
            self.market_side.poll()
            self.save_paper_state()
            self._status_sink.write(self.status())
            return
        # Execution: three isolated sides (S3c). One that raises does not stop the others in this cycle.
        self._run_side("market", self._market)
        self._run_side("exit", self.exits.run)
        self._run_side("status", lambda: self._status_sink.write(self.status()))

    def _market(self) -> None:
        self.exits.refresh_equity_if_due()
        # The gate's symbol state must be this cycle's: a pause released or a kill cleared since the last exit side
        # must not refuse the next entry as PAUSED or HALTED (S3 audit finding 1).
        self.exits.update_states()
        for session in self.sessions.values():           # entries the rate limit refused on an earlier cycle
            session.retry_deferred(int(self.clock() * 1000))
        self.market_side.poll()

    def _run_side(self, name: str, run: Callable[[], None]) -> None:
        """A failing side blocks every entry at once, and at every reconciliation until it succeeds again."""
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
