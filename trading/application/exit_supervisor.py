"""The exit side of a cycle (ADR-0013 point 4, S3c): equity and the drawdown guard, reconciliation, stop upkeep,
entry pauses, and the kill switch, drawdown and breached-stop exits.

Every problem is a typed issue (ADR-0014); what it does to trading comes from the table in
``trading.domain.issues``. A failing exit is reported, never raised: one symbol's broken exit must not stop the
other symbol's, nor the next cycle. Moved from runtime.py without changes in behaviour.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from decimal import Decimal
from typing import Any

from trading.application.alerts import ALERT_AFTER_SECONDS, IssueAlerts
from trading.application.equity import EquitySource
from trading.application.issues import Issue, with_effect, write_issue
from trading.application.ports import (Clock, DrawdownGuard, EntryPauses, EventLog, ExchangeAccount, KillSwitch,
                                       Notifier)
from trading.application.session import TargetSession
from trading.domain.events import EventKind
from trading.domain.issues import Effect, IssueKind as K
from trading.domain.strategy.overlay import StrategySpec

logger = logging.getLogger("binance_target")

RECONCILE_SECONDS = 5.0
EQUITY_SECONDS = 60.0
RELEASE_COMMAND = "python -m trading.infrastructure.entry_pause --state-dir <state dir> release --symbol {symbol} " \
                  "--reason <why>"


class ExitSupervisor:
    def __init__(self, sessions: dict[str, TargetSession], *, spec: StrategySpec, journal: EventLog, clock: Clock,
                 account: ExchangeAccount | None, equity: EquitySource, kill: KillSwitch | None,
                 drawdown: DrawdownGuard | None,
                 pauses: EntryPauses | None, notifier: Notifier | None, alert_prefix: str,
                 market_refused: Callable[[], str | None], decision_error: Callable[[str], Issue | None],
                 cycle_errors: Callable[[], list[Issue]]) -> None:
        self.sessions, self.spec, self.journal, self.clock = sessions, spec, journal, clock
        self.account, self.equity, self.pauses = account, equity, pauses
        self.kill: Any = kill               # present in execution mode, the only mode that runs the exit side
        self.drawdown: Any = drawdown
        self._market_refused, self._decision_error, self._cycle_errors = market_refused, decision_error, cycle_errors
        self.last_equity = self.last_reconcile = 0.0
        self.exit_errors: dict[str, Issue] = {}     # symbol -> the last exit failure, until an exit succeeds
        self.issue_alerts = IssueAlerts(notifier, clock, prefix=alert_prefix, wait=self.alert_wait) \
            if notifier is not None else None

    def run(self) -> None:
        """The exit side of one cycle: reconcile when due, then the risk actions."""
        if self.clock() - self.last_reconcile >= RECONCILE_SECONDS:
            self.reconcile()
        self.risk_actions()

    def refresh_equity_if_due(self) -> None:
        if self.clock() - self.last_equity >= EQUITY_SECONDS:
            self.refresh_equity()

    def refresh_equity(self) -> None:
        try:
            assert self.account is not None
            equity = self.account.get_account_equity()
            self.equity.update(Decimal(str(equity)))
            self.drawdown.observe(float(equity))
        except Exception:
            self.drawdown.invalidate("exchange equity unavailable")
            logger.warning("Equity refresh failed; entries blocked", exc_info=True)
        self.last_equity = self.clock()

    def reconcile(self) -> None:
        refused_text = self._market_refused()
        cycle_errors = self._cycle_errors()
        for symbol, session in self.sessions.items():
            try:
                session.consume_fills()
                session.executor.notify_status()
                issues = session.executor.reconcile_account(symbol) + session.stops.sync()
            except Exception as exc:
                issues = [Issue(K.RECONCILE_FAILED, symbol, f"reconciliation failed: {type(exc).__name__}: {exc}")]
                logger.exception("%s reconciliation failed; entries blocked", symbol)
            for issue in issues:
                write_issue(self.journal, symbol, issue)
            flatten = with_effect(issues, Effect.FLATTEN_SYMBOL)
            if flatten:
                self._exit(symbol, session, flatten[0].kind.value)
            refused = Issue(K.MARKET_REFUSED, symbol, f"{symbol}: {refused_text}") if refused_text is not None \
                else None
            if refused is not None:
                write_issue(self.journal, symbol, refused)
            more = [self._save_pause(symbol, session), self._require_stop(symbol, session),
                    self.exit_errors.get(symbol), self._decision_error(symbol), refused]
            found = issues + [issue for issue in more if issue is not None]
            self._block(session, found + cycle_errors)
            if self.issue_alerts is not None:
                self.issue_alerts.observe(symbol, found)
        if self.issue_alerts is not None:
            self.issue_alerts.observe("runtime", cycle_errors)
        self.last_reconcile = self.clock()

    def _block(self, session: TargetSession, issues: list[Issue]) -> None:
        """Entry blocks from the ADR-0014 table; reductions are never blocked."""
        if with_effect(issues, Effect.BLOCK_ALL_ENTRIES):
            for other in self.sessions.values():
                other.executor.block_entries()
        elif with_effect(issues, Effect.BLOCK_SYMBOL_ENTRIES):
            session.executor.block_entries()

    def alert_wait(self, issue: Issue) -> float:
        """A maker entry rests at the venue by design: an unresolved order is news only after its lifetime."""
        if issue.kind is K.UNRESOLVED_ORDER and self.spec.execution == "maker":
            return self.spec.ttl_bars * 3600.0 + 600.0
        return ALERT_AFTER_SECONDS

    def _save_pause(self, symbol: str, session: TargetSession) -> Issue | None:
        """Pause the symbol's entries for every stop trigger the order journal holds that no pause covers yet
        (decided 2026-11-14 / 2026-11-15). Derived from durable rows, so a pause that could not be written -
        or a restart before it was - is written on a later reconciliation; until then entries are blocked."""
        if self.pauses is None:
            return None
        try:
            triggers = session.stops.triggers()
            new_triggers = {t: c for t, c in triggers.items() if t not in self.pauses.covered(symbol)}
            if not new_triggers:
                return None
            breached = [c for c in new_triggers.values() if c.kind is K.STOP_BREACHED]
            cause = str(breached[0] if breached else next(iter(new_triggers.values())))
            new = self.pauses.pause(symbol, cause, triggers=sorted(new_triggers))
        except Exception as exc:
            issue = Issue(K.PAUSE_NOT_SAVED, symbol,
                          f"{symbol}: entry pause not saved ({type(exc).__name__}); entries blocked")
            write_issue(self.journal, symbol, issue)
            logger.exception("%s entry pause not saved", symbol)
            return issue
        if new:
            self.journal.write(EventKind.ALERT, symbol=symbol,
                               reason=f"entries paused: {cause}; a person releases them "
                                      f"with: {RELEASE_COMMAND.format(symbol=symbol)}")
            logger.error("%s entries paused: %s", symbol, cause)
        return None

    def pause_status(self, symbol: str) -> str | None:
        if self.pauses is None:
            return None
        try:
            return self.pauses.reason(symbol)
        except Exception:
            return "unreadable"

    def _require_stop(self, symbol: str, session: TargetSession) -> Issue | None:
        """ADR-0005 4b: a confirmed position without an active exchange stop blocks entries. Checked after a
        breached stop's flatten, so a position that flatten closed blocks nothing."""
        try:
            unprotected = session.stops.unprotected()
        except Exception:
            logger.exception("%s stop check failed; entries blocked", symbol)
            unprotected = True
        if not unprotected:
            return None
        issue = Issue(K.STOP_MISSING, symbol, f"{symbol}: position without an active protective stop; entries blocked")
        write_issue(self.journal, symbol, issue)
        return issue

    def risk_actions(self) -> None:
        kill = self.kill.state()
        close = (kill.active and kill.flatten) or self.drawdown.latched()
        for symbol, session in self.sessions.items():
            if close:
                self._exit(symbol, session, "kill_switch" if kill.active else "drawdown_latched")
            elif kill.active or self.pause_status(symbol) is not None:
                # New exposure is blocked, so a resting entry must not fill either (audit after step 12b).
                self._exit(symbol, session, None)

    def _exit(self, symbol: str, session: TargetSession, reason: str | None) -> None:
        """Flatten (or, without a reason, only cancel resting entries). A failure is reported, never raised:
        one symbol's broken exit must not stop the other symbol's, nor the next cycle."""
        try:
            if reason is None:
                session.cancel_entries()
            else:
                session.flatten(reason, now_ms=int(self.clock() * 1000))
            self.exit_errors.pop(symbol, None)
        except Exception as exc:
            what = "flatten" if reason else "entry cancellation"
            issue = Issue(K.FLATTEN_FAILED if reason else K.CANCEL_FAILED, symbol,
                          f"{symbol}: {what} failed: {type(exc).__name__}: {exc}")
            self.exit_errors[symbol] = issue
            write_issue(self.journal, symbol, issue)
            logger.exception("%s exit failed", symbol)


__all__ = ["EQUITY_SECONDS", "ExitSupervisor", "RECONCILE_SECONDS", "RELEASE_COMMAND"]
