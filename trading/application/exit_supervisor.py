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
from trading.domain.symbol_state import ALLOWED, StateInputs, SymbolState, classify

logger = logging.getLogger("binance_target")

RECONCILE_SECONDS = 5.0
ACCOUNT_CHECK_SECONDS = 60.0      # ADR-0011 / decision O: the whole account at most once a minute
EQUITY_SECONDS = 60.0
RELEASE_COMMAND = "python -m trading.infrastructure.entry_pause --state-dir <state dir> release --symbol {symbol} " \
                  "--reason <why>"


class ExitSupervisor:
    def __init__(self, sessions: dict[str, TargetSession], *, spec: StrategySpec, journal: EventLog, clock: Clock,
                 account: ExchangeAccount | None, equity: EquitySource, kill: KillSwitch | None,
                 drawdown: DrawdownGuard | None,
                 pauses: EntryPauses | None, notifier: Notifier | None, alert_prefix: str,
                 market_refused: Callable[[], str | None], decision_error: Callable[[str], Issue | None],
                 cycle_errors: Callable[[], list[Issue]], warmed: Callable[[], bool] = lambda: True,
                 account_check: Callable[[], list[Issue]] | None = None,
                 exit_observer: Callable[[str, str, str], None] | None = None) -> None:
        self.sessions, self.spec, self.journal, self.clock = sessions, spec, journal, clock
        self.account, self.equity, self.pauses = account, equity, pauses
        self.kill: Any = kill               # both modes run the exit side (S10: paper takes the execution path)
        self.drawdown: Any = drawdown
        self._market_refused, self._decision_error, self._cycle_errors = market_refused, decision_error, cycle_errors
        self._warmed = warmed                       # the whole warm-up succeeded (S3 audit finding 6)
        # ADR-0011: the start-up check (ADR-0010) passed before the runtime was built; that is the first result.
        self._account_check = account_check
        # Told (symbol, action, reason) once when a risk exit of a symbol starts or changes its cause: paper writes
        # it as an intended divergence from research (S10 audit M1). A pause is reported where it is set.
        self._exit_observer = exit_observer
        self._risk_exits: dict[str, tuple[str, str] | None] = {}
        self.account_findings: list[Issue] = []
        self.last_account_check = clock()
        self.last_equity = self.last_reconcile = 0.0
        self.exit_errors: dict[str, Issue] = {}     # symbol -> the last exit failure, until an exit succeeds
        self.last_issues: dict[str, list[Issue]] = {}   # symbol -> the issues of the last reconciliation
        self._forbidden: dict[str, str] = {}        # symbol -> the last forbidden transition alerted
        self.issue_alerts = IssueAlerts(notifier, clock, prefix=alert_prefix, wait=self.alert_wait) \
            if notifier is not None else None

    def run(self) -> None:
        """The exit side of one cycle: reconcile when due, then the risk actions. The symbol states are set before
        the risk actions (a flatten about to run shows as FLATTENING) and again after them."""
        self.check_account_if_due()
        if self.clock() - self.last_reconcile >= RECONCILE_SECONDS:
            self.reconcile()
        self.update_states()
        self.risk_actions()
        self.update_states()

    def settle(self) -> None:
        """``ModePolicy.settle_new_bar_before_decision`` (paper): after the venue matched a new bar and before the
        sessions decide on it - consume its fills, settle the orders it filled or expired, reconcile, protect and
        pause every symbol, and set the states - so a decision sees what research sees after the bar."""
        self.reconcile()
        self.update_states()

    def check_account_if_due(self) -> None:
        """ADR-0011: the whole account, with the start-up check's function, at most every 60 s. The result holds
        until the next check and is applied at every reconciliation; an account that cannot be read is a finding."""
        if self._account_check is None or self.clock() - self.last_account_check < ACCOUNT_CHECK_SECONDS:
            return
        self.last_account_check = self.clock()
        try:
            findings = list(self._account_check())
        except Exception as exc:
            findings = [Issue(K.ACCOUNT_UNREADABLE, None, f"account check failed ({type(exc).__name__}: {exc})")]
        if findings != self.account_findings:
            for issue in findings:
                write_issue(self.journal, issue.symbol, issue)
            if findings:
                logger.error("Account check: %s; every entry is blocked", "; ".join(findings))
        self.account_findings = findings
        if with_effect(findings, Effect.BLOCK_ALL_ENTRIES):
            for session in self.sessions.values():
                session.executor.block_entries()

    def update_states(self) -> None:
        """ADR-0014 point 3: each symbol's state from the last reconciliation's issues, the kill switch, the
        drawdown guard and its pause. A change is an event; a transition outside the table is BLOCKED and
        alerted once."""
        kill = self.kill.state()
        latched = bool(self.drawdown.latched())
        halted = bool(kill.active) or latched
        closing = bool(kill.active and kill.flatten) or latched
        equity_unknown = not self.drawdown.check()[0]   # the gate refuses too (S3 audit finding 8)
        warmed = self._warmed()
        for symbol, session in self.sessions.items():
            issues = self.last_issues.get(symbol, [])
            inputs = StateInputs(
                warmed=warmed and session.cursor is not None and len(session.bars) >= self.spec.window_hours(),
                flattening=(closing or bool(with_effect(issues, Effect.FLATTEN_SYMBOL))) and self._holds(session),
                halted=halted, paused=self.pause_status(symbol) is not None,
                blocked=equity_unknown or bool(with_effect(issues, Effect.BLOCK_SYMBOL_ENTRIES) or
                                               with_effect(issues, Effect.BLOCK_ALL_ENTRIES)))
            previous, state = session.state, classify(inputs)
            if (previous, state) not in ALLOWED:
                # Alerted once per state it is kept out of: BLOCKED -> WARMING after READY -> WARMING is the same.
                if self._forbidden.get(symbol) != state.value:
                    text = f"{symbol}: state {previous.value} -> {state.value} is not allowed; entries blocked"
                    self.journal.write(EventKind.ALERT, symbol=symbol, reason=text)
                    logger.error(text)
                self._forbidden[symbol] = state.value
                state = SymbolState.BLOCKED
            else:
                self._forbidden.pop(symbol, None)
            if state is not previous:
                self.journal.write(EventKind.SYMBOL_STATE, symbol=symbol, previous=previous.value, state=state.value)
                session.state = state

    @staticmethod
    def _holds(session: TargetSession) -> bool:
        try:
            return session.position_qty() != 0
        except Exception:
            return True                             # unknown: still flattening

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
        cycle_errors = self._cycle_errors() + self.account_findings      # runtime-wide, applied every time
        found_of: dict[str, list[Issue]] = {}
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
            found_of[symbol] = issues + [issue for issue in more if issue is not None]
        # Blocks are applied once every symbol is reconciled: each symbol's reconciliation resets its own entry
        # readiness, so a block-all found on one symbol must come after the others' (S3 audit finding 2).
        every = with_effect([i for found in found_of.values() for i in found] + cycle_errors, Effect.BLOCK_ALL_ENTRIES)
        for symbol, session in self.sessions.items():
            found = found_of[symbol]
            self.last_issues[symbol] = found + [i for i in every if i not in found]
            self._block(session, self.last_issues[symbol])
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
            cancelling = with_effect(self.last_issues.get(symbol, []), Effect.CANCEL_RESTING_ENTRIES)
            exit_: tuple[str, str] | None = None
            if close:
                exit_ = ("flatten", "kill_switch" if kill.active else "drawdown_latched")
                self._exit(symbol, session, exit_[1])
            elif kill.active or self.pause_status(symbol) is not None or cancelling:
                # New exposure is blocked, so a resting entry must not fill either (audit after step 12b; a foreign
                # holding or an unreadable account likewise, ADR-0011).
                if kill.active:
                    exit_ = ("cancel_entries", "kill_switch")
                elif cancelling:
                    exit_ = ("cancel_entries", cancelling[0].kind.value)
                self._exit(symbol, session, None)
            self._observe_exit(symbol, exit_)

    def _observe_exit(self, symbol: str, exit_: tuple[str, str] | None) -> None:
        if exit_ is not None and self._risk_exits.get(symbol) != exit_ and self._exit_observer is not None:
            self._exit_observer(symbol, *exit_)
        self._risk_exits[symbol] = exit_

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


__all__ = ["ACCOUNT_CHECK_SECONDS", "EQUITY_SECONDS", "ExitSupervisor", "RECONCILE_SECONDS", "RELEASE_COMMAND"]
