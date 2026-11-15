"""The target runtime's main loop: warm up, poll closed bars, reconcile, act on risk, publish status.

Moved from runner/binance_target.py (migration step 10). Everything it touches is injected: market data,
account, sessions, event log, kill switch, drawdown guard, paper store, status sink and the clock.
"""
from __future__ import annotations

import logging
from decimal import Decimal
from collections.abc import Callable
from typing import Any

from trading.application.alerts import ALERT_AFTER_SECONDS, IssueAlerts
from trading.application.equity import EquitySource
from trading.application.issues import Issue, with_effect
from trading.application.ports import (Clock, DrawdownGuard, EntryPauses, EventLog, ExchangeAccount, KillSwitch,
                                       MarketData, Notifier, PaperStore, StatusSink)
from trading.application.session import FUNDING_INTERVAL, TargetSession
from trading.domain.bars import HOUR, bar_from_kline
from trading.domain.issues import Effect, IssueKind as K
from trading.domain.strategy.overlay import StrategySpec

logger = logging.getLogger("binance_target")

POLL_SECONDS = 5.0
RECONCILE_SECONDS = 5.0
EQUITY_SECONDS = 60.0
FUNDING_WAIT_MS = 2 * 60_000
MARKET_BACKOFF_MAX = 300.0
MARKET_ALERT_AFTER = 5            # failures in a row before an alert (about 75 s with the 5/10/20/40 s backoff)
PAPER_SCHEMA = 1
RELEASE_COMMAND = "python -m trading.infrastructure.entry_pause --state-dir <state dir> release --symbol {symbol} " \
                  "--reason <why>"


class TargetRuntime:
    def __init__(self, *, spec: StrategySpec, mode: str, market: MarketData, sessions: dict[str, TargetSession],
                 journal: EventLog, equity: EquitySource, clock: Clock, status_sink: StatusSink,
                 market_errors: tuple[type[BaseException], ...], account: ExchangeAccount | None = None,
                 final_market_errors: tuple[type[BaseException], ...] = (),
                 kill: KillSwitch | None = None, drawdown: DrawdownGuard | None = None,
                 paper_store: PaperStore | None = None, reset_paper: bool = False,
                 spec_passed: object = None, pauses: EntryPauses | None = None,
                 notifier: Notifier | None = None, alert_prefix: str = "") -> None:
        if mode not in ("paper", "execution"):
            raise ValueError("mode must be paper or execution")
        self.spec, self.mode, self.market, self.account, self.clock = spec, mode, market, account, clock
        self.sessions, self.journal, self.equity = sessions, journal, equity
        self.kill: Any = kill
        self.drawdown: Any = drawdown
        self._status_sink, self._paper_store, self._market_errors = status_sink, paper_store, market_errors
        self._final_market_errors = final_market_errors   # the exchange says retrying will not help (step 14)
        self._market_refused: str | None = None           # set by such an error, cleared by a success
        self._spec_passed = spec_passed
        self._last_equity = self._last_reconcile = 0.0
        self._market_failures, self._market_retry_at = 0, 0.0
        self.reset_paper = reset_paper
        self._saved_cursors: dict[str, int | None] | None = None
        self.pauses = pauses
        self.notifier = notifier
        self.issue_alerts = IssueAlerts(notifier, clock, prefix=alert_prefix, wait=self._alert_wait) \
            if notifier is not None else None
        self._exit_errors: dict[str, Issue] = {}    # symbol -> the last exit failure, until an exit succeeds

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
            self.journal.write("warmup", symbol=symbol, bars=len(session.bars), last_open=session.cursor,
                               resumed_from=resume)
        if self.mode == "paper":
            self.save_paper_state()
        if self.mode == "execution":
            self.refresh_equity()
            self.reconcile()

    def warmup_or_hold(self) -> bool:
        """Warm up. In execution mode a market-data failure does not end the process: the exit side - equity,
        reconciliation, stop upkeep, risk actions - runs, and False asks the caller to try again later
        (step 14). Paper mode raises as before."""
        if self.mode != "execution":
            self.warmup()
            return True
        if self.clock() >= self._market_retry_at:
            for session in self.sessions.values():        # a partial earlier attempt must not leave bars behind
                session.bars.clear()
                session.funding.clear()
                session.cursor = None
            try:
                self.warmup()
                self._market_failures, self._market_refused = 0, None
                return True
            except self._market_errors as exc:
                self._market_failed(exc)
        if self.clock() - self._last_equity >= EQUITY_SECONDS:
            self.refresh_equity()
        self.reconcile()
        self.risk_actions()
        self._status_sink.write(self.status())
        return False

    def refresh_equity(self) -> None:
        try:
            assert self.account is not None
            equity = self.account.get_account_equity()
            self.equity.update(Decimal(str(equity)))
            self.drawdown.observe(float(equity))
        except Exception:
            self.drawdown.invalidate("exchange equity unavailable")
            logger.warning("Equity refresh failed; entries blocked", exc_info=True)
        self._last_equity = self.clock()

    def reconcile(self) -> None:
        for symbol, session in self.sessions.items():
            try:
                session.consume_fills()
                session.executor.notify_status()
                issues = session.executor.reconcile_account(symbol) + session.stops.sync()
            except Exception as exc:
                issues = [Issue(K.RECONCILE_FAILED, symbol, f"reconciliation failed: {type(exc).__name__}: {exc}")]
                logger.exception("%s reconciliation failed; entries blocked", symbol)
            for issue in issues:
                self._write_issue(symbol, issue)
            flatten = with_effect(issues, Effect.FLATTEN_SYMBOL)
            if flatten:
                self._exit(symbol, session, flatten[0].kind.value)
            refused = Issue(K.MARKET_REFUSED, symbol, f"{symbol}: {self._market_refused}") \
                if self._market_refused is not None else None
            if refused is not None:
                self._write_issue(symbol, refused)
            more = [self._save_pause(symbol, session), self._require_stop(symbol, session),
                    self._exit_errors.get(symbol), refused]
            found = issues + [issue for issue in more if issue is not None]
            self._block(session, found)
            if self.issue_alerts is not None:
                self.issue_alerts.observe(symbol, found)
        self._last_reconcile = self.clock()

    def _write_issue(self, symbol: str | None, issue: Issue) -> None:
        self.journal.write("issue", **({"symbol": symbol} if symbol else {}), issue=str(issue),
                           issue_kind=issue.kind.value)

    def _block(self, session: TargetSession, issues: list[Issue]) -> None:
        """Entry blocks from the ADR-0014 table; reductions are never blocked."""
        if with_effect(issues, Effect.BLOCK_ALL_ENTRIES):
            for other in self.sessions.values():
                other.executor.block_entries()
        elif with_effect(issues, Effect.BLOCK_SYMBOL_ENTRIES):
            session.executor.block_entries()

    def _alert_wait(self, issue: Issue) -> float:
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
            self._write_issue(symbol, issue)
            logger.exception("%s entry pause not saved", symbol)
            return issue
        if new:
            self.journal.write("alert", symbol=symbol, reason=f"entries paused: {cause}; a person releases them "
                               f"with: {RELEASE_COMMAND.format(symbol=symbol)}")
            logger.error("%s entries paused: %s", symbol, cause)
        return None

    def _pause_status(self, symbol: str) -> str | None:
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
        self._write_issue(symbol, issue)
        return issue

    def risk_actions(self) -> None:
        kill = self.kill.state()
        close = (kill.active and kill.flatten) or self.drawdown.latched()
        for symbol, session in self.sessions.items():
            if close:
                self._exit(symbol, session, "kill_switch" if kill.active else "drawdown_latched")
            elif kill.active or self._pause_status(symbol) is not None:
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
            self._exit_errors.pop(symbol, None)
        except Exception as exc:
            what = "flatten" if reason else "entry cancellation"
            issue = Issue(K.FLATTEN_FAILED if reason else K.CANCEL_FAILED, symbol,
                          f"{symbol}: {what} failed: {type(exc).__name__}: {exc}")
            self._exit_errors[symbol] = issue
            self._write_issue(symbol, issue)
            logger.exception("%s exit failed", symbol)

    def poll(self) -> int:
        """Process newly closed bars. A market-data failure is logged and retried with backoff instead of
        ending the process, so reconciliation, stops and risk actions keep running in the same cycle."""
        if self.clock() < self._market_retry_at:
            return 0
        try:
            processed = self._poll_market()
        except self._market_errors as exc:
            self._market_failed(exc)
            return 0
        self._market_failures, self._market_refused = 0, None
        return processed

    def _market_failed(self, exc: BaseException) -> None:
        """Count, back off, and tell a person: after MARKET_ALERT_AFTER failures in a row, or at once when the
        exchange calls the error final - entries then stay blocked until market data works again."""
        self._market_failures += 1
        final = isinstance(exc, self._final_market_errors)
        delay = MARKET_BACKOFF_MAX if final else min(POLL_SECONDS * 2 ** (self._market_failures - 1),
                                                      MARKET_BACKOFF_MAX)
        self._market_retry_at = self.clock() + delay
        text = f"{type(exc).__name__}: {exc}"
        self.journal.write("issue", issue=f"market data unavailable: {text}", issue_kind=K.MARKET_DATA.value,
                           failures=self._market_failures, retry_in=delay)
        logger.warning("Market data unavailable (%d in a row); retry in %.0fs: %s", self._market_failures, delay, exc)
        if final and self._market_refused is None:
            self._market_refused = f"market data refused by the exchange ({text}); entries blocked"
            self.journal.write("alert", reason=f"{self._market_refused} until it works again")
        elif not final and self._market_failures == MARKET_ALERT_AFTER:
            self.journal.write("alert", reason=f"market data unavailable {MARKET_ALERT_AFTER} times in a row: {text}")

    def _poll_market(self) -> int:
        processed = 0
        now_ms = int(self.clock() * 1000)
        for symbol, session in self.sessions.items():
            assert session.cursor is not None
            close = session.cursor + 2 * HOUR
            if now_ms < close:
                continue
            missing = (now_ms // HOUR * HOUR - (session.cursor + HOUR)) // HOUR
            for at, rate in self.market.get_funding_history(symbol, limit=3 + max(missing, 0) // 8):
                session.funding[round(at / HOUR) * HOUR] = rate
            settlement = close // FUNDING_INTERVAL * FUNDING_INTERVAL
            if settlement not in session.funding and now_ms - close < FUNDING_WAIT_MS:
                continue
            rows = self.market.get_klines(symbol, "60", int(min(max(missing, 1) + 2, 1500)), now_ms=now_ms)
            new = [bar for bar in map(bar_from_kline, rows) if bar.open_time > session.cursor]
            for k, bar in enumerate(new):
                # Execution decides on the newest bar only: older bars missed during an outage just extend
                # the window. A late newest bar may still reduce; entries are refused as "processed too late"
                # (approved 2026-11-14, ADR-0007). Paper replays every bar, as research does.
                session.process_bar(bar, live=self.mode == "paper" or k == len(new) - 1, now_ms=now_ms)
                processed += 1
        return processed

    # -- paper state -----------------------------------------------------------
    def _load_paper_state(self) -> dict[str, Any]:
        assert self._paper_store is not None
        if self.reset_paper:
            archived = self._paper_store.archive()
            if archived is not None:
                self.journal.write("paper_reset", archived=archived)
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

    # -- status and the cycle ------------------------------------------------------
    def status(self) -> dict[str, Any]:
        out: dict[str, Any] = {"mode": self.mode, "at": self.clock(), "spec_passed_gate": self._spec_passed,
                               "alerts": getattr(self.notifier, "channel", None),
                               "market_failures": self._market_failures, "symbols": {}}
        if self.kill is not None:
            out["kill"] = self.kill.state().__dict__
            out["drawdown_latched"] = self.drawdown.latched()
            out["equity"] = str(self.equity.value)
        for symbol, s in self.sessions.items():
            stops = s.stops
            out["symbols"][symbol] = {
                "last_bar_open": s.cursor, "target": s.last_target[0], "target_ready": s.last_target[1],
                "entry_ready": s.entry_ready, "entry_reason": s.entry_reason,
                "position": _readable(lambda: str(s.position_qty())),
                "sleeve_equity": str(s.sleeve_equity()),
                "protective_stop": _readable(stops.has_active_stop) if stops else None,
                "flatten_failures": s.flattener.failures if s.flattener else None,
                "entry_pause": _readable(lambda: self._pause_status(symbol))}
        return out

    def cycle(self) -> None:
        if self.mode == "execution" and self.clock() - self._last_equity >= EQUITY_SECONDS:
            self.refresh_equity()
        if self.mode == "execution":
            for session in self.sessions.values():       # entries the rate limit refused on an earlier cycle
                session.retry_deferred(int(self.clock() * 1000))
        self.poll()
        if self.mode == "paper":
            self.save_paper_state()
        if self.mode == "execution":
            if self.clock() - self._last_reconcile >= RECONCILE_SECONDS:
                self.reconcile()
            self.risk_actions()
        self._status_sink.write(self.status())


def _readable(read: Callable[[], Any]) -> Any:
    """A status field from a store: "unreadable" instead of an exception. A lost state database must not stop
    the cycle that reports it; reconciliation already blocks entries and raises the issue (ADR-0009)."""
    try:
        return read()
    except Exception as exc:
        return f"unreadable ({type(exc).__name__})"


__all__ = ["POLL_SECONDS", "TargetRuntime"]
