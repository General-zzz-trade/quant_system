"""Getting problems to a person (migration step 12, ADR-0005 consequences).

- ``NotifyingEventLog``: every ``alert`` row of the event log is also sent at once (critical). Alerts are
  written for things a person must act on: a flatten that failed ten times, an entry pause.
- ``IssueAlerts``: reconciliation issues are sent only once they have lasted ``ALERT_AFTER_SECONDS``
  (a passing one is noise), repeated every ``REPEAT_SECONDS`` while they last, and their end is reported.

A notifier that fails is logged and ignored: alerting never stops trading or an exit.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from trading.application.issues import Issue
from trading.application.ports import Clock, EventLog, Notifier
from trading.domain.events import EventKind
from trading.domain.issues import IssueKind

logger = logging.getLogger(__name__)

ALERT_AFTER_SECONDS = 120.0
REPEAT_SECONDS = 1800.0


def _send(notifier: Notifier, level: str, title: str, detail: str) -> None:
    try:
        notifier.alert(level, title, detail)
    except Exception:
        logger.warning("Alert not delivered: %s", title, exc_info=True)


class NotifyingEventLog:
    def __init__(self, inner: EventLog, notifier: Notifier, *, prefix: str) -> None:
        self.inner, self.notifier, self.prefix = inner, notifier, prefix

    def write(self, kind: EventKind, **fields: Any) -> None:
        self.inner.write(kind, **fields)
        if kind is EventKind.ALERT:
            _send(self.notifier, "critical", f"[{self.prefix}] {fields.get('symbol', '')}".rstrip(),
                  str(fields.get("reason", "")))


class IssueAlerts:
    """Keyed by symbol and issue kind (ADR-0014): a problem whose wording changes while it lasts (a ledger
    quantity, an order id) is still the same problem. The message shows its latest wording."""

    def __init__(self, notifier: Notifier, clock: Clock, *, prefix: str,
                 wait: Callable[[Issue], float] = lambda issue: ALERT_AFTER_SECONDS) -> None:
        self.notifier, self.clock, self.prefix, self.wait = notifier, clock, prefix, wait
        self._since: dict[tuple[str, IssueKind], float] = {}    # (symbol, kind) -> first seen
        self._sent: dict[tuple[str, IssueKind], float] = {}     # (symbol, kind) -> last sent
        self._text: dict[tuple[str, IssueKind], str] = {}       # (symbol, kind) -> latest wording

    def observe(self, symbol: str, issues: list[Issue]) -> None:
        """The symbol's issues of one reconciliation, all of them."""
        now, title = self.clock(), f"[{self.prefix}] {symbol}"
        current: dict[tuple[str, IssueKind], list[Issue]] = {}
        for issue in issues:
            current.setdefault((symbol, issue.kind), []).append(issue)
        for key in sorted(current, key=lambda k: k[1].value):
            text = self._text[key] = "; ".join(sorted({str(issue) for issue in current[key]}))
            since = self._since.setdefault(key, now)
            last = self._sent.get(key)
            wait = max(self.wait(issue) for issue in current[key])
            if now - since >= wait and (last is None or now - last >= REPEAT_SECONDS):
                self._sent[key] = now
                _send(self.notifier, "warning", title, f"{text} (for {int((now - since) // 60)} min)")
        for key in [k for k in self._since if k[0] == symbol and k not in current]:
            del self._since[key]
            text = self._text.pop(key, "")
            if self._sent.pop(key, None) is not None:
                _send(self.notifier, "info", title, f"resolved: {text}")


__all__ = ["ALERT_AFTER_SECONDS", "IssueAlerts", "NotifyingEventLog", "REPEAT_SECONDS"]
