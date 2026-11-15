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

from trading.application.ports import Clock, EventLog, Notifier

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

    def write(self, kind: str, **fields: Any) -> None:
        self.inner.write(kind, **fields)
        if kind == "alert":
            _send(self.notifier, "critical", f"[{self.prefix}] {fields.get('symbol', '')}".rstrip(),
                  str(fields.get("reason", "")))


class IssueAlerts:
    def __init__(self, notifier: Notifier, clock: Clock, *, prefix: str,
                 wait: Callable[[str], float] = lambda issue: ALERT_AFTER_SECONDS) -> None:
        self.notifier, self.clock, self.prefix, self.wait = notifier, clock, prefix, wait
        self._since: dict[tuple[str, str], float] = {}    # (symbol, issue) -> first seen
        self._sent: dict[tuple[str, str], float] = {}     # (symbol, issue) -> last sent

    def observe(self, symbol: str, issues: list[str]) -> None:
        """The symbol's issues of one reconciliation, all of them."""
        now, title = self.clock(), f"[{self.prefix}] {symbol}"
        current = {(symbol, issue) for issue in issues}
        for key in sorted(current):
            since = self._since.setdefault(key, now)
            last = self._sent.get(key)
            if now - since >= self.wait(key[1]) and (last is None or now - last >= REPEAT_SECONDS):
                self._sent[key] = now
                _send(self.notifier, "warning", title, f"{key[1]} (for {int((now - since) // 60)} min)")
        for key in [k for k in self._since if k[0] == symbol and k not in current]:
            del self._since[key]
            if self._sent.pop(key, None) is not None:
                _send(self.notifier, "info", title, f"resolved: {key[1]}")


__all__ = ["ALERT_AFTER_SECONDS", "IssueAlerts", "NotifyingEventLog", "REPEAT_SECONDS"]
