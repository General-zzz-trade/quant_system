"""An event log whose write failures never reach the caller (S3 audit finding 3; tech lead 2026-11-15).

The event log is observability only, never state. A disk that is full or a directory that was removed must not
stop a flatten (the Flattener writes its intent event before it sends) nor end the process. A failed write is
logged, and until a write succeeds again ``failure()`` returns an EVENT_LOG_FAILED issue: it blocks every entry
(no new exposure while the audit trail has a gap); exits, stops and reconciliation go on.
"""
from __future__ import annotations

import logging
from typing import Any

from trading.application.issues import Issue
from trading.application.ports import EventLog
from trading.domain.events import EventKind
from trading.domain.issues import IssueKind

logger = logging.getLogger(__name__)


class GuardedEventLog:
    def __init__(self, inner: EventLog) -> None:
        self.inner = inner
        self._error: str | None = None

    def write(self, kind: EventKind, **fields: Any) -> None:
        try:
            self.inner.write(kind, **fields)
        except Exception as exc:
            text = f"{type(exc).__name__}: {exc}"
            if self._error != text:
                logger.error("Event log not written (%s); every entry is blocked until it works again", text,
                             exc_info=True)
            self._error = text
            return
        if self._error is not None:
            logger.warning("Event log written again; the entry block lifts at the next reconciliation")
        self._error = None

    def failure(self) -> Issue | None:
        if self._error is None:
            return None
        return Issue(IssueKind.EVENT_LOG_FAILED, None, f"event log not writable ({self._error}); every entry blocked")


__all__ = ["GuardedEventLog"]
