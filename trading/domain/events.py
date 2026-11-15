"""The kinds of event the runtime writes to its event log (ADR-0014 point 2). The values are the words the log
has always used; ``order`` and ``stop`` record every status change of an order or a protective stop (data and
state architecture decision, 2026-11-15)."""
from __future__ import annotations

from enum import Enum


class EventKind(Enum):
    WARMUP = "warmup"
    INTENT = "intent"
    FILL = "fill"
    BLOCKED = "blocked"
    ISSUE = "issue"
    ALERT = "alert"
    PAPER_EQUITY = "paper_equity"
    PAPER_FILL = "paper_fill"
    PAPER_RESET = "paper_reset"
    ORDER = "order"
    STOP = "stop"


__all__ = ["EventKind"]
