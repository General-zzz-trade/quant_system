"""Alert sink protocol and supporting types.

Alert sinks are the delivery mechanism for system alerts — risk events,
health issues, anomalies, and operational notifications.  All sinks
implement the ``AlertSink`` protocol.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Protocol, Sequence


class Severity(str, Enum):
    """Alert severity levels (ascending urgency)."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass(frozen=True, slots=True)
class Alert:
    """Structured alert payload."""
    title: str
    message: str
    severity: Severity = Severity.INFO
    source: str = ""
    ts: Optional[datetime] = None
    meta: Optional[Mapping[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "title": self.title,
            "message": self.message,
            "severity": self.severity.value,
            "source": self.source,
            "ts": (self.ts or datetime.now(timezone.utc)).isoformat(),
        }
        if self.meta:
            d["meta"] = dict(self.meta)
        return d


class AlertSink(Protocol):
    """Protocol for alert delivery backends."""

    def emit(self, alert: Alert) -> None:
        """Deliver an alert to the backend."""
        ...


@dataclass
class CompositeAlertSink:
    """Fan-out sink that delivers alerts to multiple backends."""
    sinks: Sequence[AlertSink] = ()

    def emit(self, alert: Alert) -> None:
        for sink in self.sinks:
            sink.emit(alert)


@dataclass
class DedupAlertSink:
    """Wraps another sink and suppresses duplicate alerts within a window.

    Two alerts are duplicates if they share the same (title, severity).

    Parameters
    ----------
    delegate : AlertSink
        The underlying sink to forward unique alerts to.
    window_seconds : float
        Time window for deduplication (default: 300s = 5 min).
    """
    delegate: AlertSink
    window_seconds: float = 300.0
    _seen: Dict[str, float] = field(default_factory=dict, init=False)

    def emit(self, alert: Alert) -> None:
        key = f"{alert.title}::{alert.severity.value}"
        ts = (alert.ts or datetime.now(timezone.utc)).timestamp()

        last = self._seen.get(key)
        if last is not None and (ts - last) < self.window_seconds:
            return  # suppress duplicate

        self._seen[key] = ts
        self.delegate.emit(alert)
