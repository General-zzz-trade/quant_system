"""Alert sinks — delivery backends for system alerts."""
from .base import Alert, AlertSink, CompositeAlertSink, DedupAlertSink, Severity
from .console import ConsoleAlertSink

__all__ = [
    "Alert",
    "AlertSink",
    "CompositeAlertSink",
    "ConsoleAlertSink",
    "DedupAlertSink",
    "Severity",
]
