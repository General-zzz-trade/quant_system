"""Alert sinks — delivery backends for system alerts."""
from .base import Alert, AlertSink, CompositeAlertSink, DedupAlertSink, Severity
from .console import ConsoleAlertSink
from .webhook import WebhookAlertSink

__all__ = [
    "Alert",
    "AlertSink",
    "CompositeAlertSink",
    "ConsoleAlertSink",
    "DedupAlertSink",
    "Severity",
    "WebhookAlertSink",
]
