"""Alert sinks — delivery backends for system alerts."""
from .base import Alert, AlertSink, CompositeAlertSink, DedupAlertSink, Severity
from .console import ConsoleAlertSink
from .factory import build_alert_sink
from .log_sink import LogAlertSink
from .webhook import WebhookAlertSink

__all__ = [
    "Alert",
    "AlertSink",
    "CompositeAlertSink",
    "ConsoleAlertSink",
    "DedupAlertSink",
    "LogAlertSink",
    "Severity",
    "WebhookAlertSink",
    "build_alert_sink",
]
