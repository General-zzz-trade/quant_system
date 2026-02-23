"""Monitoring primitives: metrics, event logs, alerts, and health checks."""

from .metrics import Counter, Gauge, Timer, MetricsRegistry
from .eventlog import EventLogger
from .alerts.base import Alert, AlertSink, Severity
from .alerts.console import ConsoleAlertSink
from .health import SystemHealthMonitor, HealthConfig, HealthStatus

__all__ = [
    "Counter",
    "Gauge",
    "Timer",
    "MetricsRegistry",
    "EventLogger",
    "Alert",
    "AlertSink",
    "ConsoleAlertSink",
    "Severity",
    "SystemHealthMonitor",
    "HealthConfig",
    "HealthStatus",
]
