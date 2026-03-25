"""execution.observability — Execution monitoring and alerting (Domain 4: ops).

Provides structured observability for the execution pipeline:
  - Alerts: build_execution_alert for Telegram/webhook dispatch
  - AuditLog: append-only structured log of execution events
  - Incidents: reconcile/synthetic-fill/timeout alert builders
  - Metrics: counters, gauges for execution pipeline health
  - Rejections: rejection event -> alert mapping
  - Redaction: scrub sensitive fields (API keys, secrets) from logs
"""
from execution.observability.alerts import build_execution_alert
from execution.observability.audit_log import AuditLog, AuditEntry
from execution.observability.incidents import (
    reconcile_report_to_alert,
    synthetic_fill_to_alert,
    timeout_to_alert,
)
from execution.observability.metrics import ExecutionMetrics, Counter, Gauge
from execution.observability.rejections import rejection_event_to_alert
from execution.observability.redaction import redact_dict, redact_value, redact_url

__all__ = [
    # Alerts
    "build_execution_alert",
    # Audit log
    "AuditLog",
    "AuditEntry",
    # Incident builders
    "reconcile_report_to_alert",
    "synthetic_fill_to_alert",
    "timeout_to_alert",
    # Metrics
    "ExecutionMetrics",
    "Counter",
    "Gauge",
    # Rejection alerts
    "rejection_event_to_alert",
    # Redaction
    "redact_dict",
    "redact_value",
    "redact_url",
]
