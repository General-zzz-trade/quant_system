"""Backward-compatible re-exports from monitoring.alerts package.

Prefer importing from ``monitoring.alerts`` (the package) directly.
"""
from __future__ import annotations

from monitoring.alerts.base import Alert, AlertSink, Severity
from monitoring.alerts.console import ConsoleAlertSink

__all__ = ["Alert", "AlertSink", "ConsoleAlertSink", "Severity"]
