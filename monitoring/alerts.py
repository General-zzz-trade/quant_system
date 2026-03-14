"""Backward-compatible re-exports from monitoring.alerts package.

Prefer importing from ``monitoring.alerts`` (the package) directly.

This standalone module coexists with the ``monitoring/alerts/`` package
directory. It re-exports core alert types so that legacy imports of the form
``from monitoring.alerts import Alert`` continue to work without changes.
"""
from __future__ import annotations

from monitoring.alerts.base import Alert, AlertSink, Severity
from monitoring.alerts.console import ConsoleAlertSink

__all__ = ["Alert", "AlertSink", "ConsoleAlertSink", "Severity"]
