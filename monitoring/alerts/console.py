"""Console alert sink — prints structured alerts to stdout."""
from __future__ import annotations

from datetime import datetime, timezone

from .base import Alert


class ConsoleAlertSink:
    """Prints alerts to stdout with timestamp, severity, and source."""

    def emit(self, alert: Alert) -> None:
        ts = (alert.ts or datetime.now(timezone.utc)).strftime("%Y-%m-%d %H:%M:%S")
        sev = alert.severity.value.upper()
        src = f" [{alert.source}]" if alert.source else ""
        meta = ""
        if alert.meta:
            meta = " " + " ".join(f"{k}={v}" for k, v in alert.meta.items())
        print(f"[{ts}] {sev}{src} {alert.title}: {alert.message}{meta}")
