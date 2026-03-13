from __future__ import annotations

from typing import Any, Mapping, Optional

from monitoring.alerts.base import Alert, Severity


def build_execution_alert(
    *,
    title: str,
    message: str,
    severity: Severity,
    category: str,
    routing_key: str,
    source: str,
    meta: Optional[Mapping[str, Any]] = None,
) -> Alert:
    payload = {
        "category": category,
        "routing_key": routing_key,
    }
    if meta:
        payload.update(dict(meta))
    return Alert(
        title=title,
        message=message,
        severity=severity,
        source=source,
        meta=payload,
    )
