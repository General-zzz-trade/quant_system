from __future__ import annotations

from monitoring.alerts.base import Alert, Severity
from execution.observability.alerts import build_execution_alert


def rejection_event_to_alert(event: object, *, source: str = "execution:rejection") -> Alert:
    status = str(getattr(event, "status", "")).upper()
    retryable = bool(getattr(event, "retryable", False))
    symbol = str(getattr(event, "symbol", ""))
    venue = str(getattr(event, "venue", ""))
    reason = str(getattr(event, "reason", ""))
    command_id = str(getattr(event, "command_id", ""))

    return build_execution_alert(
        title=f"execution-{status.lower()}",
        message=f"{symbol} {status} on {venue}: {reason}",
        severity=Severity.WARNING if retryable else Severity.ERROR,
        source=source,
        category="execution_rejection",
        routing_key=f"{venue}:{symbol}:{status}",
        meta={
            "event_type": getattr(event, "event_type", "EXECUTION_REJECT"),
            "status": status,
            "symbol": symbol,
            "venue": venue,
            "reason": reason,
            "command_id": command_id,
            "retryable": retryable,
            "deduped": bool(getattr(event, "deduped", False)),
        },
    )
