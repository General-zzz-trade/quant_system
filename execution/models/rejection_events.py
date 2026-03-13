from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from execution.models.rejections import CanonicalRejection, rejection_routing_key


@dataclass(frozen=True, slots=True)
class CanonicalRejectionEvent:
    """Event-like public view for execution rejections, without entering the main event bus."""

    header: Any
    event_type: str

    status: str
    command_id: str
    venue: str
    symbol: str
    reason: str
    reason_family: str
    routing_key: str
    retryable: bool
    deduped: bool = False


def rejection_to_event(rejection: CanonicalRejection) -> CanonicalRejectionEvent:
    return CanonicalRejectionEvent(
        header=SimpleNamespace(
            event_type="EXECUTION_REJECT",
            event_id=None,
            ts=None,
        ),
        event_type="EXECUTION_REJECT",
        status=rejection.status,
        command_id=rejection.command_id,
        venue=rejection.venue,
        symbol=rejection.symbol,
        reason=rejection.reason,
        reason_family=rejection.reason_family,
        routing_key=rejection_routing_key(
            venue=rejection.venue,
            symbol=rejection.symbol,
            status=rejection.status,
            reason_family=rejection.reason_family,
        ),
        retryable=rejection.retryable,
        deduped=rejection.deduped,
    )
