from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from execution.models.acks import CanonicalAck, normalize_ack


@dataclass(frozen=True, slots=True)
class CanonicalRejection:
    """Normalized execution rejection/failure outcome."""

    status: str
    command_id: str
    venue: str
    symbol: str
    reason: str
    retryable: bool
    deduped: bool = False


def ack_to_rejection(ack: CanonicalAck | object, *, default_venue: str = "", default_symbol: str = "") -> Optional[CanonicalRejection]:
    """Convert non-accepted ack outcomes into a stable rejection/failure view."""
    normalized = ack if isinstance(ack, CanonicalAck) else normalize_ack(
        ack,
        default_venue=default_venue,
        default_symbol=default_symbol,
    )
    if normalized.ok:
        return None

    status = normalized.status.upper()
    reason = normalized.error or status.lower()
    retryable = status == "FAILED"

    return CanonicalRejection(
        status=status,
        command_id=normalized.command_id,
        venue=normalized.venue,
        symbol=normalized.symbol,
        reason=reason,
        retryable=retryable,
        deduped=normalized.deduped,
    )
