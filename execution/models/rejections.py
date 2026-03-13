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
    reason_family: str
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
    reason_family = classify_rejection_reason(
        reason,
        status=status,
        retryable=retryable,
        deduped=normalized.deduped,
    )

    return CanonicalRejection(
        status=status,
        command_id=normalized.command_id,
        venue=normalized.venue,
        symbol=normalized.symbol,
        reason=reason,
        reason_family=reason_family,
        retryable=retryable,
        deduped=normalized.deduped,
    )


def classify_rejection_reason(
    reason: str,
    *,
    status: str,
    retryable: bool,
    deduped: bool,
) -> str:
    """Bucket raw rejection strings into a stable family for routing/dedup."""
    if deduped:
        return "deduped"

    text = str(reason or "").strip().lower()
    if "timeout" in text:
        return "timeout"
    if "rate limit" in text or "too many requests" in text or "429" in text:
        return "rate_limit"
    if "insufficient" in text and ("balance" in text or "margin" in text):
        return "balance"
    if (
        "reduce_only" in text
        or "position side" in text
        or "risk" in text
        or "max leverage" in text
    ):
        return "risk_rule"
    if (
        "invalid" in text
        or "quantity" in text
        or "qty" in text
        or "precision" in text
        or "notional" in text
        or "price" in text
        or "post only" in text
        or "bad request" in text
    ):
        return "validation"
    if retryable:
        return "transient"
    if status.upper() == "REJECTED":
        return "venue_reject"
    return "unknown"


def rejection_routing_key(
    *,
    venue: str,
    symbol: str,
    status: str,
    reason_family: str,
) -> str:
    """Return the stable routing key for rejection alerts and timelines."""
    normalized_venue = str(venue or "unknown")
    normalized_symbol = str(symbol or "unknown")
    normalized_status = str(status or "unknown").strip().lower() or "unknown"
    normalized_family = str(reason_family or "unknown").strip().lower() or "unknown"
    return f"{normalized_venue}:{normalized_symbol}:{normalized_status}:{normalized_family}"
