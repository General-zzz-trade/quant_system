from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional


@dataclass(frozen=True, slots=True)
class CanonicalAck:
    """Minimal normalized execution acknowledgement view."""

    status: str
    ok: bool
    command_id: str
    venue: str
    symbol: str
    attempts: int
    deduped: bool = False
    result: Optional[Mapping[str, Any]] = None
    error: Optional[str] = None


def normalize_ack(ack: Any, *, default_venue: str = "", default_symbol: str = "") -> CanonicalAck:
    """Normalize execution bridge acks and test doubles into one stable shape."""
    status = str(getattr(ack, "status", "FAILED")).upper()
    ok_attr = getattr(ack, "ok", None)
    ok = bool(ok_attr) if ok_attr is not None else status == "ACCEPTED"
    if ok and status != "ACCEPTED":
        status = "ACCEPTED"

    result = getattr(ack, "result", None)
    if result is not None and not isinstance(result, Mapping):
        result = dict(result)

    return CanonicalAck(
        status=status,
        ok=ok,
        command_id=str(getattr(ack, "command_id", "")),
        venue=str(getattr(ack, "venue", default_venue) or default_venue),
        symbol=str(getattr(ack, "symbol", default_symbol) or default_symbol),
        attempts=int(getattr(ack, "attempts", 0) or 0),
        deduped=bool(getattr(ack, "deduped", False)),
        result=result,
        error=_normalize_optional_str(getattr(ack, "error", None)),
    )


def _normalize_optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    return str(value)
