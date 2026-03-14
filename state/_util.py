from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Optional

def to_decimal(x: Any, *, allow_none: bool = False) -> Optional[Decimal]:
    if x is None:
        return None if allow_none else Decimal("0")
    if isinstance(x, Decimal):
        return x
    # bool is int subclass; reject
    if isinstance(x, bool):
        raise TypeError("bool cannot be converted to Decimal")
    try:
        return Decimal(str(x))
    except (InvalidOperation, ValueError, TypeError) as e:
        raise TypeError(f"cannot convert {type(x).__name__} to Decimal: {x!r}") from e

def ensure_utc(ts: Any) -> Optional[datetime]:
    if ts is None:
        return None
    if not isinstance(ts, datetime):
        raise TypeError(f"ts must be datetime, got {type(ts).__name__}")
    if ts.tzinfo is None:
        # treat as UTC to avoid crashing legacy code paths, but prefer tz-aware in production
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)

def norm_event_type(raw: Any) -> str:
    if raw is None:
        return ""
    # Enum-like: .value or .name
    if hasattr(raw, "value"):
        raw = getattr(raw, "value")
    elif hasattr(raw, "name"):
        raw = getattr(raw, "name")
    s = str(raw).strip()
    return s.lower()

def get_header(event: Any) -> Any:
    return getattr(event, "header", None)

def get_event_ts(event: Any) -> Optional[datetime]:
    h = get_header(event)
    ts = getattr(h, "ts", None)
    if ts is None:
        ts = getattr(event, "ts", None)
    return ensure_utc(ts)

def get_event_id(event: Any) -> Optional[str]:
    h = get_header(event)
    eid = getattr(h, "event_id", None)
    if eid is None:
        eid = getattr(event, "event_id", None)
    return str(eid) if eid is not None else None

def get_event_type(event: Any) -> str:
    h = get_header(event)
    et = getattr(h, "event_type", None)
    if et is None:
        et = getattr(event, "event_type", None)
    return norm_event_type(et)

def get_symbol(event: Any, default: str = "") -> str:
    sym = getattr(event, "symbol", None)
    if sym is None:
        # sometimes nested
        bar = getattr(event, "bar", None)
        sym = getattr(bar, "symbol", None) if bar is not None else None
    return str(sym) if sym is not None else default

def signed_qty(qty: Any, side: Any) -> Decimal:
    q = to_decimal(qty)
    s = str(side).strip().lower() if side is not None else ""
    # if qty already signed and side missing, keep as is
    if s in ("buy", "long"):
        return abs(q)
    if s in ("sell", "short"):
        return -abs(q)
    return q
