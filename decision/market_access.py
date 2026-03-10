from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any, Optional


def get_decimal_attr(obj: Any, *fields: str) -> Optional[Decimal]:
    for field in fields:
        float_field = f"{field}_f"
        float_value = getattr(obj, float_field, None)
        if float_value is not None:
            try:
                return Decimal(str(float_value))
            except (InvalidOperation, ValueError, TypeError):
                continue
        raw = getattr(obj, field, None)
        if raw is not None:
            try:
                return Decimal(str(raw))
            except (InvalidOperation, ValueError, TypeError):
                continue
    return None


def get_float_attr(obj: Any, *fields: str) -> Optional[float]:
    # Fast path: try _f (float) fields directly to avoid Decimal round-trip
    for field in fields:
        float_field = f"{field}_f"
        float_value = getattr(obj, float_field, None)
        if float_value is not None:
            return float(float_value)
    # Fallback: try raw fields
    for field in fields:
        raw = getattr(obj, field, None)
        if raw is not None:
            try:
                return float(raw)
            except (TypeError, ValueError):
                continue
    return None
