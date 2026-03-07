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
    value = get_decimal_attr(obj, *fields)
    if value is None:
        return None
    return float(value)
