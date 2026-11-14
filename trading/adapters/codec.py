"""JSON encoding shared by the adapters: Decimals are stored as tagged strings so they round-trip exactly."""
from __future__ import annotations

from decimal import Decimal
from typing import Any


def encode(value: Any) -> Any:
    if isinstance(value, Decimal):
        return {"__decimal__": str(value)}
    raise TypeError(f"cannot persist {type(value).__name__}")


def decode(obj: dict[str, Any]) -> Any:
    return Decimal(obj["__decimal__"]) if obj.keys() == {"__decimal__"} else obj


__all__ = ["decode", "encode"]
