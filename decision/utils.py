from __future__ import annotations

from decimal import Decimal
from typing import Any, Mapping, Optional

from _quant_hotpath import rust_stable_hash as _rust_hash


def stable_hash(parts: list[str], *, prefix: str) -> str:
    text = "\x1f".join(parts)
    return f"{prefix}-{_rust_hash(text, 16)}"


def dec_str(x: Any) -> str:
    if isinstance(x, Decimal):
        return format(x, "f")
    return str(x)


def canonical_meta(meta: Optional[Mapping[str, Any]]) -> str:
    if not meta:
        return ""
    # stable key order
    items = [(str(k), str(meta[k])) for k in sorted(meta.keys())]
    return "|".join([f"{k}={v}" for k, v in items])
