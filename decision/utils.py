from __future__ import annotations

import hashlib
from decimal import Decimal
from typing import Any, Mapping, Optional


def stable_hash(parts: list[str], *, prefix: str) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\x1f")
    return f"{prefix}-{h.hexdigest()[:16]}"


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
