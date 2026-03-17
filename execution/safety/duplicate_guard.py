# execution/safety/duplicate_guard.py
"""Idempotency guard — detects duplicate fills/orders and payload corruption."""
from __future__ import annotations

import hashlib
import json
from threading import RLock
from time import monotonic
from typing import Any, Mapping

from _quant_hotpath import RustFillDedupGuard as _RustFillDedupGuard


class DuplicateError(RuntimeError):
    pass


class PayloadCorruptionError(RuntimeError):
    pass


def compute_digest(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(dict(payload), sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


class DuplicateGuard:
    """Rust-accelerated duplicate guard with same API as DuplicateGuard."""

    def __init__(self, *, ttl_seconds: float = 86400.0, max_size: int = 500_000) -> None:
        self._lock = RLock()
        self._inner = _RustFillDedupGuard(ttl_sec=ttl_seconds, max_size=max_size)

    def check(self, *, key: str, payload: Mapping[str, Any]) -> bool:
        digest = compute_digest(payload)
        now = monotonic()
        with self._lock:
            result = self._inner.check(key, digest, now)
        if result == "new":
            return True
        if result == "duplicate":
            return False
        # corrupted:<old>:<new>
        parts = result.split(":", 2)
        raise PayloadCorruptionError(
            f"key={key!r} payload mismatch (old={parts[1]}, new={parts[2]})"
        )

    def contains(self, key: str) -> bool:
        with self._lock:
            return bool(self._inner.contains(key))

    def size(self) -> int:
        with self._lock:
            return len(self._inner)

    def clear(self) -> None:
        with self._lock:
            self._inner.clear()


def make_duplicate_guard(**kwargs: Any) -> DuplicateGuard:
    return DuplicateGuard(**kwargs)
