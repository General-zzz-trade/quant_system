# execution/safety/duplicate_guard.py
"""Idempotency guard — detects duplicate fills/orders and payload corruption."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from threading import RLock
from time import monotonic
from typing import Any, Dict, Mapping


class DuplicateError(RuntimeError):
    pass


class PayloadCorruptionError(RuntimeError):
    pass


def compute_digest(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(dict(payload), sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


@dataclass
class _Entry:
    digest: str
    ts: float


class DuplicateGuard:
    """
    幂等性防护

    1. 新 ID → 记录 digest，返回 True（应处理）
    2. 同 ID + 同 digest → 返回 False（安全跳过）
    3. 同 ID + 不同 digest → 抛出 PayloadCorruptionError
    """

    def __init__(self, *, ttl_seconds: float = 86400.0, max_size: int = 500_000) -> None:
        self._lock = RLock()
        self._seen: Dict[str, _Entry] = {}
        self._ttl = ttl_seconds
        self._max_size = max_size
        self._last_prune: float = monotonic()

    def check(self, *, key: str, payload: Mapping[str, Any]) -> bool:
        digest = compute_digest(payload)
        now = monotonic()
        with self._lock:
            self._maybe_prune(now)
            existing = self._seen.get(key)
            if existing is not None:
                if existing.digest == digest:
                    return False
                raise PayloadCorruptionError(
                    f"key={key!r} payload mismatch (old={existing.digest}, new={digest})"
                )
            self._seen[key] = _Entry(digest=digest, ts=now)
            return True

    def contains(self, key: str) -> bool:
        with self._lock:
            return key in self._seen

    def size(self) -> int:
        with self._lock:
            return len(self._seen)

    def clear(self) -> None:
        with self._lock:
            self._seen.clear()

    def _maybe_prune(self, now: float) -> None:
        if (now - self._last_prune < 60.0) and len(self._seen) < self._max_size:
            return
        cutoff = now - self._ttl
        self._seen = {k: v for k, v in self._seen.items() if v.ts > cutoff}
        self._last_prune = now
