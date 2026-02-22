# execution/safety/message_integrity.py
"""Payload integrity checking — detects data corruption in order/fill messages."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping


class IntegrityError(RuntimeError):
    pass


def compute_payload_digest(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(dict(payload), sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True, slots=True)
class IntegrityCheckResult:
    valid: bool
    expected_digest: str
    actual_digest: str


class IntegrityChecker:
    """消息完整性校验 — 对比 payload_digest 与重新计算的 digest。"""

    @staticmethod
    def verify(*, payload: Mapping[str, Any], expected_digest: str) -> IntegrityCheckResult:
        actual = compute_payload_digest(payload)
        return IntegrityCheckResult(valid=(actual == expected_digest),
            expected_digest=expected_digest, actual_digest=actual)

    @staticmethod
    def verify_or_raise(*, payload: Mapping[str, Any], expected_digest: str) -> None:
        result = IntegrityChecker.verify(payload=payload, expected_digest=expected_digest)
        if not result.valid:
            raise IntegrityError(
                f"digest mismatch: expected={result.expected_digest}, actual={result.actual_digest}")

    @staticmethod
    def stamp(payload: Mapping[str, Any]) -> str:
        return compute_payload_digest(payload)
