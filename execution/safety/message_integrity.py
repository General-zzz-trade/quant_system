# execution/safety/message_integrity.py
"""Payload integrity checking — detects data corruption in order/fill messages.

Delegates digest computation to execution.models.digest (which uses Rust
rust_stable_hash under the hood).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


class IntegrityError(RuntimeError):
    pass


def compute_payload_digest(payload: Mapping[str, Any]) -> str:
    """Compute a 16-char SHA-256 digest of the payload via Rust."""
    from execution.models.digest import payload_digest
    return payload_digest(dict(payload), length=16)


@dataclass(frozen=True, slots=True)
class IntegrityCheckResult:
    valid: bool
    expected_digest: str
    actual_digest: str


class IntegrityChecker:
    """Message integrity checker — compares payload_digest with expected digest.

    All SHA-256 computation is delegated to Rust via execution.models.digest.
    """

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
