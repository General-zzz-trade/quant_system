# execution/adapters/common/hashing.py
"""Hashing utilities — delegates to canonical execution/models/digest.py."""
from __future__ import annotations

from _quant_hotpath import (  # type: ignore[import-untyped]
    rust_hmac_sign,
    rust_hmac_verify,
)

from execution.models.digest import (
    payload_digest,
    stable_hash,
    fill_key,
    order_key,
)


def hmac_sign(secret: str, message: str) -> str:
    """Compute HMAC-SHA256 signature using Rust kernel."""
    return str(rust_hmac_sign(secret, message))


def hmac_verify(secret: str, message: str, signature: str) -> bool:
    """Verify HMAC-SHA256 signature using Rust kernel."""
    return bool(rust_hmac_verify(secret, message, signature))


__all__ = [
    "payload_digest", "stable_hash", "fill_key", "order_key",
    "hmac_sign", "hmac_verify",
]
