# execution/adapters/common/idempotency.py
"""Idempotency key generation for adapters — delegates to Rust."""
from __future__ import annotations

from typing import Optional

from _quant_hotpath import rust_stable_hash as _rust_stable_hash


def make_fill_idem_key(
    *,
    venue: str,
    symbol: str,
    fill_id: Optional[str] = None,
    trade_id: Optional[str] = None,
    order_id: Optional[str] = None,
    side: str = "",
    qty: str = "",
    price: str = "",
) -> str:
    """
    Generate fill idempotency key via Rust SHA-256.

    Priority: fill_id > trade_id > field combo hash.
    """
    if fill_id:
        base = f"{venue}|{symbol}|fill|{fill_id}"
    elif trade_id:
        base = f"{venue}|{symbol}|trade|{trade_id}"
    else:
        base = f"{venue}|{symbol}|combo|{order_id}|{side}|{qty}|{price}"
    return str(_rust_stable_hash(base, 24))


def make_order_idem_key(
    *,
    venue: str,
    symbol: str,
    order_id: str,
) -> str:
    """Generate order idempotency key via Rust SHA-256."""
    base = f"{venue}|{symbol}|order|{order_id}"
    return str(_rust_stable_hash(base, 24))
