# execution/models/digest.py
"""Unified hashing and dedup key generation — single source of truth.

ALL payload digest, stable hash, fill key, and order key computation
MUST go through this module. No other file should import hashlib for
dedup/integrity purposes.

Replaces 5 scattered implementations:
- execution/models/orders.py _stable_hash/_stable_json
- execution/models/fill_events.py _stable_hash
- execution/adapters/binance/dedup_keys.py
- execution/adapters/binance/dedup_order_keys.py
- execution/adapters/polymarket/mapper.py _stable_hash
- execution/safety/message_integrity.py compute_payload_digest
- execution/adapters/common/hashing.py payload_digest
"""
from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from typing import Any, Mapping, Optional


def _stable_json(obj: Mapping[str, Any]) -> str:
    """Deterministic JSON serialization. Decimal → str, sorted keys, no spaces."""
    def _default(value: Any) -> Any:
        if isinstance(value, Decimal):
            return str(value)
        return str(value)

    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, default=_default,
    )


def stable_hash(obj: Mapping[str, Any]) -> str:
    """Full SHA-256 hex digest of a dict (deterministic)."""
    return hashlib.sha256(_stable_json(obj).encode("utf-8")).hexdigest()


def payload_digest(obj: Mapping[str, Any], *, length: int = 0) -> str:
    """SHA-256 digest, optionally truncated.

    length=0 (default): full 64-char hex digest.
    length=16: 16-char truncated digest (legacy message_integrity compat).
    """
    h = stable_hash(obj)
    return h[:length] if length > 0 else h


# ── Key builders ──────────────────────────────────────────────


def fill_key(*, venue: str, symbol: str, trade_id: str) -> str:
    """Dedup key for fills. Format: venue:symbol:trade_id."""
    return f"{venue}:{symbol}:{trade_id}"


def order_key(*, venue: str, symbol: str, order_id: str) -> str:
    """Dedup key for orders. Format: venue:symbol:order:order_id."""
    return f"{venue}:{symbol}:order:{order_id}"


# ── Typed digest builders ────────────────────────────────────


def fill_digest(
    *,
    symbol: str,
    order_id: str,
    trade_id: str,
    side: str,
    qty: Decimal,
    price: Decimal,
    fee: Decimal,
    fee_asset: Optional[str] = None,
    ts_ms: int = 0,
) -> str:
    """Compute payload digest for a fill (full hex)."""
    return stable_hash({
        "symbol": symbol,
        "order_id": str(order_id),
        "trade_id": str(trade_id),
        "side": side,
        "qty": qty,
        "price": price,
        "fee": fee,
        "fee_asset": fee_asset or "",
        "ts_ms": int(ts_ms),
    })


def order_digest(
    *,
    symbol: str,
    order_id: str,
    client_order_id: Optional[str] = None,
    status: str = "",
    side: str = "",
    order_type: str = "",
    tif: Optional[str] = None,
    qty: Decimal = Decimal("0"),
    price: Optional[Decimal] = None,
    filled_qty: Decimal = Decimal("0"),
    avg_price: Optional[Decimal] = None,
    ts_ms: int = 0,
) -> str:
    """Compute payload digest for an order update (full hex)."""
    return stable_hash({
        "symbol": symbol,
        "order_id": str(order_id),
        "client_order_id": client_order_id or "",
        "status": status,
        "side": side,
        "order_type": order_type,
        "tif": tif or "",
        "qty": qty,
        "price": price if price is not None else "",
        "filled_qty": filled_qty,
        "avg_price": avg_price if avg_price is not None else "",
        "ts_ms": int(ts_ms),
    })
