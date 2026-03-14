# execution/adapters/binance/dedup_order_keys.py
"""Order dedup keys — delegates to unified digest module."""
from __future__ import annotations

from decimal import Decimal
from typing import Optional

from execution.models.digest import order_key as _order_key
from execution.models.digest import order_digest


def make_order_key(*, venue: str, symbol: str, order_id: str) -> str:
    return _order_key(venue=venue, symbol=symbol, order_id=order_id)


def payload_digest_for_order(
    *,
    symbol: str,
    order_id: str,
    client_order_id: Optional[str],
    status: str,
    side: str,
    order_type: str,
    tif: Optional[str],
    qty: Decimal,
    price: Optional[Decimal],
    filled_qty: Decimal,
    avg_price: Optional[Decimal],
    ts_ms: int,
) -> str:
    return order_digest(
        symbol=symbol, order_id=order_id, client_order_id=client_order_id,
        status=status, side=side, order_type=order_type, tif=tif,
        qty=qty, price=price, filled_qty=filled_qty,
        avg_price=avg_price, ts_ms=ts_ms,
    )
