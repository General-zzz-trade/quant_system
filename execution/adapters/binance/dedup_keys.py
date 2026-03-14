# execution/adapters/binance/dedup_keys.py
"""Fill dedup keys — delegates to unified digest module."""
from __future__ import annotations

from decimal import Decimal
from typing import Optional

from execution.models.digest import fill_key as _fill_key
from execution.models.digest import fill_digest


def make_fill_id(*, venue: str, symbol: str, trade_id: str) -> str:
    return _fill_key(venue=venue, symbol=symbol, trade_id=trade_id)


def payload_digest_for_fill(
    *,
    symbol: str,
    order_id: str,
    trade_id: str,
    side: str,
    qty: Decimal,
    price: Decimal,
    fee: Decimal,
    fee_asset: Optional[str],
    ts_ms: int,
) -> str:
    return fill_digest(
        symbol=symbol, order_id=order_id, trade_id=trade_id,
        side=side, qty=qty, price=price, fee=fee,
        fee_asset=fee_asset, ts_ms=ts_ms,
    )
