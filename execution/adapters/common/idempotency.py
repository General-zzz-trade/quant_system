# execution/adapters/common/idempotency.py
"""Idempotency key generation for adapters."""
from __future__ import annotations

import hashlib
from typing import Optional


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
    生成成交幂等键。

    优先使用 fill_id，其次 trade_id，最后退化为字段组合 hash。
    """
    if fill_id:
        base = f"{venue}|{symbol}|fill|{fill_id}"
    elif trade_id:
        base = f"{venue}|{symbol}|trade|{trade_id}"
    else:
        base = f"{venue}|{symbol}|combo|{order_id}|{side}|{qty}|{price}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()[:24]


def make_order_idem_key(
    *,
    venue: str,
    symbol: str,
    order_id: str,
) -> str:
    """生成订单幂等键。"""
    base = f"{venue}|{symbol}|order|{order_id}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()[:24]
