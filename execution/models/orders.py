# execution/models/orders.py
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Mapping, Optional


@dataclass(frozen=True, slots=True)
class CanonicalOrder:
    """
    CanonicalOrder：系统内唯一标准的“订单事实快照”
    - order_key / payload_digest 用于幂等与数据损坏检测（同 key 不同 payload => fail-fast）
    """
    venue: str
    symbol: str

    order_id: str
    client_order_id: Optional[str]

    status: str            # new/partially_filled/filled/canceled/rejected/expired
    side: str              # buy/sell
    order_type: str        # limit/market/stop/... (lower)
    tif: Optional[str]     # gtc/ioc/fok/post_only/... (lower or None)

    qty: Decimal
    price: Optional[Decimal] = None

    filled_qty: Decimal = Decimal("0")
    avg_price: Optional[Decimal] = None

    ts_ms: int = 0

    order_key: str = ""
    payload_digest: str = ""

    raw: Optional[Mapping[str, Any]] = None
