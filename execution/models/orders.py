# execution/models/orders.py
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import hashlib
import json
from typing import Any, Mapping, Optional, Tuple


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


def ingress_order_dedup_identity(order: Any) -> Tuple[str, str]:
    order_key = getattr(order, "order_key", None)
    if not order_key:
        venue = getattr(order, "venue", None) or "unknown"
        symbol = getattr(order, "symbol", None) or ""
        order_id = getattr(order, "order_id", None) or getattr(order, "id", None)
        client_order_id = getattr(order, "client_order_id", None) or getattr(order, "client_id", None)
        order_key = f"{venue}:{symbol}:order:{order_id or client_order_id}"

    payload_digest = getattr(order, "payload_digest", None)
    if not payload_digest:
        payload_digest = _stable_hash(
            {
                "symbol": getattr(order, "symbol", None) or "",
                "order_id": getattr(order, "order_id", None) or getattr(order, "id", None) or "",
                "client_order_id": getattr(order, "client_order_id", None) or getattr(order, "client_id", None) or "",
                "status": getattr(order, "status", None) or "",
                "side": getattr(order, "side", None) or "",
                "order_type": getattr(order, "order_type", None) or getattr(order, "type", None) or "",
                "tif": getattr(order, "tif", None) or "",
                "qty": getattr(order, "qty", None),
                "price": getattr(order, "price", None) if getattr(order, "price", None) is not None else "",
                "filled_qty": getattr(order, "filled_qty", None),
                "avg_price": getattr(order, "avg_price", None) if getattr(order, "avg_price", None) is not None else "",
                "ts_ms": int(getattr(order, "ts_ms", 0) or 0),
            }
        )

    return str(order_key), str(payload_digest)


def _stable_hash(obj: Mapping[str, Any]) -> str:
    return hashlib.sha256(_stable_json(obj).encode("utf-8")).hexdigest()


def _stable_json(obj: Mapping[str, Any]) -> str:
    def default(value: Any) -> Any:
        if isinstance(value, Decimal):
            return str(value)
        return str(value)

    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=default)
