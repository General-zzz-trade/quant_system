# execution/adapters/binance/dedup_order_keys.py
from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from typing import Any, Dict, Optional


def make_order_key(*, venue: str, symbol: str, order_id: str) -> str:
    return f"{venue}:{symbol}:order:{order_id}"


def _stable_json(obj: Dict[str, Any]) -> str:
    def default(o: Any) -> Any:
        if isinstance(o, Decimal):
            return str(o)
        return str(o)
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=default)


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
    payload = {
        "symbol": symbol,
        "order_id": order_id,
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
    }
    s = _stable_json(payload).encode("utf-8")
    return hashlib.sha256(s).hexdigest()
