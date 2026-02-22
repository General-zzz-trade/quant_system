# execution/adapters/binance/dedup_keys.py
from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from typing import Any, Dict, Optional


def make_fill_id(*, venue: str, symbol: str, trade_id: str) -> str:
    # 交易所内 trade_id 唯一，但为多交易所/多品种隔离加入 namespace
    return f"{venue}:{symbol}:{trade_id}"


def _stable_json(obj: Dict[str, Any]) -> str:
    # Decimal -> str，确保 hash 稳定
    def default(o: Any) -> Any:
        if isinstance(o, Decimal):
            return str(o)
        return str(o)

    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=default)


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
    payload = {
        "symbol": symbol,
        "order_id": str(order_id),
        "trade_id": str(trade_id),
        "side": side,
        "qty": qty,
        "price": price,
        "fee": fee,
        "fee_asset": fee_asset or "",
        "ts_ms": int(ts_ms),
    }
    s = _stable_json(payload).encode("utf-8")
    return hashlib.sha256(s).hexdigest()
