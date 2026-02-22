# execution/models/fills.py
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Mapping, Optional


@dataclass(frozen=True, slots=True)
class CanonicalFill:
    """
    CanonicalFill：系统内唯一标准的成交事实

    关键制度：
    - fill_id 必须稳定（同一真实成交重复到达，fill_id 必须相同）
    - payload_digest 用于检测 “同 fill_id 不同 payload” 的数据损坏
    """
    venue: str
    symbol: str

    order_id: str
    trade_id: str
    fill_id: str

    side: str              # "buy" / "sell"
    qty: Decimal           # base qty, >0
    price: Decimal         # >0

    fee: Decimal = Decimal("0")
    fee_asset: Optional[str] = None
    liquidity: Optional[str] = None  # "maker" / "taker" / None

    ts_ms: int = 0         # epoch ms
    payload_digest: str = ""

    raw: Optional[Mapping[str, Any]] = None
