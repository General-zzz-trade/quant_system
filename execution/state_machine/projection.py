# execution/state_machine/projection.py
"""Project order state from a sequence of events (for replay / reconciliation)."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Mapping, Optional, Sequence

from execution.state_machine.transitions import OrderStatus


@dataclass(frozen=True, slots=True)
class OrderProjection:
    """从事件序列投影出的订单状态快照。"""
    order_id: str
    status: OrderStatus
    qty: Decimal
    filled_qty: Decimal
    avg_price: Optional[Decimal]
    last_ts_ms: int


def project_order(events: Sequence[Mapping[str, Any]]) -> Optional[OrderProjection]:
    """
    从一系列订单事件投影出当前状态。

    events 按时间排序，每个 event 需包含:
    - order_id: str
    - status: str (对应 OrderStatus)
    - filled_qty: str/Decimal (可选)
    - avg_price: str/Decimal (可选)
    - ts_ms: int (可选)
    """
    if not events:
        return None

    order_id = str(events[0].get("order_id", ""))
    status = OrderStatus.PENDING_NEW
    qty = Decimal("0")
    filled_qty = Decimal("0")
    avg_price: Optional[Decimal] = None
    last_ts = 0

    for ev in events:
        raw_status = ev.get("status")
        if raw_status is not None:
            try:
                status = OrderStatus(str(raw_status).lower())
            except ValueError:
                pass

        raw_qty = ev.get("qty")
        if raw_qty is not None:
            try:
                qty = Decimal(str(raw_qty))
            except Exception:
                pass

        raw_fq = ev.get("filled_qty")
        if raw_fq is not None:
            try:
                filled_qty = Decimal(str(raw_fq))
            except Exception:
                pass

        raw_ap = ev.get("avg_price")
        if raw_ap is not None:
            try:
                avg_price = Decimal(str(raw_ap))
            except Exception:
                pass

        raw_ts = ev.get("ts_ms")
        if raw_ts is not None:
            try:
                last_ts = int(raw_ts)
            except Exception:
                pass

    return OrderProjection(
        order_id=order_id,
        status=status,
        qty=qty,
        filled_qty=filled_qty,
        avg_price=avg_price,
        last_ts_ms=last_ts,
    )
