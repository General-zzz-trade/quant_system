# execution/models/intents.py
"""Execution intent — bridges decision layer OrderSpec to execution layer."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Any, Mapping, Optional


class IntentStatus(str, Enum):
    """执行意图生命周期状态。"""
    PENDING = "pending"           # 等待执行
    SUBMITTED = "submitted"       # 已提交到交易所
    PARTIALLY_DONE = "partial"    # 部分完成
    DONE = "done"                 # 完全完成
    REJECTED = "rejected"         # 被风控/验证拒绝
    CANCELED = "canceled"         # 被取消
    EXPIRED = "expired"           # 超时


@dataclass(frozen=True, slots=True)
class ExecutionIntent:
    """
    执行意图：将决策层的 OrderSpec 转化为执行层可操作的指令。

    一个 intent 可能产生多个 command（例如拆单），
    intent_id 是跨 command 的追踪标识。
    """
    intent_id: str
    strategy_id: str

    venue: str
    symbol: str
    side: str               # "buy" / "sell"
    qty: Decimal
    order_type: str = "limit"
    price: Optional[Decimal] = None
    tif: Optional[str] = None
    reduce_only: bool = False

    status: IntentStatus = IntentStatus.PENDING
    filled_qty: Decimal = Decimal("0")

    # 关联的 command ID 列表
    command_ids: tuple[str, ...] = ()

    # 来源追踪
    origin_order_id: Optional[str] = None   # decision OrderSpec.order_id
    reason: str = ""

    meta: Optional[Mapping[str, Any]] = None

    @property
    def remaining_qty(self) -> Decimal:
        return max(Decimal("0"), self.qty - self.filled_qty)

    @property
    def is_terminal(self) -> bool:
        return self.status in (
            IntentStatus.DONE,
            IntentStatus.REJECTED,
            IntentStatus.CANCELED,
            IntentStatus.EXPIRED,
        )

    def with_status(self, status: IntentStatus) -> ExecutionIntent:
        return ExecutionIntent(
            intent_id=self.intent_id,
            strategy_id=self.strategy_id,
            venue=self.venue,
            symbol=self.symbol,
            side=self.side,
            qty=self.qty,
            order_type=self.order_type,
            price=self.price,
            tif=self.tif,
            reduce_only=self.reduce_only,
            status=status,
            filled_qty=self.filled_qty,
            command_ids=self.command_ids,
            origin_order_id=self.origin_order_id,
            reason=self.reason,
            meta=self.meta,
        )

    def with_fill(self, fill_qty: Decimal) -> ExecutionIntent:
        new_filled = self.filled_qty + fill_qty
        new_status = IntentStatus.DONE if new_filled >= self.qty else IntentStatus.PARTIALLY_DONE
        return ExecutionIntent(
            intent_id=self.intent_id,
            strategy_id=self.strategy_id,
            venue=self.venue,
            symbol=self.symbol,
            side=self.side,
            qty=self.qty,
            order_type=self.order_type,
            price=self.price,
            tif=self.tif,
            reduce_only=self.reduce_only,
            status=new_status,
            filled_qty=new_filled,
            command_ids=self.command_ids,
            origin_order_id=self.origin_order_id,
            reason=self.reason,
            meta=self.meta,
        )
