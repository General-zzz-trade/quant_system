# execution/state_machine/transitions.py
"""Order status enum and valid transition rules.

RustOrderTransition is available for Rust-side transition validation.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, FrozenSet, Set

from _quant_hotpath import RustOrderTransition as _RustOrderTransition


class OrderStatus(str, Enum):
    PENDING_NEW = "pending_new"
    NEW = "new"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    PENDING_CANCEL = "pending_cancel"
    CANCELED = "canceled"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass(frozen=True, slots=True)
class Transition:
    from_status: OrderStatus
    to_status: OrderStatus
    ts_ms: int = 0
    reason: str = ""


# 合法状态转换图
VALID_TRANSITIONS: Dict[OrderStatus, Set[OrderStatus]] = {
    OrderStatus.PENDING_NEW: {
        OrderStatus.NEW,
        OrderStatus.REJECTED,
        OrderStatus.PARTIALLY_FILLED,  # 极端情况：还没收到 NEW 就收到 FILL
        OrderStatus.FILLED,
    },
    OrderStatus.NEW: {
        OrderStatus.PARTIALLY_FILLED,
        OrderStatus.FILLED,
        OrderStatus.PENDING_CANCEL,
        OrderStatus.CANCELED,
        OrderStatus.EXPIRED,
    },
    OrderStatus.PARTIALLY_FILLED: {
        OrderStatus.PARTIALLY_FILLED,  # 多次部分成交
        OrderStatus.FILLED,
        OrderStatus.PENDING_CANCEL,
        OrderStatus.CANCELED,
    },
    OrderStatus.PENDING_CANCEL: {
        OrderStatus.CANCELED,
        OrderStatus.FILLED,            # 撤单请求发出但已成交
        OrderStatus.PARTIALLY_FILLED,  # 撤单请求期间又有部分成交
        OrderStatus.REJECTED,          # 撤单被拒
    },
    # 终态：不允许任何后续转换
    OrderStatus.FILLED: set(),
    OrderStatus.CANCELED: set(),
    OrderStatus.REJECTED: set(),
    OrderStatus.EXPIRED: set(),
}

TERMINAL_STATUSES: FrozenSet[OrderStatus] = frozenset({
    OrderStatus.FILLED,
    OrderStatus.CANCELED,
    OrderStatus.REJECTED,
    OrderStatus.EXPIRED,
})


def transition_from_rust(raw: "_RustOrderTransition | object") -> Transition:
    """Convert a RustOrderTransition to a Python Transition."""
    return Transition(
        from_status=OrderStatus(str(getattr(raw, "from_status")).lower()),
        to_status=OrderStatus(str(getattr(raw, "to_status")).lower()),
        ts_ms=int(getattr(raw, "ts_ms", 0)),
        reason=str(getattr(raw, "reason", "")),
    )
