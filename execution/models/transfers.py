# execution/models/transfers.py
"""Transfer models — margin transfers, funding, withdrawals."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Any, Mapping, Optional


class TransferType(str, Enum):
    """划转类型。"""
    MARGIN_TO_SPOT = "margin_to_spot"
    SPOT_TO_MARGIN = "spot_to_margin"
    CROSS_TO_ISOLATED = "cross_to_isolated"
    ISOLATED_TO_CROSS = "isolated_to_cross"
    FUNDING = "funding"               # 资金费率结算
    INTERNAL = "internal"             # 内部账户间划转


class TransferStatus(str, Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class TransferRequest:
    """划转请求。"""
    transfer_id: str
    venue: str
    asset: str
    amount: Decimal
    transfer_type: TransferType
    from_account: str = ""        # 来源账户标识
    to_account: str = ""          # 目标账户标识
    symbol: Optional[str] = None  # 逐仓划转时需要


@dataclass(frozen=True, slots=True)
class TransferResult:
    """划转结果。"""
    transfer_id: str
    status: TransferStatus
    venue: str
    asset: str
    amount: Decimal
    transfer_type: TransferType
    ts_ms: int = 0
    error: Optional[str] = None
    raw: Optional[Mapping[str, Any]] = None

    @property
    def ok(self) -> bool:
        return self.status == TransferStatus.COMPLETED


@dataclass(frozen=True, slots=True)
class FundingRecord:
    """资金费率记录。"""
    venue: str
    symbol: str
    funding_rate: Decimal
    amount: Decimal              # 正=收到，负=支付
    position_qty: Decimal
    ts_ms: int = 0
    raw: Optional[Mapping[str, Any]] = None
