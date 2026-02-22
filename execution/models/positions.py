# execution/models/positions.py
"""Venue-reported position snapshot (execution layer view)."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Mapping, Optional, Sequence


@dataclass(frozen=True, slots=True)
class VenuePosition:
    """
    交易所报告的仓位快照。

    与 state/position.py 的 PositionState（内部SSOT）不同，
    VenuePosition 是交易所原始上报数据，用于对账。
    """
    venue: str
    symbol: str

    qty: Decimal                   # 有符号：正=多，负=空
    entry_price: Optional[Decimal] = None
    mark_price: Optional[Decimal] = None
    liquidation_price: Optional[Decimal] = None

    unrealized_pnl: Decimal = Decimal("0")
    leverage: Optional[int] = None
    margin_type: Optional[str] = None   # "cross" / "isolated"

    ts_ms: int = 0
    raw: Optional[Mapping[str, Any]] = None

    @property
    def is_long(self) -> bool:
        return self.qty > 0

    @property
    def is_short(self) -> bool:
        return self.qty < 0

    @property
    def is_flat(self) -> bool:
        return self.qty == 0

    @property
    def abs_qty(self) -> Decimal:
        return abs(self.qty)

    @property
    def side(self) -> str:
        if self.qty > 0:
            return "buy"
        elif self.qty < 0:
            return "sell"
        return "flat"


@dataclass(frozen=True, slots=True)
class PositionSnapshot:
    """一次完整的持仓快照（多个品种）。"""
    venue: str
    positions: tuple[VenuePosition, ...]
    ts_ms: int = 0

    def get(self, symbol: str) -> Optional[VenuePosition]:
        s = symbol.upper()
        for p in self.positions:
            if p.symbol == s:
                return p
        return None

    @property
    def symbols(self) -> Sequence[str]:
        return [p.symbol for p in self.positions]

    @property
    def active(self) -> Sequence[VenuePosition]:
        return [p for p in self.positions if not p.is_flat]


# 兼容别名
Position = VenuePosition
