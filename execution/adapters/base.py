# execution/adapters/base.py
"""Base venue adapter protocol — all venue implementations must satisfy this."""
from __future__ import annotations

from typing import Any, Mapping, Optional, Protocol, Sequence, Tuple

from execution.models.balances import BalanceSnapshot
from execution.models.instruments import InstrumentInfo
from execution.models.positions import VenuePosition
from execution.models.orders import CanonicalOrder
from execution.models.fills import CanonicalFill


class VenueAdapter(Protocol):
    """
    交易所适配器协议。

    所有交易所（Binance、OKX、模拟等）必须实现此接口。
    ExecutionBridge / SimVenueAdapter / BinanceAdapter 都满足此协议。
    """
    venue: str

    def list_instruments(self) -> Tuple[InstrumentInfo, ...]:
        """获取所有可交易品种的元数据。"""
        ...

    def get_balances(self) -> BalanceSnapshot:
        """获取当前账户余额快照。"""
        ...

    def get_positions(self) -> Tuple[VenuePosition, ...]:
        """获取当前持仓列表。"""
        ...

    def get_open_orders(
        self, *, symbol: Optional[str] = None,
    ) -> Tuple[CanonicalOrder, ...]:
        """获取当前活跃订单。"""
        ...

    def get_recent_fills(
        self, *, symbol: Optional[str] = None, since_ms: int = 0,
    ) -> Tuple[CanonicalFill, ...]:
        """获取最近成交记录。"""
        ...
