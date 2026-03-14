# execution/sim/venue_emulator.py
"""Venue emulator — simulates venue API for integration testing."""
from __future__ import annotations

from decimal import Decimal
from typing import Any, List, Mapping, Optional

from execution.sim.paper_broker import PaperBroker, PaperBrokerConfig
from execution.sim.slippage import SlippageModel


class VenueEmulator:
    """
    交易所模拟器 — 提供与 VenueClient 协议兼容的接口。

    底层使用 PaperBroker 实现撮合逻辑。
    可用于集成测试和端到端模拟。
    """

    def __init__(
        self,
        *,
        venue: str = "sim",
        config: Optional[PaperBrokerConfig] = None,
        slippage: Optional[SlippageModel] = None,
        auto_fill: bool = True,
        fill_price: Optional[Decimal] = None,
    ) -> None:
        self._venue = venue
        self._broker = PaperBroker(config=config, slippage=slippage)
        self._auto_fill = auto_fill
        self._fill_price = fill_price or Decimal("100")
        self._submitted: List[Mapping[str, Any]] = []

    def submit_order(self, cmd: Any) -> Mapping[str, Any]:
        """提交订单（模拟 VenueClient 协议）。"""
        symbol = str(getattr(cmd, "symbol", "BTCUSDT"))
        side = str(getattr(cmd, "side", "buy"))
        qty = getattr(cmd, "qty", Decimal("1"))
        price = getattr(cmd, "price", None) or self._fill_price

        order = self._broker.submit_order(
            symbol=symbol, side=side, qty=qty, price=price,
        )
        self._submitted.append({"action": "submit", "order_id": order.order_id})

        if self._auto_fill:
            self._broker.try_fill(order.order_id, market_price=self._fill_price)

        return {"order_id": order.order_id, "status": "ACCEPTED"}

    def cancel_order(self, cmd: Any) -> Mapping[str, Any]:
        """撤单（模拟 VenueClient 协议）。"""
        order_id = str(getattr(cmd, "order_id", ""))
        ok = self._broker.cancel_order(order_id)
        return {"order_id": order_id, "status": "CANCELED" if ok else "FAILED"}

    @property
    def broker(self) -> PaperBroker:
        return self._broker

    @property
    def submitted(self) -> List[Mapping[str, Any]]:
        return list(self._submitted)
