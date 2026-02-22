# execution/sim/paper_broker.py
"""Paper broker — simulated order matching for backtesting."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, List, Mapping, Optional, Sequence

from execution.models.orders import CanonicalOrder
from execution.models.fills import CanonicalFill
from execution.sim.slippage import SlippageModel, NoSlippage
from execution.sim.latency import LatencyModel, FixedLatency
from execution.adapters.common.time import now_ms


@dataclass(frozen=True, slots=True)
class PaperBrokerConfig:
    """纸上交易配置。"""
    initial_balance: Decimal = Decimal("10000")
    quote_asset: str = "USDT"
    maker_fee_bps: Decimal = Decimal("2")     # 0.02%
    taker_fee_bps: Decimal = Decimal("5")     # 0.05%
    fill_ratio: Decimal = Decimal("1")         # 成交比例 (0-1)


class PaperBroker:
    """
    纸上交易经纪商 — 模拟订单撮合。

    用于回测和模拟交易，不与真实交易所交互。
    """

    def __init__(
        self,
        *,
        config: Optional[PaperBrokerConfig] = None,
        slippage: Optional[SlippageModel] = None,
        latency: Optional[LatencyModel] = None,
    ) -> None:
        self._cfg = config or PaperBrokerConfig()
        self._slippage = slippage or NoSlippage()
        self._latency = latency or FixedLatency(0.0)

        self._balance = self._cfg.initial_balance
        self._positions: Dict[str, Decimal] = {}  # symbol → signed qty
        self._orders: Dict[str, CanonicalOrder] = {}
        self._fills: List[CanonicalFill] = []
        self._order_counter = 0

    def submit_order(
        self,
        *,
        symbol: str,
        side: str,
        qty: Decimal,
        price: Decimal,
        order_type: str = "limit",
        client_order_id: str = "",
    ) -> CanonicalOrder:
        """提交模拟订单。"""
        self._order_counter += 1
        order_id = f"paper-{self._order_counter}"
        coid = client_order_id or order_id

        order = CanonicalOrder(
            venue="paper",
            symbol=symbol.upper(),
            order_id=order_id,
            client_order_id=coid,
            status="new",
            side=side,
            order_type=order_type,
            tif="gtc",
            qty=qty,
            price=price,
            ts_ms=now_ms(),
        )
        self._orders[order_id] = order
        return order

    def try_fill(
        self,
        order_id: str,
        *,
        market_price: Decimal,
    ) -> Optional[CanonicalFill]:
        """尝试以市场价成交订单。"""
        order = self._orders.get(order_id)
        if order is None or order.status in ("filled", "canceled"):
            return None

        # 限价单检查
        if order.order_type == "limit":
            if order.side == "buy" and market_price > (order.price or Decimal("0")):
                return None
            if order.side == "sell" and market_price < (order.price or Decimal("inf")):
                return None

        fill_qty = order.qty * self._cfg.fill_ratio
        exec_price = self._slippage.apply(
            price=market_price, side=order.side, qty=fill_qty,
        )

        # 计算手续费
        fee_bps = self._cfg.taker_fee_bps if order.order_type == "market" else self._cfg.maker_fee_bps
        notional = fill_qty * exec_price
        fee = notional * fee_bps / Decimal("10000")

        fill = CanonicalFill(
            venue="paper",
            symbol=order.symbol,
            order_id=order_id,
            trade_id=f"trade-{uuid.uuid4().hex[:8]}",
            fill_id=f"fill-{uuid.uuid4().hex[:8]}",
            side=order.side,
            qty=fill_qty,
            price=exec_price,
            fee=fee,
            fee_asset=self._cfg.quote_asset,
            liquidity="taker" if order.order_type == "market" else "maker",
            ts_ms=now_ms(),
        )
        self._fills.append(fill)

        # 更新仓位
        delta = fill_qty if order.side == "buy" else -fill_qty
        current = self._positions.get(order.symbol, Decimal("0"))
        self._positions[order.symbol] = current + delta

        # 更新余额
        cost = notional if order.side == "buy" else -notional
        self._balance -= cost + fee

        # 更新订单状态
        self._orders[order_id] = CanonicalOrder(
            venue=order.venue, symbol=order.symbol,
            order_id=order.order_id, client_order_id=order.client_order_id,
            status="filled", side=order.side, order_type=order.order_type,
            tif=order.tif, qty=order.qty, price=order.price,
            filled_qty=fill_qty, avg_price=exec_price, ts_ms=now_ms(),
        )

        return fill

    def cancel_order(self, order_id: str) -> bool:
        """撤销模拟订单。"""
        order = self._orders.get(order_id)
        if order is None or order.status in ("filled", "canceled"):
            return False
        self._orders[order_id] = CanonicalOrder(
            venue=order.venue, symbol=order.symbol,
            order_id=order.order_id, client_order_id=order.client_order_id,
            status="canceled", side=order.side, order_type=order.order_type,
            tif=order.tif, qty=order.qty, price=order.price,
            filled_qty=order.filled_qty, avg_price=order.avg_price,
            ts_ms=now_ms(),
        )
        return True

    @property
    def balance(self) -> Decimal:
        return self._balance

    @property
    def positions(self) -> Dict[str, Decimal]:
        return dict(self._positions)

    @property
    def open_orders(self) -> Sequence[CanonicalOrder]:
        return [o for o in self._orders.values() if o.status not in ("filled", "canceled")]

    @property
    def fills(self) -> Sequence[CanonicalFill]:
        return list(self._fills)
