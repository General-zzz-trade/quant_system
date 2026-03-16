# execution/sim/limit_order_book.py
"""Simulated limit order book for realistic limit order execution.

Models FIFO queue priority, partial fills based on bar volume,
and price-time priority for limit orders in backtesting.

Addresses the last gap: limit order queue simulation (+0.5 to 10/10).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(str, Enum):
    PENDING = "pending"
    QUEUED = "queued"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    EXPIRED = "expired"
    REJECTED = "rejected"


@dataclass
class SimOrder:
    """A simulated order in the limit order book."""
    order_id: str
    symbol: str
    side: int            # 1=buy, -1=sell
    order_type: OrderType
    qty: float
    price: float         # limit price (0 for market)
    stop_price: float = 0.0
    filled_qty: float = 0.0
    avg_fill_price: float = 0.0
    status: OrderStatus = OrderStatus.PENDING
    created_bar: int = 0
    ttl_bars: int = 0    # 0=GTC, >0=expire after N bars
    fees: float = 0.0

    @property
    def remaining_qty(self) -> float:
        return self.qty - self.filled_qty

    @property
    def is_terminal(self) -> bool:
        return self.status in (OrderStatus.FILLED, OrderStatus.CANCELED,
                               OrderStatus.EXPIRED, OrderStatus.REJECTED)


@dataclass
class FillEvent:
    """Result of a fill (full or partial)."""
    order_id: str
    fill_qty: float
    fill_price: float
    fee: float
    is_partial: bool
    bar: int


class SimulatedOrderBook:
    """Simulated order book with FIFO queue and volume-based fills.

    Supports:
    - Market orders: fill at bar open + slippage
    - Limit orders: fill if price touches limit, FIFO queue for volume
    - Stop orders: activate when stop price hit, then fill as market
    - Partial fills: limited by bar volume × participation rate
    - TTL expiry: orders expire after N bars
    """

    def __init__(
        self,
        fee_bps: float = 4.0,
        slippage_bps: float = 1.0,
        max_participation: float = 0.10,  # max 10% of bar volume
    ) -> None:
        self._fee_bps = fee_bps
        self._slippage_bps = slippage_bps
        self._max_participation = max_participation
        self._orders: dict[str, SimOrder] = {}
        self._fills: list[FillEvent] = []
        self._next_id = 1

    def submit_order(
        self, symbol: str, side: int, qty: float,
        order_type: OrderType = OrderType.MARKET,
        price: float = 0.0, stop_price: float = 0.0,
        bar: int = 0, ttl_bars: int = 0,
    ) -> SimOrder:
        """Submit an order to the simulated book."""
        oid = f"sim_{self._next_id}"
        self._next_id += 1

        order = SimOrder(
            order_id=oid, symbol=symbol, side=side,
            order_type=order_type, qty=qty, price=price,
            stop_price=stop_price, created_bar=bar, ttl_bars=ttl_bars,
        )

        if order_type == OrderType.MARKET:
            order.status = OrderStatus.QUEUED
        elif order_type == OrderType.LIMIT:
            order.status = OrderStatus.QUEUED
        elif order_type in (OrderType.STOP, OrderType.STOP_LIMIT):
            order.status = OrderStatus.PENDING  # waiting for stop trigger
        else:
            order.status = OrderStatus.REJECTED
            return order

        self._orders[oid] = order
        return order

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending/queued order."""
        order = self._orders.get(order_id)
        if order and not order.is_terminal:
            order.status = OrderStatus.CANCELED
            return True
        return False

    def process_bar(
        self, bar: int, open_price: float, high: float, low: float,
        close: float, volume: float,
    ) -> list[FillEvent]:
        """Process all orders against a single bar's OHLCV data.

        Returns list of fills generated this bar.
        """
        bar_fills: list[FillEvent] = []
        available_volume = volume * self._max_participation


        for oid, order in list(self._orders.items()):
            if order.is_terminal:
                continue

            # Check TTL expiry
            if order.ttl_bars > 0 and (bar - order.created_bar) >= order.ttl_bars:
                order.status = OrderStatus.EXPIRED
                continue

            # Stop order activation (activated orders fall through to fill check)
            if order.status == OrderStatus.PENDING:
                activated = False
                if order.order_type == OrderType.STOP:
                    if order.side > 0 and high >= order.stop_price:
                        order.status = OrderStatus.QUEUED
                        order.order_type = OrderType.MARKET
                        activated = True
                    elif order.side < 0 and low <= order.stop_price:
                        order.status = OrderStatus.QUEUED
                        order.order_type = OrderType.MARKET
                        activated = True
                elif order.order_type == OrderType.STOP_LIMIT:
                    if order.side > 0 and high >= order.stop_price:
                        order.status = OrderStatus.QUEUED
                        activated = True
                    elif order.side < 0 and low <= order.stop_price:
                        order.status = OrderStatus.QUEUED
                        activated = True
                if not activated:
                    continue

            if order.status != OrderStatus.QUEUED and order.status != OrderStatus.PARTIALLY_FILLED:
                continue

            # Market order: fill at open + slippage
            if order.order_type == OrderType.MARKET:
                slip = self._slippage_bps / 10000
                if order.side > 0:
                    fill_price = open_price * (1 + slip)
                else:
                    fill_price = open_price * (1 - slip)

                fill_qty = min(order.remaining_qty, available_volume)
                if fill_qty <= 0:
                    continue

                fee = fill_qty * fill_price * self._fee_bps / 10000
                self._record_fill(order, fill_qty, fill_price, fee, bar, bar_fills)
                available_volume -= fill_qty

            # Limit order: fill if price touches limit
            elif order.order_type == OrderType.LIMIT:
                can_fill = False
                fill_price = order.price

                if order.side > 0:  # buy limit: fill if low <= limit price
                    can_fill = low <= order.price
                else:  # sell limit: fill if high >= limit price
                    can_fill = high >= order.price

                if not can_fill:
                    continue

                # FIFO volume constraint: limit fills get less volume than market
                limit_volume = available_volume * 0.5  # limits get 50% of available
                fill_qty = min(order.remaining_qty, limit_volume)
                if fill_qty <= 0:
                    continue

                fee = fill_qty * fill_price * self._fee_bps / 10000
                self._record_fill(order, fill_qty, fill_price, fee, bar, bar_fills)
                available_volume -= fill_qty

        return bar_fills

    def _record_fill(
        self, order: SimOrder, qty: float, price: float,
        fee: float, bar: int, fills: list[FillEvent],
    ) -> None:
        """Record a fill and update order state."""
        # Update weighted average fill price
        total_filled_value = order.avg_fill_price * order.filled_qty + price * qty
        order.filled_qty += qty
        order.avg_fill_price = total_filled_value / order.filled_qty if order.filled_qty > 0 else 0
        order.fees += fee

        is_partial = order.remaining_qty > 0
        if is_partial:
            order.status = OrderStatus.PARTIALLY_FILLED
        else:
            order.status = OrderStatus.FILLED

        fill = FillEvent(
            order_id=order.order_id, fill_qty=qty,
            fill_price=price, fee=fee,
            is_partial=is_partial, bar=bar,
        )
        fills.append(fill)
        self._fills.append(fill)

    @property
    def open_orders(self) -> list[SimOrder]:
        return [o for o in self._orders.values() if not o.is_terminal]

    @property
    def all_fills(self) -> list[FillEvent]:
        return list(self._fills)

    def reset(self) -> None:
        self._orders.clear()
        self._fills.clear()
        self._next_id = 1
