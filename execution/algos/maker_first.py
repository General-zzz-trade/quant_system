"""Maker-first execution — place limit orders first, switch to market near deadline.

Strategy:
1. Place limit order at mid + offset (maker side)
2. Monitor fill progress
3. If fill_ratio < threshold near timeout, convert remaining to market order
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from decimal import Decimal
from types import SimpleNamespace
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class MakerFirstConfig:
    """Configuration for maker-first execution."""

    maker_timeout_sec: float = 60.0  # Time to wait for maker fills
    price_offset_bps: float = 1.0  # Offset from mid for limit orders
    min_fill_ratio: float = 0.3  # Min fill before switching to taker
    check_interval_sec: float = 1.0


@dataclass
class MakerFirstOrder:
    """State of a maker-first order."""

    symbol: str
    side: str
    total_qty: Decimal
    filled_qty: Decimal = field(default_factory=lambda: Decimal("0"))
    limit_price: Optional[Decimal] = None
    created_at: float = field(default_factory=time.monotonic)
    is_complete: bool = False

    @property
    def remaining_qty(self) -> Decimal:
        return self.total_qty - self.filled_qty

    @property
    def fill_ratio(self) -> float:
        if self.total_qty == Decimal("0"):
            return 1.0
        return float(self.filled_qty / self.total_qty)


class MakerFirstAlgo:
    """Maker-first execution algorithm.

    Places a limit order on the maker side of the book (buy below mid,
    sell above mid). Near the deadline, any unfilled portion is converted
    to a market order to ensure completion.
    """

    def __init__(
        self,
        submit_fn: Callable[[str, str, Decimal], Optional[Decimal]],
        config: Optional[MakerFirstConfig] = None,
    ) -> None:
        self._submit_fn = submit_fn
        self._config = config or MakerFirstConfig()

    def create(
        self,
        symbol: str,
        side: str,
        qty: Decimal,
        *,
        mid_price: Decimal,
    ) -> MakerFirstOrder:
        """Create a maker-first order with limit price offset from mid."""
        offset = mid_price * Decimal(str(self._config.price_offset_bps)) / Decimal("10000")
        if side == "buy":
            limit_price = mid_price - offset
        else:
            limit_price = mid_price + offset

        order = MakerFirstOrder(
            symbol=symbol,
            side=side,
            total_qty=qty,
            limit_price=limit_price,
        )

        logger.info(
            "MakerFirst created: %s %s %s, limit=%s (mid=%s, offset=%sbps)",
            side, qty, symbol, limit_price, mid_price,
            self._config.price_offset_bps,
        )
        return order

    def tick(self, order: MakerFirstOrder) -> Optional[Any]:
        """Tick the order. Returns fill result or None.

        Near the timeout deadline, switches from maker to taker
        for the remaining quantity.
        """
        if order.is_complete:
            return None

        remaining = order.remaining_qty
        if remaining <= Decimal("0"):
            order.is_complete = True
            return None

        elapsed = time.monotonic() - order.created_at
        timeout = self._config.maker_timeout_sec

        # Near timeout: force market order for remaining
        if elapsed >= timeout * 0.9:
            return self._force_fill(order, remaining, reason="timeout")

        # Below min fill ratio past halfway: switch to taker
        if elapsed >= timeout * 0.5:
            if order.fill_ratio < self._config.min_fill_ratio:
                return self._force_fill(order, remaining, reason="low_fill_ratio")

        # Attempt passive fill via limit
        try:
            fill_price = self._submit_fn(order.symbol, order.side, remaining)
            if fill_price is not None:
                order.filled_qty += remaining
                order.is_complete = True
                logger.info(
                    "MakerFirst passive fill: %s %s @ %s",
                    order.side, remaining, fill_price,
                )
                return SimpleNamespace(fill_price=fill_price, qty=remaining, mode="maker")
        except Exception as exc:
            logger.warning("MakerFirst passive attempt failed: %s", exc)

        return None

    def _force_fill(
        self,
        order: MakerFirstOrder,
        qty: Decimal,
        *,
        reason: str,
    ) -> Optional[Any]:
        """Force a market fill for remaining quantity."""
        try:
            fill_price = self._submit_fn(order.symbol, order.side, qty)
            if fill_price is not None:
                order.filled_qty += qty
                order.is_complete = True
                logger.info(
                    "MakerFirst taker fill (%s): %s %s @ %s",
                    reason, order.side, qty, fill_price,
                )
                return SimpleNamespace(fill_price=fill_price, qty=qty, mode="taker")
        except Exception as exc:
            logger.warning("MakerFirst taker fill failed: %s", exc)
        return None
