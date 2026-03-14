"""TWAP — Time-Weighted Average Price execution algorithm.

Splits a large order into equal-sized child orders spread evenly over time.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class TWAPSlice:
    """Single TWAP child order."""
    slice_idx: int
    qty: Decimal
    scheduled_at: float   # monotonic time
    executed_at: Optional[float] = None
    fill_price: Optional[Decimal] = None
    status: str = "pending"  # pending, executed, failed


@dataclass
class TWAPOrder:
    """TWAP order state."""
    symbol: str
    side: str
    total_qty: Decimal
    n_slices: int
    duration_sec: float
    slices: List[TWAPSlice] = field(default_factory=list)
    start_time: float = 0.0

    @property
    def filled_qty(self) -> Decimal:
        return sum(s.qty for s in self.slices if s.status == "executed")

    @property
    def remaining_qty(self) -> Decimal:
        return self.total_qty - self.filled_qty

    @property
    def is_complete(self) -> bool:
        return all(s.status != "pending" for s in self.slices)

    @property
    def avg_price(self) -> Optional[Decimal]:
        executed = [s for s in self.slices if s.fill_price is not None]
        if not executed:
            return None
        total_notional = sum(s.qty * s.fill_price for s in executed)
        total_qty = sum(s.qty for s in executed)
        return total_notional / total_qty if total_qty else None


class TWAPAlgo:
    """TWAP execution algorithm.

    Usage:
        algo = TWAPAlgo(submit_fn=exchange.submit_order)
        order = algo.create("BTCUSDT", "buy", Decimal("1.0"), n_slices=10, duration_sec=600)
        # Call algo.tick() periodically
        while not order.is_complete:
            algo.tick(order)
    """

    def __init__(
        self,
        submit_fn: Callable[[str, str, Decimal], Optional[Decimal]],
    ) -> None:
        """
        Args:
            submit_fn: Callable(symbol, side, qty) -> fill_price or None.
        """
        self._submit = submit_fn

    def create(
        self,
        symbol: str,
        side: str,
        total_qty: Decimal,
        *,
        n_slices: int = 10,
        duration_sec: float = 600,
    ) -> TWAPOrder:
        """Create a TWAP order plan."""
        slice_qty = total_qty / n_slices
        remainder = total_qty - slice_qty * n_slices
        interval = duration_sec / n_slices
        now = time.monotonic()

        slices = []
        for i in range(n_slices):
            qty = slice_qty + (remainder if i == n_slices - 1 else Decimal("0"))
            slices.append(TWAPSlice(
                slice_idx=i,
                qty=qty,
                scheduled_at=now + i * interval,
            ))

        order = TWAPOrder(
            symbol=symbol,
            side=side,
            total_qty=total_qty,
            n_slices=n_slices,
            duration_sec=duration_sec,
            slices=slices,
            start_time=now,
        )

        logger.info(
            "TWAP created: %s %s %s in %d slices over %ds",
            side, total_qty, symbol, n_slices, duration_sec,
        )
        return order

    def tick(self, order: TWAPOrder) -> Optional[TWAPSlice]:
        """Check if any slice is due and execute it. Returns executed slice or None."""
        now = time.monotonic()

        for i, s in enumerate(order.slices):
            if s.status != "pending":
                continue
            if now < s.scheduled_at:
                continue

            try:
                fill_price = self._submit(order.symbol, order.side, s.qty)
                order.slices[i] = TWAPSlice(
                    slice_idx=s.slice_idx,
                    qty=s.qty,
                    scheduled_at=s.scheduled_at,
                    executed_at=now,
                    fill_price=fill_price,
                    status="executed" if fill_price else "failed",
                )
                logger.info(
                    "TWAP slice %d/%d: %s %s @ %s",
                    s.slice_idx + 1, order.n_slices, order.side, s.qty, fill_price,
                )
                return order.slices[i]
            except Exception as e:
                logger.warning("TWAP slice %d failed: %s", s.slice_idx, e)
                order.slices[i] = TWAPSlice(
                    slice_idx=s.slice_idx,
                    qty=s.qty,
                    scheduled_at=s.scheduled_at,
                    executed_at=now,
                    status="failed",
                )
                return order.slices[i]

        return None
