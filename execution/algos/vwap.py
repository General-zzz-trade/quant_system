"""VWAP — Volume-Weighted Average Price execution algorithm.

Distributes order slices according to historical volume profile,
concentrating execution during high-volume periods.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Callable, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class VWAPSlice:
    """Single VWAP child order."""
    slice_idx: int
    qty: Decimal
    volume_weight: float
    scheduled_at: float
    executed_at: Optional[float] = None
    fill_price: Optional[Decimal] = None
    status: str = "pending"


@dataclass
class VWAPOrder:
    """VWAP order state."""
    symbol: str
    side: str
    total_qty: Decimal
    slices: List[VWAPSlice] = field(default_factory=list)

    @property
    def filled_qty(self) -> Decimal:
        return sum(s.qty for s in self.slices if s.status == "executed")

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


def _default_volume_profile(n_slices: int) -> List[float]:
    """Default U-shaped volume profile (high at open/close, low mid-day)."""
    import math
    profile = []
    for i in range(n_slices):
        x = (i / max(n_slices - 1, 1)) * 2 - 1  # -1 to 1
        weight = 1.0 + 0.5 * x * x  # U-shape
        profile.append(weight)
    total = sum(profile)
    return [w / total for w in profile]


class VWAPAlgo:
    """VWAP execution algorithm."""

    def __init__(
        self,
        submit_fn: Callable[[str, str, Decimal], Optional[Decimal]],
    ) -> None:
        self._submit = submit_fn

    def create(
        self,
        symbol: str,
        side: str,
        total_qty: Decimal,
        *,
        n_slices: int = 10,
        duration_sec: float = 600,
        volume_profile: Optional[Sequence[float]] = None,
    ) -> VWAPOrder:
        """Create a VWAP order plan."""
        profile = list(volume_profile) if volume_profile else _default_volume_profile(n_slices)

        if len(profile) != n_slices:
            raise ValueError(f"volume_profile length {len(profile)} != n_slices {n_slices}")

        # Normalize
        total_w = sum(profile)
        weights = [w / total_w for w in profile]

        now = time.monotonic()
        interval = duration_sec / n_slices

        slices = []
        allocated = Decimal("0")
        for i in range(n_slices):
            if i == n_slices - 1:
                qty = total_qty - allocated
            else:
                qty = (total_qty * Decimal(str(weights[i]))).quantize(Decimal("0.00000001"))
                allocated += qty

            slices.append(VWAPSlice(
                slice_idx=i,
                qty=qty,
                volume_weight=weights[i],
                scheduled_at=now + i * interval,
            ))

        order = VWAPOrder(
            symbol=symbol, side=side, total_qty=total_qty, slices=slices,
        )

        logger.info(
            "VWAP created: %s %s %s in %d slices over %ds",
            side, total_qty, symbol, n_slices, duration_sec,
        )
        return order

    def tick(self, order: VWAPOrder) -> Optional[VWAPSlice]:
        """Execute due slices."""
        now = time.monotonic()

        for i, s in enumerate(order.slices):
            if s.status != "pending" or now < s.scheduled_at:
                continue

            try:
                fill_price = self._submit(order.symbol, order.side, s.qty)
                order.slices[i] = VWAPSlice(
                    slice_idx=s.slice_idx,
                    qty=s.qty,
                    volume_weight=s.volume_weight,
                    scheduled_at=s.scheduled_at,
                    executed_at=now,
                    fill_price=fill_price,
                    status="executed" if fill_price else "failed",
                )
                return order.slices[i]
            except Exception as e:
                logger.warning("VWAP slice %d failed: %s", s.slice_idx, e)
                order.slices[i] = VWAPSlice(
                    slice_idx=s.slice_idx,
                    qty=s.qty,
                    volume_weight=s.volume_weight,
                    scheduled_at=s.scheduled_at,
                    executed_at=now,
                    status="failed",
                )
                return order.slices[i]

        return None
