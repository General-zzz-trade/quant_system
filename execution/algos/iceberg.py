"""Iceberg order — hides large orders by showing only small visible portions.

Splits a large order into small visible clips, submitting the next clip
only after the previous one fills.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class IcebergClip:
    """Single visible portion of the iceberg."""
    clip_idx: int
    qty: Decimal
    fill_price: Optional[Decimal] = None
    status: str = "pending"


@dataclass
class IcebergOrder:
    """Iceberg order state."""
    symbol: str
    side: str
    total_qty: Decimal
    clip_size: Decimal
    clips: List[IcebergClip] = field(default_factory=list)
    current_clip: int = 0

    @property
    def filled_qty(self) -> Decimal:
        return sum(c.qty for c in self.clips if c.status == "filled")

    @property
    def remaining_qty(self) -> Decimal:
        return self.total_qty - self.filled_qty

    @property
    def is_complete(self) -> bool:
        return self.remaining_qty <= Decimal("0")

    @property
    def avg_price(self) -> Optional[Decimal]:
        filled = [c for c in self.clips if c.fill_price is not None]
        if not filled:
            return None
        total_notional = sum(c.qty * c.fill_price for c in filled)
        total_qty = sum(c.qty for c in filled)
        return total_notional / total_qty if total_qty else None


class IcebergAlgo:
    """Iceberg execution algorithm.

    Submits one clip at a time, waiting for fill before sending next.
    """

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
        clip_size: Decimal = Decimal("0.1"),
    ) -> IcebergOrder:
        """Create an iceberg order."""
        n_clips = int(total_qty / clip_size)
        remainder = total_qty - clip_size * n_clips

        clips = [IcebergClip(clip_idx=i, qty=clip_size) for i in range(n_clips)]
        if remainder > 0:
            clips.append(IcebergClip(clip_idx=n_clips, qty=remainder))

        order = IcebergOrder(
            symbol=symbol, side=side, total_qty=total_qty,
            clip_size=clip_size, clips=clips,
        )

        logger.info(
            "Iceberg created: %s %s %s, clip_size=%s, n_clips=%d",
            side, total_qty, symbol, clip_size, len(clips),
        )
        return order

    def tick(self, order: IcebergOrder) -> Optional[IcebergClip]:
        """Submit next clip if previous is filled."""
        if order.is_complete or order.current_clip >= len(order.clips):
            return None

        clip = order.clips[order.current_clip]
        if clip.status != "pending":
            if clip.status == "filled":
                order.current_clip += 1
                return self.tick(order)
            return None

        try:
            fill_price = self._submit(order.symbol, order.side, clip.qty)
            if fill_price is not None:
                order.clips[order.current_clip] = IcebergClip(
                    clip_idx=clip.clip_idx,
                    qty=clip.qty,
                    fill_price=fill_price,
                    status="filled",
                )
                order.current_clip += 1
                logger.info(
                    "Iceberg clip %d: %s %s @ %s (remaining: %s)",
                    clip.clip_idx, order.side, clip.qty, fill_price, order.remaining_qty,
                )
                return order.clips[order.current_clip - 1]
        except Exception as e:
            logger.warning("Iceberg clip %d failed: %s", clip.clip_idx, e)

        return None
