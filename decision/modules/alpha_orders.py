"""Order event factory methods extracted from AlphaDecisionModule."""
from __future__ import annotations

import logging
from decimal import Decimal
from event.header import EventHeader
from event.types import EventType, OrderEvent

logger = logging.getLogger(__name__)


def make_open_order(
    symbol: str,
    runner_key: str,
    price: float,
    signal: int,
    qty: Decimal,
) -> list[OrderEvent]:
    """Create OrderEvent for opening a new position."""
    header = EventHeader.new_root(
        event_type=EventType.ORDER,
        version=1,
        source=f"alpha.{runner_key}",
    )
    side = "buy" if signal == 1 else "sell"
    return [
        OrderEvent(
            header=header,
            order_id=header.event_id,
            intent_id=header.event_id,
            symbol=symbol,
            side=side,
            qty=qty,
            price=Decimal(str(price)),
        )
    ]


def make_close_order(
    symbol: str,
    runner_key: str,
    price: float,
    old_signal: int,
    reason: str,
    current_qty: Decimal,
    min_size: Decimal,
) -> list[OrderEvent]:
    """Create OrderEvent for closing current position."""
    # Use tracked qty; fallback to min_size to avoid zero-qty rejection
    qty = current_qty if current_qty > 0 else min_size
    header = EventHeader.new_root(
        event_type=EventType.ORDER,
        version=1,
        source=f"alpha.{runner_key}",
    )
    # Close side is opposite of position
    side = "sell" if old_signal == 1 else "buy"
    logger.info(
        "%s CLOSE %s qty=%.6f reason=%s price=%.2f",
        runner_key, side, float(qty), reason, price,
    )
    return [
        OrderEvent(
            header=header,
            order_id=header.event_id,
            intent_id=header.event_id,
            symbol=symbol,
            side=side,
            qty=Decimal(str(qty)),
            price=Decimal(str(price)),
        )
    ]
