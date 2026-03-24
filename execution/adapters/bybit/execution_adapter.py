"""BybitExecutionAdapter — bridges BybitAdapter to the framework ExecutionBridge.

Wraps the existing BybitAdapter (REST V5) so that it satisfies the
ExecutionAdapter protocol expected by runner.ExecutionBridge:

    send_order(order_event) -> Iterable[FillEvent]
"""
from __future__ import annotations

import logging
import time
from decimal import Decimal
from typing import Any, Iterable

from event.header import EventHeader
from event.types import EventType, FillEvent
from event.domain import TimeInForce
from execution.order_utils import reliable_close_position

logger = logging.getLogger(__name__)


class BybitExecutionAdapter:
    """Adapt BybitAdapter to the framework ExecutionAdapter protocol."""

    def __init__(self, adapter: Any) -> None:
        self._adapter = adapter

    # Default time-in-force for market orders
    DEFAULT_TIF: TimeInForce = TimeInForce.GTC

    # ------------------------------------------------------------------
    def send_order(self, order_event: Any) -> Iterable[Any]:
        """Execute *order_event* via Bybit REST and return FillEvent(s).

        Returns an empty tuple on any failure so the pipeline can
        continue without raising.
        """
        try:
            symbol: str = order_event.symbol
            side: str = order_event.side
            qty: Decimal = order_event.qty

            # Resolve time-in-force from order event or use default
            tif: TimeInForce = getattr(
                order_event, "time_in_force", self.DEFAULT_TIF,
            )
            if isinstance(tif, str):
                tif = TimeInForce(tif)

            # --- dispatch -------------------------------------------
            if qty == 0:
                resp = reliable_close_position(self._adapter, symbol)
            else:
                resp = self._adapter.send_market_order(
                    symbol, side, float(qty),
                )

            # --- check result ---------------------------------------
            status = resp.get("status", "")
            if status in ("error", "failed"):
                logger.warning(
                    "bybit order failed: symbol=%s side=%s qty=%s resp=%s",
                    symbol, side, qty, resp,
                )
                return ()

            # --- fetch actual fill price ----------------------------
            time.sleep(0.3)
            try:
                fills = self._adapter.get_recent_fills(symbol=symbol)
                fill_price = fills[0].price if fills else float(qty)
            except Exception:
                fill_price = 0.0

            # --- build FillEvent ------------------------------------
            header = EventHeader.from_parent(
                parent=order_event.header,
                event_type=EventType.FILL,
                version=1,
                source="bybit",
            )
            fill = FillEvent(
                header=header,
                fill_id=header.event_id,
                order_id=order_event.order_id,
                symbol=symbol,
                qty=order_event.qty,
                price=Decimal(str(fill_price)),
                side=side,
            )
            return (fill,)

        except Exception:
            logger.exception("BybitExecutionAdapter.send_order failed")
            return ()
