# execution/adapters/binance/execution_adapter.py
"""BinanceExecutionAdapter -- bridges BinanceAdapter to the framework ExecutionBridge.

Wraps the BinanceAdapter (REST) so that it satisfies the
ExecutionAdapter protocol expected by runner.ExecutionBridge:

    send_order(order_event) -> Iterable[FillEvent]
"""
from __future__ import annotations

import logging
import time
from decimal import Decimal
from typing import Any

from event.header import EventHeader
from event.types import EventType, FillEvent
from event.domain import TimeInForce
from execution.order_utils import reliable_close_position

logger = logging.getLogger(__name__)


class BinanceExecutionAdapter:
    """Adapt BinanceAdapter to the framework ExecutionAdapter protocol."""

    def __init__(self, adapter: Any) -> None:
        self._adapter = adapter

    DEFAULT_TIF: TimeInForce = TimeInForce.GTC

    _MAX_RETRIES: int = 3
    _RETRY_DELAY: float = 0.5

    # ------------------------------------------------------------------
    def _send_with_retry(self, symbol: str, side: str, qty: float) -> dict[str, Any]:
        """Send market order with retries on transient failures."""
        last_err: Exception | None = None
        for attempt in range(1, self._MAX_RETRIES + 1):
            try:
                return self._adapter.send_market_order(symbol, side, qty)
            except Exception as e:
                last_err = e
                logger.warning(
                    "send_market_order attempt %d/%d failed: %s",
                    attempt, self._MAX_RETRIES, e,
                )
                if attempt < self._MAX_RETRIES:
                    time.sleep(self._RETRY_DELAY)
        logger.error("send_market_order exhausted %d retries", self._MAX_RETRIES)
        return {"status": "error", "msg": str(last_err)}

    # ------------------------------------------------------------------
    def send_order(self, order_event: Any) -> tuple[FillEvent, ...]:
        """Execute *order_event* via Binance REST and return FillEvent(s).

        Returns an empty tuple on any failure so the pipeline can
        continue without raising.
        """
        try:
            symbol: str = order_event.symbol
            side: str = order_event.side
            qty: Decimal = order_event.qty

            tif: TimeInForce = getattr(
                order_event, "time_in_force", self.DEFAULT_TIF,
            )
            if isinstance(tif, str):
                tif = TimeInForce(tif)

            # --- dispatch -------------------------------------------
            if qty == 0:
                resp = reliable_close_position(self._adapter, symbol)
            else:
                resp = self._send_with_retry(symbol, side, float(qty))

            # --- check result ---------------------------------------
            status = resp.get("status", "")
            if status in ("error", "failed"):
                logger.warning(
                    "binance order failed: symbol=%s side=%s qty=%s resp=%s",
                    symbol, side, qty, resp,
                )
                return ()

            # --- fetch actual fill price ----------------------------
            time.sleep(0.3)
            try:
                fills = self._adapter.get_recent_fills(symbol=symbol)
                fill_price = fills[0].price if fills else 0.0
            except Exception:
                fill_price = 0.0

            # --- build FillEvent ------------------------------------
            header = EventHeader.from_parent(
                parent=order_event.header,
                event_type=EventType.FILL,
                version=1,
                source="binance",
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
            logger.exception("BinanceExecutionAdapter.send_order failed")
            return ()
