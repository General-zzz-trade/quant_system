"""OkxExecutionAdapter — bridges OkxAdapter to the framework ExecutionBridge.

Mirrors BinanceExecutionAdapter so runner.ExecutionBridge can swap venues
without modification. The OKX-specific concerns (contract conversion, tick
rounding, notional cap) all live inside OkxAdapter; this layer only adds
the retry envelope and FillEvent construction.
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
from monitoring.tca import TCALogger

logger = logging.getLogger(__name__)


class OkxExecutionAdapter:
    """Adapt OkxAdapter to the framework ExecutionAdapter protocol."""

    DEFAULT_TIF: TimeInForce = TimeInForce.GTC

    _MAX_RETRIES: int = 3
    _RETRY_DELAY: float = 0.5

    def __init__(self, adapter: Any, circuit_breaker: Any = None) -> None:
        self._adapter = adapter
        self._cb = circuit_breaker  # RustCircuitBreaker (optional)
        self._tca = TCALogger(venue="okx")

    def _send_with_retry(self, symbol: str, side: str, qty: float) -> dict[str, Any]:
        last_err: Exception | None = None
        for attempt in range(1, self._MAX_RETRIES + 1):
            try:
                resp = self._adapter.send_market_order(symbol, side, qty)
                # If the response indicates a retryable transient error, retry;
                # otherwise return immediately.
                if resp.get("status") == "error" and resp.get("retryable"):
                    last_err = RuntimeError(resp.get("msg", ""))
                    logger.warning(
                        "OKX order attempt %d/%d retryable error: %s",
                        attempt, self._MAX_RETRIES, last_err,
                    )
                    if attempt < self._MAX_RETRIES:
                        time.sleep(self._RETRY_DELAY)
                    continue
                return resp
            except Exception as e:
                last_err = e
                logger.warning(
                    "OKX send_market_order attempt %d/%d failed: %s",
                    attempt, self._MAX_RETRIES, e,
                )
                if attempt < self._MAX_RETRIES:
                    time.sleep(self._RETRY_DELAY)
        logger.error("OKX send_market_order exhausted %d retries", self._MAX_RETRIES)
        return {"status": "error", "msg": str(last_err) if last_err else "unknown"}

    def send_order(self, order_event: Any) -> tuple[FillEvent, ...]:
        try:
            symbol: str = order_event.symbol
            side: str = order_event.side
            qty: Decimal = order_event.qty

            tif: TimeInForce = getattr(
                order_event, "time_in_force", self.DEFAULT_TIF,
            )
            if isinstance(tif, str):
                tif = TimeInForce(tif)

            # Circuit breaker gate (only for non-flatten orders)
            if self._cb is not None and qty != 0 and not self._cb.allow_request():
                logger.warning(
                    "OKX order blocked by circuit breaker: symbol=%s side=%s",
                    symbol, side,
                )
                return ()

            # Reference price for TCA slippage = the bar-close price that
            # the decision module stamped on OrderEvent.price when it built
            # the order (see decision/modules/alpha_orders.py).  Robust to
            # None/0 via the fail-open guard in TCALogger.
            try:
                ref_price = float(getattr(order_event, "price", None) or 0.0)
            except Exception:
                ref_price = 0.0
            _send_ts = time.time()

            if qty == 0:
                resp = reliable_close_position(self._adapter, symbol)
            else:
                resp = self._send_with_retry(symbol, side, float(qty))

            status = resp.get("status", "")
            if status in ("error", "failed"):
                if self._cb is not None:
                    self._cb.record_failure()
                logger.warning(
                    "OKX order failed: symbol=%s side=%s qty=%s resp=%s",
                    symbol, side, qty, resp,
                )
                return ()

            # Give OKX a moment to settle then fetch the fill
            time.sleep(0.3)
            fill_qty = qty
            fill_price = Decimal("0")
            try:
                fills = self._adapter.get_recent_fills(symbol=symbol)
                if fills:
                    fill_price = Decimal(str(fills[0].price))
                    if qty == 0:
                        fill_qty = Decimal(str(fills[0].qty))
            except Exception:
                pass

            if fill_qty == 0:
                return ()

            if self._cb is not None:
                self._cb.record_success()

            header = EventHeader.from_parent(
                parent=order_event.header,
                event_type=EventType.FILL,
                version=1,
                source="okx",
            )
            fill = FillEvent(
                header=header,
                fill_id=header.event_id,
                order_id=order_event.order_id,
                symbol=symbol,
                qty=fill_qty,
                price=fill_price,
                side=side,
            )

            # TCA — slippage + latency (fail-open: never blocks the return).
            try:
                self._tca.record_fill(
                    symbol=symbol,
                    side=side,
                    qty=float(fill_qty),
                    ref_price=ref_price,
                    fill_price=float(fill_price),
                    latency_ms=(time.time() - _send_ts) * 1000.0,
                    order_id=str(order_event.order_id),
                    fill_id=str(header.event_id),
                )
            except Exception:
                logger.debug("TCA record_fill failed", exc_info=True)

            return (fill,)

        except Exception:
            logger.exception("OkxExecutionAdapter.send_order failed")
            return ()
