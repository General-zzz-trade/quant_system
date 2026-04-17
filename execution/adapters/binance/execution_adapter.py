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
from monitoring.tca import TCALogger

logger = logging.getLogger(__name__)


class BinanceExecutionAdapter:
    """Adapt BinanceAdapter to the framework ExecutionAdapter protocol."""

    def __init__(self, adapter: Any, circuit_breaker: Any = None) -> None:
        self._adapter = adapter
        self._cb = circuit_breaker  # RustCircuitBreaker (optional)
        self._tca = TCALogger(venue="binance")

    DEFAULT_TIF: TimeInForce = TimeInForce.GTC

    _MAX_RETRIES: int = 2
    _RETRY_DELAY: float = 0.3

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

            # --- circuit breaker gate --------------------------------
            if self._cb is not None and qty != 0 and not self._cb.allow_request():
                logger.warning(
                    "binance order blocked by circuit breaker: symbol=%s side=%s",
                    symbol, side,
                )
                return ()

            # TCA: capture real-time mid-price for accurate slippage
            # measurement (bar close is stale by the time order is sent).
            ref_price = 0.0
            try:
                tk = self._adapter.get_ticker(symbol)
                if tk and hasattr(tk, "bid") and hasattr(tk, "ask"):
                    bid, ask = float(tk.bid), float(tk.ask)
                    if bid > 0 and ask > 0:
                        ref_price = (bid + ask) / 2
                if ref_price <= 0:
                    ref_price = float(getattr(order_event, "price", None) or 0.0)
            except Exception:
                try:
                    ref_price = float(getattr(order_event, "price", None) or 0.0)
                except Exception:
                    ref_price = 0.0
            _send_ts = time.time()

            # --- dispatch -------------------------------------------
            if qty == 0:
                resp = reliable_close_position(self._adapter, symbol)
            else:
                resp = self._send_with_retry(symbol, side, float(qty))

            # --- check result ---------------------------------------
            status = resp.get("status", "")
            if status in ("error", "failed"):
                if self._cb is not None:
                    self._cb.record_failure()
                logger.warning(
                    "binance order failed: symbol=%s side=%s qty=%s resp=%s",
                    symbol, side, qty, resp,
                )
                return ()

            # --- fetch actual fill qty and price ----------------------
            time.sleep(0.3)
            fill_qty = qty
            fill_price = Decimal("0")
            try:
                fills = self._adapter.get_recent_fills(symbol=symbol)
                if fills:
                    fill_price = Decimal(str(fills[0].price))
                    # For close orders (qty=0), use the actual executed qty
                    if qty == 0:
                        fill_qty = Decimal(str(fills[0].qty))
            except Exception:
                pass

            # Skip FillEvent if nothing was actually executed
            if fill_qty == 0:
                return ()

            if self._cb is not None:
                self._cb.record_success()

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
                qty=fill_qty,
                price=fill_price,
                side=side,
            )

            # TCA — fail-open, never blocks the return.
            latency_ms = (time.time() - _send_ts) * 1000.0
            try:
                self._tca.record_fill(
                    symbol=symbol,
                    side=side,
                    qty=float(fill_qty),
                    ref_price=ref_price,
                    fill_price=float(fill_price),
                    latency_ms=latency_ms,
                    order_id=str(order_event.order_id),
                    fill_id=str(header.event_id),
                )
            except Exception:
                logger.debug("TCA record_fill failed", exc_info=True)

            # Warn on excessive slippage or latency
            if ref_price > 0 and float(fill_price) > 0:
                slippage_bps = abs(float(fill_price) - ref_price) / ref_price * 10_000
                if slippage_bps > 100:
                    logger.warning(
                        "HIGH SLIPPAGE: %s %s %.0f bps (ref=%.2f fill=%.2f latency=%.0fms)",
                        symbol, side, slippage_bps, ref_price, float(fill_price), latency_ms,
                    )

            return (fill,)

        except Exception:
            logger.exception("BinanceExecutionAdapter.send_order failed")
            return ()
