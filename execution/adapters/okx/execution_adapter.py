"""OkxExecutionAdapter — bridges OkxAdapter to the framework ExecutionBridge.

Mirrors BinanceExecutionAdapter so runner.ExecutionBridge can swap venues
without modification. The OKX-specific concerns (contract conversion, tick
rounding, notional cap) all live inside OkxAdapter; this layer only adds
the retry envelope and FillEvent construction.

Entry orders use post_only limit orders at best bid/ask for maker fee
savings (2bps vs 5bps taker).  Falls back to market after 3s if not filled.
Exit orders always use market orders for immediate execution.

Order splitting: large entry orders are split into 2-3 chunks with 30s
intervals to reduce market impact. Weekend qty is reduced by 30%.
Exits/closes are NEVER split — immediate execution for safety.
"""
from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from event.header import EventHeader
from event.types import EventType, FillEvent
from event.domain import TimeInForce
from execution.order_utils import reliable_close_position
from monitoring.tca import TCALogger

logger = logging.getLogger(__name__)

# Entry limit order timeout: cancel and fall back to market after this
_LIMIT_FILL_TIMEOUT_S = float(os.environ.get("OKX_LIMIT_FILL_TIMEOUT_S", "3.0"))

# Order splitting thresholds (notional USD)
_SPLIT_THRESHOLD_NORMAL = float(os.environ.get("OKX_SPLIT_THRESHOLD", "1000.0"))
_SPLIT_THRESHOLD_WEEKEND = float(os.environ.get("OKX_SPLIT_THRESHOLD_WEEKEND", "500.0"))
_SPLIT_INTERVAL_S = float(os.environ.get("OKX_SPLIT_INTERVAL_S", "30.0"))

# Weekend qty reduction factor (0.7 = reduce by 30%)
_WEEKEND_QTY_FACTOR = Decimal(os.environ.get("OKX_WEEKEND_QTY_FACTOR", "0.7"))


class OkxExecutionAdapter:
    """Adapt OkxAdapter to the framework ExecutionAdapter protocol."""

    DEFAULT_TIF: TimeInForce = TimeInForce.GTC

    _MAX_RETRIES: int = 2
    _RETRY_DELAY: float = 0.3

    def __init__(self, adapter: Any, circuit_breaker: Any = None) -> None:
        self._adapter = adapter
        self._cb = circuit_breaker  # RustCircuitBreaker (optional)
        self._tca = TCALogger(venue="okx")

    # ------------------------------------------------------------------
    # Limit entry: post_only at best bid/ask → maker fee (2bps vs 5bps)
    # ------------------------------------------------------------------
    def _send_limit_entry(self, symbol: str, side: str, qty: float) -> dict[str, Any]:
        """Post_only limit at best bid/ask. Falls back to market if not filled."""
        try:
            tk = self._adapter._client.request_public(
                method="GET",
                path="/api/v5/market/ticker",
                params={"instId": self._adapter._get_meta(symbol).inst_id},
            )
            data = (tk.get("data") or [{}])[0]
            best_bid = float(data.get("bidPx", 0))
            best_ask = float(data.get("askPx", 0))
        except Exception:
            return self._adapter.send_market_order(symbol, side, qty)

        if best_bid <= 0 or best_ask <= 0:
            return self._adapter.send_market_order(symbol, side, qty)

        limit_price = best_bid if side.lower() == "buy" else best_ask

        resp = self._adapter.send_limit_order(
            symbol, side, qty, limit_price, post_only=True,
        )
        if resp.get("status") != "submitted":
            logger.info("OKX limit rejected, falling back to market: %s", resp.get("msg", ""))
            return self._adapter.send_market_order(symbol, side, qty)

        order_id = resp.get("orderId", "")

        # Poll for fill up to timeout
        polls = max(1, int(_LIMIT_FILL_TIMEOUT_S / 0.5))
        for _ in range(polls):
            time.sleep(0.5)
            try:
                fills = self._adapter.get_recent_fills(symbol=symbol)
                if fills and str(getattr(fills[0], "order_id", "")) == order_id:
                    logger.info("OKX limit entry FILLED: %s %s @ %.2f (maker fee)",
                                symbol, side, limit_price)
                    return resp
            except Exception:
                pass

        # Not filled — cancel and fall back to market
        try:
            self._adapter.cancel_order(symbol, order_id)
        except Exception:
            pass
        logger.info("OKX limit not filled in %.0fs → market fallback: %s %s",
                     _LIMIT_FILL_TIMEOUT_S, symbol, side)
        return self._adapter.send_market_order(symbol, side, qty)

    # ------------------------------------------------------------------
    # Weekend / order splitting helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _is_weekend() -> bool:
        return datetime.now(timezone.utc).weekday() >= 5

    def _should_split(self, qty: float, price: float) -> tuple[bool, int]:
        """Decide whether to split an entry order and into how many chunks."""
        notional = qty * price
        threshold = _SPLIT_THRESHOLD_WEEKEND if self._is_weekend() else _SPLIT_THRESHOLD_NORMAL
        if notional < threshold:
            return False, 1
        n_chunks = min(3, max(2, int(notional / threshold)))
        return True, n_chunks

    def _apply_weekend_scaling(self, qty: Decimal) -> Decimal:
        """Reduce qty by 30% during weekends for lower liquidity."""
        if self._is_weekend():
            scaled = qty * _WEEKEND_QTY_FACTOR
            logger.info("Weekend liquidity: qty scaled to %.0f%%: %s → %s",
                        float(_WEEKEND_QTY_FACTOR) * 100, qty, scaled)
            return scaled
        return qty

    def _send_split_entry(self, symbol: str, side: str, qty: float,
                          price: float) -> list[dict[str, Any]]:
        """Split a large entry order into chunks with intervals between them.

        Returns a list of response dicts (one per chunk).
        """
        _, n_chunks = self._should_split(qty, price)
        chunk_qty = qty / n_chunks
        responses: list[dict[str, Any]] = []

        for i in range(n_chunks):
            logger.info(
                "OKX split order %d/%d: %s %s qty=%.6f (total=%.6f)",
                i + 1, n_chunks, symbol, side, chunk_qty, qty,
            )
            resp = self._send_with_retry(symbol, side, chunk_qty, use_limit=True)
            responses.append(resp)

            # If this chunk failed, stop sending more
            if resp.get("status") in ("error", "failed"):
                logger.warning(
                    "OKX split chunk %d/%d failed, aborting remaining: %s",
                    i + 1, n_chunks, resp.get("msg", ""),
                )
                break

            # Sleep between chunks (but not after the last one)
            if i < n_chunks - 1:
                time.sleep(_SPLIT_INTERVAL_S)

        return responses

    # ------------------------------------------------------------------
    # Retry wrapper
    # ------------------------------------------------------------------
    def _send_with_retry(self, symbol: str, side: str, qty: float,
                         *, use_limit: bool = False) -> dict[str, Any]:
        last_err: Exception | None = None
        for attempt in range(1, self._MAX_RETRIES + 1):
            try:
                if use_limit:
                    resp = self._send_limit_entry(symbol, side, qty)
                else:
                    resp = self._adapter.send_market_order(symbol, side, qty)
                if resp.get("status") == "error" and resp.get("retryable"):
                    last_err = RuntimeError(resp.get("msg", ""))
                    logger.warning(
                        "OKX order attempt %d/%d retryable: %s",
                        attempt, self._MAX_RETRIES, last_err,
                    )
                    if attempt < self._MAX_RETRIES:
                        time.sleep(self._RETRY_DELAY)
                    continue
                return resp
            except Exception as e:
                last_err = e
                logger.warning(
                    "OKX send_order attempt %d/%d failed: %s",
                    attempt, self._MAX_RETRIES, e,
                )
                if attempt < self._MAX_RETRIES:
                    time.sleep(self._RETRY_DELAY)
        logger.error("OKX send_order exhausted %d retries", self._MAX_RETRIES)
        return {"status": "error", "msg": str(last_err) if last_err else "unknown"}

    # ------------------------------------------------------------------
    # Fetch real-time mid-price for accurate TCA reference
    # ------------------------------------------------------------------
    def _get_mid_price(self, symbol: str) -> float:
        """Fetch current mid-price from ticker for TCA reference."""
        try:
            tk = self._adapter._client.request_public(
                method="GET",
                path="/api/v5/market/ticker",
                params={"instId": self._adapter._get_meta(symbol).inst_id},
            )
            data = (tk.get("data") or [{}])[0]
            bid = float(data.get("bidPx", 0))
            ask = float(data.get("askPx", 0))
            if bid > 0 and ask > 0:
                return (bid + ask) / 2
            return float(data.get("last", 0))
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
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

            # Circuit breaker gate
            if self._cb is not None and qty != 0 and not self._cb.allow_request():
                logger.warning(
                    "OKX order blocked by circuit breaker: symbol=%s side=%s",
                    symbol, side,
                )
                return ()

            # TCA ref_price: use real-time mid-price instead of bar close
            # for accurate slippage measurement
            ref_price = self._get_mid_price(symbol)
            if ref_price <= 0:
                try:
                    ref_price = float(getattr(order_event, "price", None) or 0.0)
                except Exception:
                    ref_price = 0.0
            _send_ts = time.time()

            # Dispatch: entries use limit (maker fee), exits use market
            is_entry = qty != 0 and not getattr(order_event, "reduce_only", False)
            is_split = False

            if qty == 0:
                # Close position — always market for immediate exit
                resp = reliable_close_position(self._adapter, symbol)
            elif not is_entry:
                # Exit / reduce-only — always immediate, never split
                resp = self._send_with_retry(
                    symbol, side, float(qty),
                    use_limit=False,
                )
            else:
                # New entry — apply weekend scaling, then check for splitting
                qty = self._apply_weekend_scaling(qty)

                should_split, _ = self._should_split(float(qty), ref_price)
                if should_split and ref_price > 0:
                    is_split = True
                    split_responses = self._send_split_entry(
                        symbol, side, float(qty), ref_price,
                    )
                    # Use first non-error response as primary resp
                    resp = next(
                        (r for r in split_responses
                         if r.get("status") not in ("error", "failed")),
                        split_responses[0] if split_responses else
                        {"status": "error", "msg": "no split responses"},
                    )
                else:
                    resp = self._send_with_retry(
                        symbol, side, float(qty),
                        use_limit=True,
                    )

            status = resp.get("status", "")
            if status in ("error", "failed"):
                if self._cb is not None:
                    self._cb.record_failure()
                logger.warning(
                    "OKX order failed: symbol=%s side=%s qty=%s resp=%s",
                    symbol, side, qty, resp,
                )
                return ()

            # Fetch fill details
            time.sleep(0.3)
            fill_qty = qty
            fill_price = Decimal("0")
            try:
                fills = self._adapter.get_recent_fills(symbol=symbol)
                if fills:
                    if is_split and len(fills) > 1:
                        # Aggregate split fills: VWAP price, sum qty
                        n_expect = min(3, len(fills))
                        agg_qty = Decimal("0")
                        agg_notional = Decimal("0")
                        for f in fills[:n_expect]:
                            fq = Decimal(str(f.qty))
                            fp = Decimal(str(f.price))
                            agg_qty += fq
                            agg_notional += fq * fp
                        fill_qty = agg_qty if agg_qty > 0 else qty
                        fill_price = (agg_notional / agg_qty) if agg_qty > 0 else Decimal("0")
                        logger.info(
                            "OKX split fill aggregated: %d fills, qty=%s, vwap=%s",
                            n_expect, fill_qty, fill_price,
                        )
                    else:
                        if qty == 0 and len(fills) > 1:
                            # close_position may produce multiple partial fills
                            agg_qty = Decimal("0")
                            agg_notional = Decimal("0")
                            for f in fills:
                                fq = Decimal(str(f.qty))
                                fp = Decimal(str(f.price))
                                agg_qty += fq
                                agg_notional += fq * fp
                            fill_qty = agg_qty if agg_qty > 0 else Decimal(str(fills[0].qty))
                            fill_price = (agg_notional / agg_qty) if agg_qty > 0 else Decimal(str(fills[0].price))
                            logger.info(
                                "OKX close aggregated: %d fills, qty=%s, vwap=%s",
                                len(fills), fill_qty, fill_price,
                            )
                        else:
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

            # TCA with real-time mid-price reference
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

            # Warn on excessive slippage
            if ref_price > 0 and float(fill_price) > 0:
                slippage_bps = abs(float(fill_price) - ref_price) / ref_price * 10_000
                if slippage_bps > 50:
                    logger.warning(
                        "HIGH SLIPPAGE: %s %s %.0f bps (ref=%.2f fill=%.2f lat=%.0fms)",
                        symbol, side, slippage_bps, ref_price, float(fill_price), latency_ms,
                    )

            return (fill,)

        except Exception:
            logger.exception("OkxExecutionAdapter.send_order failed")
            return ()
