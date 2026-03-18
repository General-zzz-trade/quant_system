# execution/adapters/binance/ws_order_adapter.py
"""WS-API order execution adapter — wraps BinanceWsOrderGateway as ExecutionAdapter.

Provides synchronous send_order() via threading.Event wait on WS response.
Falls back to REST adapter on WS failure or disconnect.

Latency: ~4ms (WS-API) vs ~30ms (REST).
"""
from __future__ import annotations

from dataclasses import dataclass
import logging
import threading
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_NON_FALLBACK_WS_ERROR_CODES = {-2015, -2014, -1002}
_NON_FALLBACK_HTTP_STATUSES = {401, 403}


class WsOrderAdapter:
    """ExecutionAdapter that routes orders through WS-API with REST fallback.

    Usage:
        adapter = WsOrderAdapter(
            rest_adapter=venue_client,
            api_key="...",
            api_secret="...",
            testnet=True,
        )
        adapter.start()
        results = adapter.send_order(order_event)
    """

    def __init__(
        self,
        rest_adapter: Any,
        *,
        api_key: str,
        api_secret: str,
        testnet: bool = False,
        response_timeout_sec: float = 0.5,
    ) -> None:
        self._rest = rest_adapter
        self._api_key = api_key
        self._api_secret = api_secret
        self._testnet = testnet
        self._timeout = response_timeout_sec
        self._gateway: Optional[Any] = None
        self._pending: Dict[str, _PendingOrder] = {}
        self._lock = threading.Lock()
        self._started = False
        self._last_outcome: Optional[WsOrderOutcome] = None

    def start(self) -> None:
        """Connect WS gateway. Safe to call multiple times."""
        if self._started:
            return
        try:
            from execution.adapters.binance.ws_order_gateway import BinanceWsOrderGateway

            self._gateway = BinanceWsOrderGateway(
                api_key=self._api_key,
                api_secret=self._api_secret,
                testnet=self._testnet,
                on_response=self._on_response,
                on_error=self._on_error,
            )
            self._gateway.start()
            self._started = True
            logger.info("WS order gateway started (testnet=%s)", self._testnet)
        except Exception:
            logger.warning("WS order gateway start failed — will use REST", exc_info=True)
            self._gateway = None

    def stop(self) -> None:
        """Disconnect WS gateway."""
        if self._gateway is not None:
            try:
                self._gateway.stop()
            except Exception:
                logger.warning("WS order gateway stop error", exc_info=True)
            self._gateway = None
        self._started = False

    @property
    def is_ws_connected(self) -> bool:
        return self._gateway is not None and getattr(self._gateway, "is_running", False)

    @property
    def last_order_outcome(self) -> Optional[WsOrderOutcome]:
        """Return the most recent order submission outcome observed by this adapter."""
        with self._lock:
            return self._last_outcome

    def send_order(self, order_event: Any) -> list:
        """ExecutionAdapter protocol: submit order via WS-API, fallback to REST.

        Returns list of result events (fills, rejects, etc.).
        """
        if not self.is_ws_connected:
            self._set_last_outcome(route="rest_fallback", reason="ws_disconnected")
            return self._fallback_rest(order_event, reason="ws_disconnected")

        symbol = str(getattr(order_event, "symbol", ""))
        side = str(getattr(order_event, "side", "BUY")).upper()
        order_type = str(getattr(order_event, "order_type", "MARKET")).upper()
        qty = getattr(order_event, "qty", None) or getattr(order_event, "quantity", None)
        price = getattr(order_event, "price", None)
        tif = getattr(order_event, "time_in_force", None)
        reduce_only = getattr(order_event, "reduce_only", None)
        client_order_id = getattr(order_event, "client_order_id", None)

        # Prepare synchronous wait
        event = threading.Event()
        pending = _PendingOrder(event=event)

        try:
            req_id = self._gateway.submit_order(
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=str(qty) if qty is not None else None,
                price=str(price) if price is not None else None,
                time_in_force=str(tif) if tif is not None else None,
                reduce_only=reduce_only,
                client_order_id=str(client_order_id) if client_order_id else None,
            )
        except Exception:
            self._set_last_outcome(route="rest_fallback", reason="ws_submit_error")
            logger.warning("WS submit_order failed, falling back to REST", exc_info=True)
            return self._fallback_rest(order_event, reason="ws_submit_error")

        with self._lock:
            self._pending[req_id] = pending

        # Wait for response
        if not event.wait(timeout=self._timeout):
            logger.warning(
                "WS order timeout after %.1fs (req_id=%s), falling back to REST",
                self._timeout, req_id,
            )
            with self._lock:
                self._pending.pop(req_id, None)
            self._set_last_outcome(route="rest_fallback", reason="ws_timeout", req_id=req_id)
            return self._fallback_rest(order_event, reason="ws_timeout")

        with self._lock:
            self._pending.pop(req_id, None)

        if pending.error is not None:
            if not self._should_fallback_on_error(pending.error):
                self._set_last_outcome(
                    route="ws_error_no_fallback",
                    reason="ws_auth_or_permission_error",
                    req_id=req_id,
                    error=pending.error,
                )
                logger.warning(
                    "WS order error (req_id=%s): %s, not falling back to REST",
                    req_id, pending.error,
                )
                return []
            logger.warning(
                "WS order error (req_id=%s): %s, falling back to REST",
                req_id, pending.error,
            )
            self._set_last_outcome(
                route="rest_fallback",
                reason="ws_error",
                req_id=req_id,
                error=pending.error,
            )
            return self._fallback_rest(order_event, reason="ws_error")

        # WS success — delegate fill handling to REST adapter's result processing
        # The WS response is the raw exchange ack; wrap in same format as REST
        latency_ms = pending.latency_ms
        logger.info(
            "WS order filled: %s %s qty=%s latency=%.1fms",
            side, symbol, qty, latency_ms,
        )

        if pending.response is None:
            self._set_last_outcome(
                route="rest_fallback",
                reason="ws_empty_response",
                req_id=req_id,
                latency_ms=latency_ms,
            )
            logger.warning(
                "WS order completed without response payload (req_id=%s), falling back to REST",
                req_id,
            )
            return self._fallback_rest(order_event, reason="ws_empty_response")

        self._set_last_outcome(
            route="ws_success",
            reason="ws_response",
            req_id=req_id,
            latency_ms=latency_ms,
            response=pending.response,
        )
        return []

    def _fallback_rest(self, order_event: Any, reason: str) -> list:
        """Fall back to REST adapter."""
        logger.debug("REST fallback: reason=%s", reason)
        return list(self._rest.send_order(order_event))

    def _set_last_outcome(
        self,
        *,
        route: str,
        reason: str,
        req_id: Optional[str] = None,
        latency_ms: float = 0.0,
        response: Optional[Dict[str, Any]] = None,
        error: Optional[Dict[str, Any]] = None,
    ) -> None:
        with self._lock:
            self._last_outcome = WsOrderOutcome(
                route=route,
                reason=reason,
                req_id=req_id,
                latency_ms=latency_ms,
                response=response,
                error=error,
            )

    @staticmethod
    def _should_fallback_on_error(resp: Dict[str, Any]) -> bool:
        error = resp.get("error") if isinstance(resp, dict) else None
        code = None
        if isinstance(error, dict):
            code = error.get("code")
        elif isinstance(resp, dict):
            code = resp.get("code")
        status = resp.get("status") if isinstance(resp, dict) else None
        try:
            if status is not None and int(status) in _NON_FALLBACK_HTTP_STATUSES:
                return False
        except (TypeError, ValueError):
            pass
        try:
            if code is not None and int(code) in _NON_FALLBACK_WS_ERROR_CODES:
                return False
        except (TypeError, ValueError):
            pass
        return True

    def _on_response(self, resp: Dict[str, Any]) -> None:
        """WS response callback (called from gateway recv thread)."""
        req_id = resp.get("id") or resp.get("req_id")
        if req_id is None:
            return
        req_id = str(req_id)
        with self._lock:
            pending = self._pending.get(req_id)
        if pending is not None:
            pending.response = resp
            pending.latency_ms = resp.get("_latency_ms", 0.0)
            pending.event.set()

    def _on_error(self, req_id: str, resp: Dict[str, Any]) -> None:
        """WS error callback (called from gateway recv thread)."""
        with self._lock:
            pending = self._pending.get(str(req_id))
        if pending is not None:
            pending.error = resp
            pending.event.set()


class _PendingOrder:
    """Tracks a pending WS order submission."""
    __slots__ = ("event", "response", "error", "latency_ms")

    def __init__(self, event: threading.Event) -> None:
        self.event = event
        self.response: Optional[Dict[str, Any]] = None
        self.error: Optional[Dict[str, Any]] = None
        self.latency_ms: float = 0.0


@dataclass(frozen=True)
class WsOrderOutcome:
    """Most recent order submission outcome for observability and integration guards."""

    route: str
    reason: str
    req_id: Optional[str] = None
    latency_ms: float = 0.0
    response: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None
