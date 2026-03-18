# execution/adapters/binance/ws_order_gateway.py
"""WS-API order gateway for Binance Futures.

Sends orders via WebSocket (wss://ws-fapi.binance.com/ws-fapi/v1) instead of REST.
Latency: ~4ms (1 RTT) vs ~30-200ms (REST: TCP+TLS+HTTP).

Requires a dedicated RustWsClient for the trading WS-API endpoint.
Market data WS and order WS are separate connections.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any, Callable, Dict, Optional

from _quant_hotpath import RustWsClient, RustWsOrderGateway

logger = logging.getLogger(__name__)

# Binance Futures WS-API endpoints
WS_FAPI_MAINNET = "wss://ws-fapi.binance.com/ws-fapi/v1"
WS_FAPI_TESTNET = "wss://testnet.binancefuture.com/ws-fapi/v1"


class BinanceWsOrderGateway:
    """WebSocket order gateway for Binance USDT-M Futures.

    Architecture:
    - RustWsClient handles connection + recv (GIL-free, tokio)
    - RustWsOrderGateway builds signed JSON-RPC messages (pure Rust, ~5μs)
    - Background thread processes responses and invokes callbacks

    Usage:
        gw = BinanceWsOrderGateway(api_key, api_secret, testnet=True)
        gw.start()
        req_id = gw.submit_order(symbol="BTCUSDT", side="BUY", order_type="MARKET", qty="0.001")
        # response arrives via on_response callback
        gw.stop()
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        *,
        testnet: bool = False,
        recv_window: int = 5000,
        on_response: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_error: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> None:
        self._url = WS_FAPI_TESTNET if testnet else WS_FAPI_MAINNET
        self._ws = RustWsClient(buffer_size=1024)
        self._gateway = RustWsOrderGateway(api_key, api_secret, recv_window)
        self._on_response = on_response
        self._on_error = on_error
        self._pending: Dict[str, float] = {}  # req_id -> submit_ts
        self._recv_thread: Optional[threading.Thread] = None
        self._running = False

    def start(self) -> None:
        """Connect and start response listener."""
        if self._running:
            return
        self._running = True
        self._ws.connect(self._url)
        self._recv_thread = threading.Thread(
            target=self._recv_loop, name="ws-order-recv", daemon=True,
        )
        self._recv_thread.start()
        logger.info("WS order gateway connected: %s", self._url)

    def stop(self) -> None:
        """Close connection and stop listener."""
        self._running = False
        self._ws.close()
        if self._recv_thread is not None:
            self._recv_thread.join(timeout=3.0)
            self._recv_thread = None

    @property
    def is_running(self) -> bool:
        return self._running and self._ws.is_running()

    def submit_order(
        self,
        *,
        symbol: str,
        side: str,
        order_type: str = "MARKET",
        quantity: Optional[str] = None,
        price: Optional[str] = None,
        time_in_force: Optional[str] = None,
        reduce_only: Optional[bool] = None,
        client_order_id: Optional[str] = None,
    ) -> str:
        """Submit order via WS-API. Returns request_id for response correlation."""
        msg, req_id = self._gateway.build_order_message(
            symbol, side, order_type,
            quantity=quantity,
            price=price,
            time_in_force=time_in_force,
            reduce_only=reduce_only,
            new_client_order_id=client_order_id,
        )
        self._pending[req_id] = time.monotonic()
        self._ws.send(msg)
        return req_id

    def cancel_order(
        self,
        *,
        symbol: str,
        order_id: Optional[int] = None,
        orig_client_order_id: Optional[str] = None,
    ) -> str:
        """Cancel order via WS-API. Returns request_id."""
        msg, req_id = self._gateway.build_cancel_message(
            symbol,
            order_id=order_id,
            orig_client_order_id=orig_client_order_id,
        )
        self._pending[req_id] = time.monotonic()
        self._ws.send(msg)
        return req_id

    def modify_order(
        self,
        *,
        symbol: str,
        side: str,
        order_type: str = "LIMIT",
        quantity: str,
        price: str,
        order_id: Optional[int] = None,
        orig_client_order_id: Optional[str] = None,
        time_in_force: Optional[str] = "GTC",
    ) -> str:
        """Modify (cancel-replace) order via WS-API in 1 RTT. Returns request_id."""
        msg, req_id = self._gateway.build_modify_message(
            symbol, side, order_type, quantity, price,
            order_id=order_id,
            orig_client_order_id=orig_client_order_id,
            time_in_force=time_in_force,
        )
        self._pending[req_id] = time.monotonic()
        self._ws.send(msg)
        return req_id

    def query_order(
        self,
        *,
        symbol: str,
        order_id: Optional[int] = None,
        orig_client_order_id: Optional[str] = None,
    ) -> str:
        """Query order status via WS-API. Returns request_id."""
        msg, req_id = self._gateway.build_query_message(
            symbol,
            order_id=order_id,
            orig_client_order_id=orig_client_order_id,
        )
        self._pending[req_id] = time.monotonic()
        self._ws.send(msg)
        return req_id

    def _recv_loop(self) -> None:
        """Background thread: receive and dispatch WS-API responses."""
        while self._running:
            raw = self._ws.recv(timeout_ms=1000)
            if raw is None:
                continue
            try:
                resp = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("WS order: invalid JSON response: %s", raw[:200])
                continue

            req_id = resp.get("id", "")
            submit_ts = self._pending.pop(req_id, None)
            if submit_ts is not None:
                latency_ms = (time.monotonic() - submit_ts) * 1000
                resp["_latency_ms"] = latency_ms

            status = resp.get("status")
            if status and status != 200:
                # Error response
                error = resp.get("error", {})
                logger.warning(
                    "WS order error: id=%s code=%s msg=%s",
                    req_id, error.get("code"), error.get("msg"),
                )
                if self._on_error is not None:
                    self._on_error(req_id, resp)
            else:
                if self._on_response is not None:
                    self._on_response(resp)
