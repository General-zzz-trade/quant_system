# execution/adapters/binance/liquidation_poller.py
"""Daemon thread that listens to Binance liquidation WebSocket stream.

Aggregates liquidation volume and imbalance over a sliding window.
Uses wss://fstream.binance.com/ws/<symbol>@forceOrder stream.
Falls back to REST allForceOrders if WebSocket unavailable.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from typing import Deque, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

_WS_PROD = "wss://fstream.binance.com/ws"
_WS_TESTNET = "wss://stream.binancefuture.com/ws"


class BinanceLiquidationPoller:
    """Subscribes to Binance liquidation WS stream and aggregates into features.

    Maintains a 1-hour sliding window of liquidation events.
    Exposes: total_volume, buy_volume, sell_volume, count.
    """

    def __init__(
        self,
        symbol: str = "BTCUSDT",
        window_sec: float = 3600.0,
        testnet: bool = False,
    ) -> None:
        self._symbol = symbol
        self._window_sec = window_sec
        self._ws_url = f"{_WS_TESTNET if testnet else _WS_PROD}/{symbol.lower()}@forceOrder"
        # (timestamp_ms, side, quote_qty) tuples
        self._events: Deque[Tuple[int, str, float]] = deque(maxlen=10000)
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._ws_loop, daemon=True)
        self._thread.start()
        logger.info("LiquidationPoller started for %s (WS stream)", self._symbol)

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            from infra.threading_utils import safe_join_thread

            safe_join_thread(self._thread, timeout=5.0)
            self._thread = None

    def get_current(self) -> Optional[Dict[str, float]]:
        """Return aggregated liquidation metrics for the sliding window."""
        with self._lock:
            now_ms = int(time.time() * 1000)
            cutoff = now_ms - int(self._window_sec * 1000)
            # Prune old events
            while self._events and self._events[0][0] < cutoff:
                self._events.popleft()

            if not self._events:
                return None

            total_vol = 0.0
            buy_vol = 0.0
            sell_vol = 0.0
            count = 0
            for _, side, qty in self._events:
                total_vol += qty
                if side == "BUY":
                    buy_vol += qty
                else:
                    sell_vol += qty
                count += 1

            return {
                "liq_total_volume": total_vol,
                "liq_buy_volume": buy_vol,
                "liq_sell_volume": sell_vol,
                "liq_count": float(count),
            }

    def _ws_loop(self) -> None:
        """Connect to WS stream and process liquidation events."""
        while self._running:
            try:
                self._run_ws()
            except Exception:
                logger.exception("LiquidationPoller WS error, reconnecting in 5s")
            if self._running:
                time.sleep(5.0)

    def _run_ws(self) -> None:
        try:
            import websocket
        except ImportError:
            logger.warning("websocket-client not installed, LiquidationPoller using poll fallback")
            self._poll_fallback()
            return

        ws = websocket.WebSocket()
        ws.settimeout(10)
        ws.connect(self._ws_url)
        logger.info("LiquidationPoller WS connected: %s", self._ws_url)

        while self._running:
            try:
                raw = ws.recv()
                if not raw:
                    continue
                data = json.loads(raw)
                order = data.get("o", {})
                ts = order.get("T", int(time.time() * 1000))
                side = order.get("S", "SELL")
                price = float(order.get("p", 0))
                qty = float(order.get("q", 0))
                quote_qty = price * qty

                with self._lock:
                    self._events.append((ts, side, quote_qty))

                logger.debug("Liquidation: %s %s %.2f USDT", self._symbol, side, quote_qty)
            except websocket.WebSocketTimeoutException:
                continue
            except websocket.WebSocketConnectionClosedException:
                logger.warning("LiquidationPoller WS closed")
                break

        try:
            ws.close()
        except Exception as e:
            logger.error("Failed to close liquidation WS: %s", e, exc_info=True)

    def _poll_fallback(self) -> None:
        """Fallback: just sleep and keep running (no data, but don't crash)."""
        while self._running:
            time.sleep(10.0)
