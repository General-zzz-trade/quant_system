# execution/adapters/bybit/ws_client.py
"""Bybit V5 WebSocket client for real-time kline data.

Subscribes to kline channel and fires callback on confirmed (closed) bars.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any, Callable

import websocket

logger = logging.getLogger(__name__)

# Bybit public WS endpoints
WS_PUBLIC = "wss://stream.bybit.com/v5/public/linear"
WS_DEMO = "wss://stream-demo.bybit.com/v5/public/linear"


class BybitWsClient:
    """Bybit WebSocket client for kline subscriptions.

    Fires on_bar callback ONLY when a bar is confirmed (closed).
    Manages reconnection with exponential backoff.
    """

    def __init__(
        self,
        symbols: list[str],
        interval: str = "60",
        on_bar: Callable[[str, dict], None] | None = None,
        on_tick: Callable[[str, float], None] | None = None,
        demo: bool = True,
    ) -> None:
        self._symbols = symbols
        self._interval = interval
        self._on_bar = on_bar
        self._on_tick = on_tick
        self._last_prices: dict[str, float] = {}
        # Public market data always uses production WS (same data for demo/live)
        self._url = WS_PUBLIC
        self._ws: Any = None
        self._thread: threading.Thread | None = None
        self._running = False
        self._backoff = 1.0
        self._max_backoff = 60.0
        self._last_bar_ts: dict[str, int] = {}
        self._last_message_time: float = 0.0
        self._reconnect_count: int = 0

    def start(self) -> None:
        """Start WebSocket connection in background thread."""
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True,
                                        name="bybit-ws")
        self._thread.start()
        logger.info("Bybit WS started: %s symbols=%s interval=%s",
                     self._url, self._symbols, self._interval)

    def stop(self) -> None:
        """Stop WebSocket connection."""
        self._running = False
        if self._ws:
            self._ws.close()
        logger.info("Bybit WS stopped")

    def _run_loop(self) -> None:
        """Connection loop with reconnection."""
        while self._running:
            try:
                self._connect()
            except Exception:
                logger.exception("Bybit WS connection error")

            if not self._running:
                break

            logger.warning("Bybit WS reconnecting in %.0fs", self._backoff)
            time.sleep(self._backoff)
            self._backoff = min(self._backoff * 2, self._max_backoff)

    def _connect(self) -> None:
        """Single WebSocket connection lifecycle."""
        self._ws = websocket.WebSocketApp(
            self._url,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )
        self._ws.run_forever(ping_interval=20, ping_timeout=10)

    def _on_open(self, ws: Any) -> None:
        topics = [f"kline.{self._interval}.{s}" for s in self._symbols]
        if self._on_tick:
            topics += [f"tickers.{s}" for s in self._symbols]
        ws.send(json.dumps({"op": "subscribe", "args": topics}))
        self._backoff = 1.0  # reset on successful connect
        if self._reconnect_count > 0:
            logger.info("Bybit WS reconnected (after %d retries), subscribed: %s",
                         self._reconnect_count, topics)
            self._reconnect_count = 0
        else:
            logger.info("Bybit WS connected, subscribed: %s", topics)

    @property
    def last_message_time(self) -> float:
        """Timestamp of last received WS message (time.time() epoch)."""
        return self._last_message_time

    @property
    def seconds_since_last_message(self) -> float:
        """Seconds elapsed since last WS message. Returns inf if never received."""
        if self._last_message_time == 0.0:
            return float("inf")
        return time.time() - self._last_message_time

    def _on_message(self, ws: Any, msg: str) -> None:
        self._last_message_time = time.time()
        try:
            data = json.loads(msg)
        except json.JSONDecodeError:
            return

        topic = data.get("topic", "")

        # Handle ticker updates (real-time price for stop-loss)
        if "tickers" in topic and "data" in data:
            td = data["data"]
            symbol = td.get("symbol", "")
            last = td.get("lastPrice")
            if last and last != "?":
                price = float(last)
                self._last_prices[symbol] = price
                if self._on_tick:
                    try:
                        self._on_tick(symbol, price)
                    except Exception:
                        pass  # non-fatal
            return

        if "kline" not in topic or "data" not in data:
            return

        for kline in data["data"]:
            confirm = kline.get("confirm", False)
            if not confirm:
                continue  # skip unconfirmed (still forming) bars

            # Extract symbol + interval from topic: "kline.60.ETHUSDT" → interval=60, symbol=ETHUSDT
            parts = topic.split(".")
            symbol = parts[2] if len(parts) >= 3 else ""
            ws_interval = parts[1] if len(parts) >= 2 else str(self._interval)

            bar_ts = int(kline.get("start", 0))

            # Deduplicate: same bar might arrive multiple times
            if bar_ts <= self._last_bar_ts.get(symbol, 0):
                continue
            self._last_bar_ts[symbol] = bar_ts

            bar = {
                "time": bar_ts // 1000,
                "open": float(kline.get("open", 0)),
                "high": float(kline.get("high", 0)),
                "low": float(kline.get("low", 0)),
                "close": float(kline.get("close", 0)),
                "volume": float(kline.get("volume", 0)),
                "turnover": float(kline.get("turnover", 0)),
                "confirm": True,
                "interval": ws_interval,
            }

            logger.info("WS bar: %s close=$%.2f vol=%.0f", symbol, bar["close"], bar["volume"])

            if self._on_bar:
                try:
                    self._on_bar(symbol, bar)
                except Exception:
                    logger.exception("on_bar callback error for %s", symbol)

    def get_last_price(self, symbol: str) -> float:
        """Get last known price from ticker stream (0 if not yet received)."""
        return self._last_prices.get(symbol, 0.0)

    def _on_error(self, ws: Any, error: Any) -> None:
        self._reconnect_count += 1
        logger.warning("Bybit WS error (reconnect #%d): %s", self._reconnect_count, error)

    def _on_close(self, ws: Any, close_status: Any, close_msg: Any) -> None:
        logger.info("Bybit WS closed (reconnect #%d): %s %s",
                     self._reconnect_count, close_status, close_msg)
