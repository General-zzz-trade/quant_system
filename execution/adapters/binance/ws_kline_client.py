# execution/adapters/binance/ws_kline_client.py
"""Binance Futures WebSocket client for real-time kline + ticker data.

Subscribes to kline and miniTicker streams, fires callbacks on confirmed bars.
Compatible with the same on_bar/on_tick interface as BybitWsClient.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any, Callable

import websocket

logger = logging.getLogger(__name__)

# Binance Futures WS endpoints
WS_FUTURES = "wss://fstream.binance.com"
WS_TESTNET = "wss://stream.binancefuture.com"

# Interval mapping: internal format → Binance stream name
_INTERVAL_MAP = {
    "60": "1h",
    "240": "4h",
    "15": "15m",
    "5": "5m",
    "1": "1m",
}


class BinanceWsClient:
    """Binance Futures WebSocket client for kline subscriptions.

    Fires on_bar callback ONLY when a bar is confirmed (closed).
    Fires on_tick with latest price from miniTicker stream.
    Manages reconnection with exponential backoff.
    """

    def __init__(
        self,
        symbols: list[str],
        interval: str = "60",
        on_bar: Callable[[str, dict], None] | None = None,
        on_tick: Callable[[str, float], None] | None = None,
        testnet: bool = True,
    ) -> None:
        self._symbols = [s.lower() for s in symbols]
        self._symbols_upper = [s.upper() for s in symbols]
        self._interval = interval
        self._binance_interval = _INTERVAL_MAP.get(interval, "1h")
        self._on_bar = on_bar
        self._on_tick = on_tick
        self._last_prices: dict[str, float] = {}
        self._last_funding_rates: dict[str, float] = {}
        self._last_bar_ts: dict[str, int] = {}
        self._url = WS_TESTNET if testnet else WS_FUTURES
        self._ws: Any = None
        self._thread: threading.Thread | None = None
        self._running = False
        self._backoff = 1.0
        self._max_backoff = 60.0

    def start(self) -> None:
        """Start WebSocket connection in background thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True,
            name=f"binance-ws-{self._binance_interval}",
        )
        self._thread.start()
        logger.info(
            "Binance WS started: %s symbols=%s interval=%s",
            self._url, self._symbols_upper, self._binance_interval,
        )

    def stop(self) -> None:
        """Stop WebSocket connection."""
        self._running = False
        if self._ws:
            try:
                self._ws.close()
            except Exception:
                pass
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3)
        logger.info("Binance WS stopped")

    def _run_loop(self) -> None:
        """Reconnection loop with exponential backoff."""
        reconnect_count = 0
        while self._running:
            try:
                self._connect_and_listen()
            except Exception as e:
                if not self._running:
                    break
                reconnect_count += 1
                logger.warning(
                    "Binance WS error (reconnect #%d): %s",
                    reconnect_count, e,
                )
            if self._running:
                delay = min(self._backoff, self._max_backoff)
                logger.warning("Binance WS reconnecting in %.0fs", delay)
                time.sleep(delay)
                self._backoff = min(self._backoff * 2, self._max_backoff)

    def _connect_and_listen(self) -> None:
        """Connect to Binance combined stream and process messages."""
        # Build combined stream URL
        # Format: wss://fstream.binance.com/stream?streams=btcusdt@kline_1h/ethusdt@kline_1h/...
        streams = []
        for sym in self._symbols:
            streams.append(f"{sym}@kline_{self._binance_interval}")
            streams.append(f"{sym}@miniTicker")
        stream_path = "/".join(streams)
        url = f"{self._url}/stream?streams={stream_path}"

        self._ws = websocket.WebSocket()
        self._ws.connect(url, timeout=30)
        self._backoff = 1.0  # reset on successful connect

        subscribed = [f"{s}@kline_{self._binance_interval}" for s in self._symbols]
        subscribed += [f"{s}@miniTicker" for s in self._symbols]
        logger.info(
            "Binance WS connected, subscribed: %s", subscribed,
        )

        while self._running:
            try:
                raw = self._ws.recv()
                if not raw:
                    continue
                data = json.loads(raw)
                self._handle_message(data)
            except websocket.WebSocketTimeoutException:
                continue
            except websocket.WebSocketConnectionClosedException:
                logger.warning("Binance WS connection closed")
                break

    def _handle_message(self, msg: dict) -> None:
        """Route incoming WS message to appropriate handler."""
        # Combined stream format: {"stream": "btcusdt@kline_1h", "data": {...}}
        stream = msg.get("stream", "")
        data = msg.get("data", msg)  # fallback to msg itself

        if "@kline_" in stream:
            self._handle_kline(data)
        elif "@miniTicker" in stream:
            self._handle_ticker(data)

    def _handle_kline(self, data: dict) -> None:
        """Process kline message — fire on_bar only for confirmed bars."""
        k = data.get("k", {})
        if not k:
            return

        is_closed = k.get("x", False)
        if not is_closed:
            return  # skip unconfirmed bars

        symbol = k.get("s", "").upper()  # "BTCUSDT"
        bar_ts = int(k.get("t", 0))  # open time in ms

        # Deduplicate
        if bar_ts <= self._last_bar_ts.get(symbol, 0):
            return
        self._last_bar_ts[symbol] = bar_ts

        bar = {
            "time": bar_ts // 1000,
            "open": float(k.get("o", 0)),
            "high": float(k.get("h", 0)),
            "low": float(k.get("l", 0)),
            "close": float(k.get("c", 0)),
            "volume": float(k.get("v", 0)),
            "turnover": float(k.get("q", 0)),  # quote volume
            "confirm": True,
            "interval": k.get("i", self._binance_interval),
        }

        # Map Binance interval back to internal format
        interval_map_rev = {v: k for k, v in _INTERVAL_MAP.items()}
        bar["interval"] = interval_map_rev.get(
            bar["interval"], str(self._interval)
        )

        logger.info(
            "WS bar: %s interval=%s close=$%.2f vol=%.0f",
            symbol, bar["interval"], bar["close"], bar["volume"],
        )

        if self._on_bar:
            try:
                self._on_bar(symbol, bar)
            except Exception:
                logger.exception("on_bar callback error for %s", symbol)

    def _handle_ticker(self, data: dict) -> None:
        """Process miniTicker — update last price and fire on_tick."""
        symbol = data.get("s", "").upper()
        price = float(data.get("c", 0))  # close price
        if price <= 0:
            return

        self._last_prices[symbol] = price

        if self._on_tick:
            try:
                self._on_tick(symbol, price)
            except Exception:
                pass

    def get_last_price(self, symbol: str) -> float:
        """Get last known price from ticker stream."""
        return self._last_prices.get(symbol, 0.0)

    def get_last_funding_rate(self, symbol: str) -> float:
        """Get last known funding rate (not available via miniTicker)."""
        return self._last_funding_rates.get(symbol, float("nan"))
