"""Bitget V2 WebSocket client for real-time kline and ticker data."""
from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any, Callable

logger = logging.getLogger(__name__)


class BitgetWsClient:
    """WebSocket client for Bitget public linear futures data.

    Subscribes to kline (confirmed bars) and ticker (real-time price) streams.
    Public endpoint — no authentication required.

    WS URL: wss://ws.bitget.com/v2/ws/public
    Topics:
      - kline: {"instType":"USDT-FUTURES","channel":"candle1H","instId":"ETHUSDT"}
      - ticker: {"instType":"USDT-FUTURES","channel":"ticker","instId":"ETHUSDT"}
    """

    _GRANULARITY_MAP = {
        "1": "candle1m", "5": "candle5m", "15": "candle15m",
        "30": "candle30m", "60": "candle1H", "240": "candle4H",
        "360": "candle6H", "720": "candle12H", "D": "candle1D",
    }

    def __init__(
        self,
        symbols: list[str],
        interval: str = "60",
        on_bar: Callable[[str, dict], None] | None = None,
        on_tick: Callable[[str, float], None] | None = None,
        product_type: str = "USDT-FUTURES",
    ) -> None:
        self._symbols = symbols
        self._interval = interval
        self._on_bar = on_bar
        self._on_tick = on_tick
        self._product_type = product_type
        self._url = "wss://ws.bitget.com/v2/ws/public"
        self._running = False
        self._ws: Any = None
        self._thread: threading.Thread | None = None
        self._backoff = 1.0
        self._last_bar_ts: dict[str, int] = {}

    def start(self) -> None:
        """Start WS connection in background thread."""
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        channel = self._GRANULARITY_MAP.get(self._interval, f"candle{self._interval}m")
        logger.info(
            "Bitget WS started: %s symbols=%s interval=%s",
            self._url, self._symbols, channel,
        )

    def stop(self) -> None:
        self._running = False
        if self._ws:
            try:
                self._ws.close()
            except Exception:
                pass

    def _run_loop(self) -> None:
        try:
            import websocket
        except ImportError:
            logger.error("websocket-client not installed: pip install websocket-client")
            return

        while self._running:
            try:
                self._ws = websocket.WebSocketApp(
                    self._url,
                    on_message=self._on_message,
                    on_open=self._on_open,
                    on_error=self._on_error,
                    on_close=self._on_close,
                )
                self._ws.run_forever(ping_interval=20, ping_timeout=10)
            except Exception as e:
                logger.error("Bitget WS error: %s", e)

            if self._running:
                logger.info("Bitget WS reconnecting in %.0fs...", self._backoff)
                time.sleep(self._backoff)
                self._backoff = min(self._backoff * 2, 60)

    def _on_open(self, ws: Any) -> None:
        self._backoff = 1.0
        channel = self._GRANULARITY_MAP.get(self._interval, f"candle{self._interval}m")

        # Subscribe to klines + tickers for each symbol
        args = []
        for sym in self._symbols:
            args.append({
                "instType": self._product_type,
                "channel": channel,
                "instId": sym,
            })
            if self._on_tick:
                args.append({
                    "instType": self._product_type,
                    "channel": "ticker",
                    "instId": sym,
                })

        sub_msg = json.dumps({"op": "subscribe", "args": args})
        ws.send(sub_msg)
        channels = [a["channel"] for a in args]
        logger.info("Bitget WS connected, subscribed: %s", channels)

    def _on_message(self, ws: Any, msg: str) -> None:
        try:
            data = json.loads(msg)
        except json.JSONDecodeError:
            return

        action = data.get("action", "")
        arg = data.get("arg", {})
        channel = arg.get("channel", "")
        inst_id = arg.get("instId", "")

        # Handle ticker
        if "ticker" in channel and self._on_tick:
            for item in data.get("data", []):
                price = float(item.get("lastPr", item.get("last", 0)))
                if price > 0:
                    self._on_tick(inst_id, price)
            return

        # Handle kline
        if "candle" in channel and self._on_bar:
            for item in data.get("data", []):
                if not isinstance(item, list) or len(item) < 6:
                    continue
                ts = int(item[0])
                # Bitget: last element is "1" for confirmed bar
                confirmed = str(item[-1]) == "1" if len(item) > 6 else True

                if not confirmed:
                    continue

                # Dedup
                if self._last_bar_ts.get(inst_id, 0) >= ts:
                    continue
                self._last_bar_ts[inst_id] = ts

                bar = {
                    "time": ts // 1000,
                    "open": float(item[1]),
                    "high": float(item[2]),
                    "low": float(item[3]),
                    "close": float(item[4]),
                    "volume": float(item[5]),
                }
                logger.info("WS bar: %s close=$%.2f vol=%.0f",
                            inst_id, bar["close"], bar["volume"])
                self._on_bar(inst_id, bar)

    def _on_error(self, ws: Any, error: Any) -> None:
        logger.warning("Bitget WS error: %s", error)

    def _on_close(self, ws: Any, code: Any = None, reason: Any = None) -> None:
        logger.info("Bitget WS closed: code=%s reason=%s", code, reason)
