# execution/adapters/bitget/ws_client.py
"""Bitget V2 WebSocket market data client using websocket-client."""
from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

import websocket

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class BitgetWsConfig:
    ws_url: str = "wss://ws.bitget.com/v2/ws/public"
    ping_interval: float = 25.0
    reconnect_delay: float = 3.0
    inst_type: str = "USDT-FUTURES"


@dataclass
class BitgetWsMarketStreamClient:
    """WebSocket client for Bitget V2 public market data streams.

    Uses websocket-client WebSocketApp in a background thread.
    Subscribes to kline (candle) channels and dispatches parsed data
    via the on_kline callback.
    """

    symbols: Sequence[str]
    channel: str = "candle1m"
    cfg: BitgetWsConfig = field(default_factory=BitgetWsConfig)
    on_kline: Optional[Callable[[Any], None]] = None

    _ws: Optional[websocket.WebSocketApp] = field(default=None, init=False, repr=False)
    _thread: Optional[threading.Thread] = field(default=None, init=False, repr=False)
    _running: bool = field(default=False, init=False)

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._ws = websocket.WebSocketApp(
            self.cfg.ws_url,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )
        self._thread = threading.Thread(
            target=self._run, name="bitget-ws-market", daemon=True,
        )
        self._thread.start()
        logger.info("BitgetWsMarketStreamClient started (%d symbols)", len(self.symbols))

    def stop(self) -> None:
        self._running = False
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        logger.info("BitgetWsMarketStreamClient stopped")

    def _run(self) -> None:
        while self._running:
            try:
                self._ws.run_forever(
                    ping_interval=int(self.cfg.ping_interval),
                    ping_timeout=10,
                )
            except Exception:
                logger.exception("WS run_forever error")
            if self._running:
                import time
                time.sleep(self.cfg.reconnect_delay)
                logger.info("Reconnecting Bitget WS...")

    def _on_open(self, ws: Any) -> None:
        args = [
            {
                "instType": self.cfg.inst_type,
                "channel": self.channel,
                "instId": sym.upper(),
            }
            for sym in self.symbols
        ]
        sub_msg = json.dumps({"op": "subscribe", "args": args})
        ws.send(sub_msg)
        logger.info("Bitget WS subscribed: %s", sub_msg)

    def _on_message(self, ws: Any, message: str) -> None:
        try:
            payload = json.loads(message)
        except (json.JSONDecodeError, TypeError):
            return

        if not isinstance(payload, dict):
            return

        # Subscription confirmation or pong — skip
        if "event" in payload or "op" in payload:
            return

        arg = payload.get("arg")
        data = payload.get("data")
        if not isinstance(arg, dict) or not isinstance(data, list):
            return

        channel = arg.get("channel", "")
        if not channel.startswith("candle"):
            return

        inst_id = str(arg.get("instId", "")).upper()
        for candle in data:
            if not isinstance(candle, (list, tuple)) or len(candle) < 6:
                continue
            # Bitget candle format: [ts, open, high, low, close, vol, quoteVol, ...]
            from execution.adapters.bitget.kline_processor import BitgetKlineRaw
            try:
                raw = BitgetKlineRaw(
                    symbol=inst_id,
                    ts_ms=int(candle[0]),
                    open=float(candle[1]),
                    high=float(candle[2]),
                    low=float(candle[3]),
                    close=float(candle[4]),
                    volume=float(candle[5]),
                )
            except (ValueError, IndexError):
                logger.warning("Failed to parse Bitget kline: %s", candle)
                continue

            if self.on_kline is not None:
                try:
                    self.on_kline(raw)
                except Exception:
                    logger.exception("on_kline callback error")

    def _on_error(self, ws: Any, error: Any) -> None:
        logger.error("Bitget WS error: %s", error)

    def _on_close(self, ws: Any, close_status_code: Any, close_msg: Any) -> None:
        logger.info("Bitget WS closed: code=%s msg=%s", close_status_code, close_msg)
