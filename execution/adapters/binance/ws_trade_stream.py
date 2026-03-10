"""Binance aggTrade WebSocket client.

Mirrors the BinanceDepthStreamClient pattern for trade ticks.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

from _quant_hotpath import rust_parse_agg_trade

from event.tick_types import TradeTickEvent
from execution.adapters.binance.ws_transport import WsTransport

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class TradeStreamConfig:
    ws_base_url: str = "wss://fstream.binance.com/stream"
    recv_timeout_s: float = 5.0


@dataclass(slots=True)
class BinanceTradeStreamClient:
    """Tick-driven WebSocket client for Binance aggTrade stream.

    Usage:
        client = BinanceTradeStreamClient(
            transport=transport,
            streams=("btcusdt@aggTrade",),
        )
        tick = client.step()  # Returns Optional[TradeTickEvent]
    """

    transport: WsTransport
    streams: tuple[str, ...]
    cfg: TradeStreamConfig = TradeStreamConfig()

    _connected_url: Optional[str] = None

    def _build_url(self) -> str:
        base = self.cfg.ws_base_url.rstrip("/")
        stream_param = "/".join(self.streams)
        return f"{base}?streams={stream_param}"

    def connect(self) -> str:
        url = self._build_url()
        self.transport.connect(url)
        self._connected_url = url
        return url

    def close(self) -> None:
        try:
            self.transport.close()
        finally:
            self._connected_url = None

    def step(self) -> Optional[TradeTickEvent]:
        """Receive one message and convert to TradeTickEvent."""
        if not self._connected_url:
            self.connect()

        raw = self.transport.recv(timeout_s=self.cfg.recv_timeout_s)
        if not raw:
            return None

        return self._parse(raw)

    def _parse(self, raw: str) -> Optional[TradeTickEvent]:
        parsed = rust_parse_agg_trade(raw)
        if parsed is None:
            return None

        return TradeTickEvent(
            symbol=parsed["symbol"],
            price=Decimal(parsed["price"]),
            qty=Decimal(parsed["qty"]),
            side=parsed["side"],
            trade_id=parsed["trade_id"],
            ts_ms=parsed["ts_ms"],
            received_at=time.monotonic(),
        )
