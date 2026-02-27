"""Binance depth data WebSocket client.

Mirrors the BinanceUmMarketStreamWsClient pattern for order book data.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from execution.adapters.binance.depth_processor import DepthProcessor, OrderBookSnapshot
from execution.adapters.binance.ws_transport import WsTransport


@dataclass(frozen=True, slots=True)
class DepthStreamConfig:
    ws_base_url: str = "wss://fstream.binance.com/stream"
    recv_timeout_s: float = 5.0


@dataclass(slots=True)
class BinanceDepthStreamClient:
    """Tick-driven WebSocket client for Binance order book depth data.

    Usage:
        client = BinanceDepthStreamClient(
            transport=transport,
            processor=DepthProcessor(),
            streams=("btcusdt@depth20@100ms",),
        )
        snapshot = client.step()  # Returns Optional[OrderBookSnapshot]
    """

    transport: WsTransport
    processor: DepthProcessor
    streams: tuple[str, ...]
    cfg: DepthStreamConfig = DepthStreamConfig()

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

    def step(self) -> Optional[OrderBookSnapshot]:
        """Receive one message and convert to OrderBookSnapshot."""
        if not self._connected_url:
            self.connect()

        raw = self.transport.recv(timeout_s=self.cfg.recv_timeout_s)
        if not raw:
            return None

        return self.processor.process_raw(raw)
