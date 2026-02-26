# execution/adapters/binance/ws_market_stream_um.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from event.types import MarketEvent
from execution.adapters.binance.kline_processor import KlineProcessor
from execution.adapters.binance.ws_transport import WsTransport


@dataclass(frozen=True, slots=True)
class MarketStreamConfig:
    ws_base_url: str = "wss://fstream.binance.com/stream"
    recv_timeout_s: float = 5.0


@dataclass(slots=True)
class BinanceUmMarketStreamWsClient:
    """Tick-driven WebSocket client for Binance UM kline market data.

    Mirrors BinanceUmUserStreamWsClient pattern:
    - connect() establishes WebSocket
    - step() receives one message and returns Optional[MarketEvent]
    - close() tears down connection
    """

    transport: WsTransport
    processor: KlineProcessor
    streams: tuple[str, ...]      # e.g. ("btcusdt@kline_1m",)
    cfg: MarketStreamConfig = MarketStreamConfig()

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

    def step(self) -> Optional[MarketEvent]:
        """Execute one step: recv one message, convert to MarketEvent if applicable.

        Returns MarketEvent for closed klines, None otherwise (timeout, partial, non-kline).
        Auto-connects on first call.
        """
        if not self._connected_url:
            self.connect()

        raw = self.transport.recv(timeout_s=self.cfg.recv_timeout_s)
        if not raw:
            return None

        return self.processor.process_raw(raw)
