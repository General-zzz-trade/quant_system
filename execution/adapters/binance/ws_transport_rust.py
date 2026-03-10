"""Rust-accelerated WS transport — GIL-free recv via RustWsClient.

Drop-in replacement for WebsocketClientTransport. Uses tokio-tungstenite
for async WS and crossbeam channel for message passing. The recv() call
releases the GIL while blocking on the channel.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from _quant_hotpath import RustWsClient

from execution.adapters.binance.ws_transport import WsTransport


@dataclass(slots=True)
class RustWsTransport(WsTransport):
    """Rust WebSocket transport with GIL-free recv."""

    buffer_size: int = 4096
    _client: Optional[RustWsClient] = field(default=None, init=False)
    _url: Optional[str] = field(default=None, init=False)

    def connect(self, url: str) -> None:
        if self._client is not None:
            self.close()
        self._client = RustWsClient(buffer_size=self.buffer_size)
        self._url = url
        self._client.connect(url)

    def recv(self, *, timeout_s: Optional[float] = None) -> str:
        if self._client is None:
            return ""
        timeout_ms = int((timeout_s or 5.0) * 1000)
        result = self._client.recv(timeout_ms=timeout_ms)
        return result if result is not None else ""

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
            self._url = None
