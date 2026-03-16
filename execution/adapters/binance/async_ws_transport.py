# execution/adapters/binance/async_ws_transport.py
"""Async WebSocket transport using the websockets library."""
from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator, Optional

import websockets
from websockets.asyncio.client import ClientConnection

logger = logging.getLogger(__name__)


class AsyncWsTransport:
    """Async WebSocket transport.

    Implements an async counterpart to the sync WsTransport protocol.
    Uses the modern websockets library for async connections.
    """

    def __init__(
        self,
        *,
        ping_interval: float = 20.0,
        ping_timeout: float = 10.0,
        close_timeout: float = 5.0,
    ) -> None:
        self._ping_interval = ping_interval
        self._ping_timeout = ping_timeout
        self._close_timeout = close_timeout
        self._ws: Optional[ClientConnection] = None
        self._url: str = ""

    @property
    def connected(self) -> bool:
        return self._ws is not None

    async def connect(self, url: str) -> None:
        self._url = url
        self._ws = await websockets.connect(
            url,
            ping_interval=self._ping_interval,
            ping_timeout=self._ping_timeout,
            close_timeout=self._close_timeout,
        )
        logger.info("WS connected: %s", url)

    async def recv(self, *, timeout_s: Optional[float] = None) -> str:
        if self._ws is None:
            raise RuntimeError("WebSocket not connected")
        try:
            msg = await asyncio.wait_for(self._ws.recv(), timeout=timeout_s)
            return str(msg)
        except asyncio.TimeoutError:
            raise TimeoutError("WebSocket recv timed out")

    async def send(self, data: str) -> None:
        if self._ws is None:
            raise RuntimeError("WebSocket not connected")
        await self._ws.send(data)

    async def close(self) -> None:
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception as e:
                logger.error("Failed to close async WS connection: %s", e, exc_info=True)
            self._ws = None
            logger.info("WS closed: %s", self._url)

    async def messages(self) -> AsyncIterator[str]:
        """Iterate over incoming messages until connection closes."""
        if self._ws is None:
            raise RuntimeError("WebSocket not connected")
        async for msg in self._ws:
            yield str(msg)
