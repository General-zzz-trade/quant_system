from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from execution.adapters.binance.ws_transport import WsTransport


@dataclass(slots=True)
class WebsocketClientTransport(WsTransport):
    """
    同步 transport：依赖 websocket-client（Windows 上最省事）
    pip install websocket-client
    """
    _ws: any = None

    def connect(self, url: str) -> None:
        try:
            import websocket  # type: ignore
        except Exception as e:
            raise RuntimeError("missing dependency: websocket-client. Run: pip install websocket-client") from e

        import socket
        self._ws = websocket.create_connection(url, timeout=10)
        # Low-latency: disable Nagle's algorithm
        sock = self._ws.sock
        if sock is not None:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def recv(self, *, timeout_s: Optional[float] = None) -> str:
        if self._ws is None:
            return ""
        if timeout_s is not None:
            self._ws.settimeout(timeout_s)
        try:
            msg = self._ws.recv()
            return msg if isinstance(msg, str) else (msg.decode("utf-8", errors="replace") if msg else "")
        except Exception:
            return ""

    def close(self) -> None:
        try:
            if self._ws is not None:
                self._ws.close()
        finally:
            self._ws = None
