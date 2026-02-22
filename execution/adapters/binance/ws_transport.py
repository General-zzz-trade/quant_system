from __future__ import annotations

from typing import Optional, Protocol


class WsTransport(Protocol):
    """
    可注入 transport，避免 tests 依赖真实网络/第三方库。
    """
    def connect(self, url: str) -> None: ...
    def recv(self, *, timeout_s: Optional[float] = None) -> str: ...
    def close(self) -> None: ...
