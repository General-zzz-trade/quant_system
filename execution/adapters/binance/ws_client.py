# execution/adapters/binance/ws_client.py
"""WebSocket client configuration for Binance user data stream."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True, slots=True)
class WsConfig:
    """WebSocket 连接配置。"""
    base_url: str = "wss://fstream.binance.com"
    ping_interval_s: float = 30.0
    pong_timeout_s: float = 10.0
    max_reconnect_attempts: int = 10
    reconnect_delay_s: float = 1.0


@dataclass(frozen=True, slots=True)
class WsSubscription:
    """WebSocket 订阅信息。"""
    stream_name: str
    listen_key: Optional[str] = None

    @property
    def url(self) -> str:
        if self.listen_key:
            return f"wss://fstream.binance.com/ws/{self.listen_key}"
        return f"wss://fstream.binance.com/ws/{self.stream_name}"
