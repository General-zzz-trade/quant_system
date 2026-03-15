# execution/adapters/bybit/config.py
"""Bybit adapter configuration."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class BybitConfig:
    """Bybit V5 API connection configuration.

    Attributes:
        api_key: Bybit API key.
        api_secret: Bybit API secret.
        base_url: API endpoint — demo/testnet/live.
        recv_window: Request timeout window in milliseconds.
        category: Product type — "linear" (USDT perp), "inverse", "spot".
        account_type: Account type for balance queries — "UNIFIED" or "CONTRACT".
    """
    api_key: str = ""
    api_secret: str = ""
    base_url: str = "https://api-demo.bybit.com"
    recv_window: str = "20000"
    category: str = "linear"
    account_type: str = "UNIFIED"

    @property
    def is_demo(self) -> bool:
        return "demo" in self.base_url

    @property
    def is_testnet(self) -> bool:
        return "testnet" in self.base_url

    @classmethod
    def demo(cls, *, api_key: str, api_secret: str, **kw: Any) -> "BybitConfig":
        return cls(api_key=api_key, api_secret=api_secret,
                   base_url="https://api-demo.bybit.com", **kw)

    @classmethod
    def testnet(cls, *, api_key: str, api_secret: str, **kw: Any) -> "BybitConfig":
        return cls(api_key=api_key, api_secret=api_secret,
                   base_url="https://api-testnet.bybit.com", **kw)

    @classmethod
    def live(cls, *, api_key: str, api_secret: str, **kw: Any) -> "BybitConfig":
        return cls(api_key=api_key, api_secret=api_secret,
                   base_url="https://api.bybit.com", **kw)
