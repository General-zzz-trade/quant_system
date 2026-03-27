# execution/adapters/binance/config.py
"""Binance Futures adapter configuration."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from execution.adapters.binance.urls import resolve_binance_urls


@dataclass(frozen=True, slots=True)
class BinanceConfig:
    """Binance USDT-M Futures API connection configuration.

    Attributes:
        api_key: Binance API key.
        api_secret: Binance API secret.
        testnet: Whether to use the testnet endpoint.
        category: Product type -- "linear" (USDT-M futures).
    """
    api_key: str = ""
    api_secret: str = ""
    testnet: bool = True
    category: str = "linear"

    @property
    def base_url(self) -> str:
        return resolve_binance_urls(testnet=self.testnet).rest_base

    @property
    def is_testnet(self) -> bool:
        return self.testnet

    @property
    def is_demo(self) -> bool:
        return self.testnet

    @classmethod
    def from_env(
        cls,
        *,
        api_key: str,
        api_secret: str,
        testnet: bool = True,
        **kw: Any,
    ) -> "BinanceConfig":
        return cls(api_key=api_key, api_secret=api_secret, testnet=testnet, **kw)
