"""Bitget V2 API configuration."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BitgetConfig:
    """Bitget V2 API connection configuration."""

    api_key: str = ""
    api_secret: str = ""
    passphrase: str = ""  # Bitget requires passphrase (set when creating API key)
    base_url: str = "https://api.bitget.com"
    recv_window: str = "20000"
    product_type: str = "USDT-FUTURES"  # "USDT-FUTURES", "COIN-FUTURES", "USDC-FUTURES"
    margin_coin: str = "USDT"

    @property
    def is_demo(self) -> bool:
        return "demo" in self.base_url.lower()

    @classmethod
    def demo(cls, *, api_key: str, api_secret: str, passphrase: str, **kw) -> BitgetConfig:
        """Demo trading — same base URL, uses demo API keys."""
        return cls(api_key=api_key, api_secret=api_secret,
                   passphrase=passphrase, **kw)

    @classmethod
    def live(cls, *, api_key: str, api_secret: str, passphrase: str, **kw) -> BitgetConfig:
        return cls(api_key=api_key, api_secret=api_secret,
                   passphrase=passphrase,
                   base_url="https://api.bitget.com", **kw)
