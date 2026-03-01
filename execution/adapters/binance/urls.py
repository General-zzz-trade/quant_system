# execution/adapters/binance/urls.py
"""Binance Futures URL registry — single source of truth for prod vs testnet."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BinanceURLs:
    rest_base: str
    ws_market_stream: str
    ws_user_stream: str


_PRODUCTION = BinanceURLs(
    rest_base="https://fapi.binance.com",
    ws_market_stream="wss://fstream.binance.com/stream",
    ws_user_stream="wss://fstream.binance.com/ws",
)

_TESTNET = BinanceURLs(
    rest_base="https://testnet.binancefuture.com",
    ws_market_stream="wss://stream.binancefuture.com/stream",
    ws_user_stream="wss://stream.binancefuture.com/ws",
)


def resolve_binance_urls(testnet: bool = False) -> BinanceURLs:
    """Return the appropriate URL set for production or testnet."""
    return _TESTNET if testnet else _PRODUCTION
