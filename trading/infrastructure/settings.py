"""Configuration read from the environment. This package is the only place that reads it."""
from __future__ import annotations

import os

from trading.adapters.binance.config import BinanceConfig


def binance_config_from_env(*, testnet: bool) -> BinanceConfig:
    """API credentials for the chosen environment; missing credentials are a start-up error."""
    if testnet:
        key = os.environ.get("BINANCE_TESTNET_API_KEY", "")
        secret = os.environ.get("BINANCE_TESTNET_API_SECRET", "")
    else:
        key = os.environ.get("BINANCE_API_KEY", "")
        secret = os.environ.get("BINANCE_API_SECRET", "")
    if not key or not secret:
        raise RuntimeError("Binance API key/secret not set in environment")
    return BinanceConfig(api_key=key, api_secret=secret, testnet=testnet)
