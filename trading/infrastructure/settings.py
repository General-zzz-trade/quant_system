"""Configuration read from the environment. This package is the only place that reads it."""
from __future__ import annotations

import os
from collections.abc import Mapping

from trading.adapters.binance.adapter import DEFAULT_LEVERAGE, MARGIN_TYPES, MAX_LEVERAGE
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


def exchange_settings(env: Mapping[str, str] | None = None) -> tuple[int, str | None]:
    """(exchange leverage setting, margin type or None) from BINANCE_EXCHANGE_LEVERAGE / BINANCE_MARGIN_TYPE.

    Defaults: leverage 3, margin type left as it is. These change margin per position and
    the liquidation distance only; actual exposure stays capped at 3x equity."""
    env = os.environ if env is None else env
    raw = env.get("BINANCE_EXCHANGE_LEVERAGE", "").strip()
    leverage = int(raw) if raw else DEFAULT_LEVERAGE
    if not 1 <= leverage <= MAX_LEVERAGE:
        raise RuntimeError(f"BINANCE_EXCHANGE_LEVERAGE must be in [1, {MAX_LEVERAGE}]")
    margin = env.get("BINANCE_MARGIN_TYPE", "").strip().upper() or None
    if margin is not None and margin not in MARGIN_TYPES:
        raise RuntimeError(f"BINANCE_MARGIN_TYPE must be one of {sorted(MARGIN_TYPES)}")
    return leverage, margin
