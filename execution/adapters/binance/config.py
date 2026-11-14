"""Compatibility shell for trading.adapters.binance.config (migration step 8).

The new config does not read the environment (only trading.infrastructure does). This subclass keeps
the legacy ``BinanceConfig.from_env`` for the runners and scripts that still call it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from trading.adapters.binance.config import BinanceConfig as _BinanceConfig
from trading.infrastructure.settings import binance_config_from_env


@dataclass(frozen=True, slots=True)
class BinanceConfig(_BinanceConfig):
    @classmethod
    def from_env(cls, testnet: bool = True, **kw: Any) -> "BinanceConfig":
        base = binance_config_from_env(testnet=testnet)
        return cls(api_key=base.api_key, api_secret=base.api_secret, testnet=testnet, **kw)


__all__ = ["BinanceConfig"]
