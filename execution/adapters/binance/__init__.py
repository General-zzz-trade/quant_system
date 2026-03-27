"""execution.adapters.binance -- Binance USDT-M futures adapter.

Provides REST client, adapter, execution adapter, depth processor, and transport.
"""
from execution.adapters.binance.adapter import BinanceAdapter
from execution.adapters.binance.config import BinanceConfig

__all__ = ["BinanceAdapter", "BinanceConfig"]
