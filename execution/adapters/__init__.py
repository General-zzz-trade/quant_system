"""execution.adapters — Exchange connectivity adapters (Domain 1).

Each sub-package wraps a specific venue API into the ExecutionAdapter protocol:
  bybit/        Bybit V5 REST + WS (PRODUCTION — USDT perpetuals)
  binance/      Binance USDT-M futures (REST + WS + depth)
  hyperliquid/  Hyperliquid REST (USDT perpetuals)
  polymarket/   Polymarket prediction market client
  common/       Shared utilities (decimals, hashing, idempotency, symbols, time)
"""
from execution.adapters.bybit import BybitAdapter, BybitConfig
from execution.adapters.hyperliquid import HyperliquidAdapter, HyperliquidConfig

__all__ = [
    "BybitAdapter",
    "BybitConfig",
    "HyperliquidAdapter",
    "HyperliquidConfig",
]
