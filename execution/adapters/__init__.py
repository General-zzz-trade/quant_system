"""execution.adapters — Exchange connectivity adapters.

Each sub-package wraps a specific venue API into the ExecutionAdapter protocol:
  okx/      OKX SWAP (PRODUCTION)
  binance/  Binance USDT-M futures (testnet demo)
  bybit/    Bybit V5 REST + WS
  common/   Shared utilities
"""
from execution.adapters.bybit import BybitAdapter, BybitConfig

__all__ = [
    "BybitAdapter",
    "BybitConfig",
]
