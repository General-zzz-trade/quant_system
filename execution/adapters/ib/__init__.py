# execution/adapters/ib/__init__.py
"""Interactive Brokers venue adapter — multi-asset trading via IB Gateway/TWS.

Supports: Stocks, Options, Futures, Forex, CFDs, Crypto, Bonds.
Runs natively on Linux via IB Gateway (headless) or TWS.
"""
from execution.adapters.ib.adapter import IBAdapter
from execution.adapters.ib.config import IBConfig

__all__ = ["IBAdapter", "IBConfig"]
