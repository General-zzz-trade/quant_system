# execution/sim/cost_constants.py
"""Unified trading cost constants — single source of truth.

ALL backtest scripts MUST import costs from here, not define their own.
Values calibrated from Bybit/Binance USDT perpetual tier-0 fees.

To update: change values here, all scripts automatically use new values.
"""

# Fee structure (per side)
MAKER_FEE_BPS = 2.0     # 2 bps maker (Binance/Bybit VIP0)
TAKER_FEE_BPS = 4.0     # 4 bps taker
AVERAGE_FEE_BPS = 4.0   # assume taker (conservative)

# Slippage (per side, base — increases with size)
BASE_SLIPPAGE_BPS = 1.0  # 1 bp for small orders (<$10K notional)

# Half-spread cost (per side)
SPREAD_BPS = 1.0          # ~1 bp for ETH/BTC top-of-book

# Combined round-trip cost (open + close)
COST_PER_TRADE = (AVERAGE_FEE_BPS + BASE_SLIPPAGE_BPS + SPREAD_BPS) / 10000  # 6 bps = 0.0006

# Shorthand for backward compatibility
FEE_BPS = AVERAGE_FEE_BPS / 10000      # 0.0004
SLIPPAGE_BPS = BASE_SLIPPAGE_BPS / 10000  # 0.0001

# Funding rate (applied per 8h settlement)
DEFAULT_FUNDING_RATE = 0.0001  # 1 bp per 8h (fallback when no data)
