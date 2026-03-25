"""Shared configuration constants for the Bybit alpha runner."""
from __future__ import annotations

import os
from pathlib import Path

MODEL_BASE = Path("models_v8")
INTERVAL = "60"  # Bybit: "60" = 1h
WARMUP_BARS = 800  # Must be > zscore_window(720) + zscore_warmup(180) for full z-score convergence
POLL_INTERVAL = 60  # seconds between checks

# Safety cap: 250% of equity (normal max ~ 150% at 10x leverage).
# 67% headroom above normal. Catches sizing bugs without blocking normal trades.
# Absolute ceiling: $100K regardless of equity.
MAX_ORDER_NOTIONAL_PCT = 2.50
MAX_ORDER_NOTIONAL_FLOOR = 100.0
MAX_ORDER_NOTIONAL_CEILING = 100_000.0


def get_max_order_notional(equity: float) -> float:
    """Dynamic max order notional based on current equity."""
    notional = equity * MAX_ORDER_NOTIONAL_PCT
    return max(MAX_ORDER_NOTIONAL_FLOOR, min(notional, MAX_ORDER_NOTIONAL_CEILING))


# Backward compat: static value used by code that imports MAX_ORDER_NOTIONAL directly
MAX_ORDER_NOTIONAL = 5_000.0  # fallback if equity unknown

# Live/Demo mode detection from BYBIT_BASE_URL
_IS_LIVE = os.environ.get("BYBIT_BASE_URL", "").startswith("https://api.bybit.com")

# Leverage: 3x for live (Kelly half=7x, conservative=3x), 10x for demo
LEVERAGE_LADDER = [
    (0, 3.0 if _IS_LIVE else 10.0),
]


# Default symbols + position sizes
# 2026-03-21: Focused on BTC + ETH only (altcoins removed due to poor liquidity)
SYMBOL_CONFIG = {
    # BTC 1h: dz=1.0, mh=18, maxh=120 (Sharpe 2.34, retrained 2026-03-23)
    "BTCUSDT": {"size": 0.001, "model_dir": "BTCUSDT_gate_v2", "max_qty": 1190, "step": 0.001,
                 "use_composite_regime": True},
    # ETH 1h: dz=0.9, mh=18, maxh=60, long_only (Sharpe 3.92)
    "ETHUSDT": {"size": 0.01, "model_dir": "ETHUSDT_gate_v2", "max_qty": 8000, "step": 0.01},
    # ETH 15m: DISABLED — WF FAIL (Sharpe -1.36)
    "ETHUSDT_15m": {"size": 0.01, "model_dir": "ETHUSDT_15m", "symbol": "ETHUSDT",
                    "interval": "15", "warmup": 800, "step": 0.01},
    # BTC 15m: DISABLED — WF FAIL (Sharpe 0.27)
    "BTCUSDT_15m": {"size": 0.001, "model_dir": "BTCUSDT_15m", "symbol": "BTCUSDT",
                    "interval": "15", "warmup": 800, "step": 0.001},
    # BTC 4h: Strategy H primary, WF Sharpe 3.62 (20/22)
    # warmup=300: zscore_window(180)+zscore_warmup(45)+feature_warmup(28)=253, +margin
    "BTCUSDT_4h": {"size": 0.001, "model_dir": "BTCUSDT_4h", "symbol": "BTCUSDT",
                   "interval": "240", "warmup": 300, "step": 0.001,
                   "use_composite_regime": True},
    # ETH 4h: Strategy H primary, WF Sharpe 4.57 (21/21)
    # warmup=300: zscore_window(180)+zscore_warmup(45)+feature_warmup(28)=253, +margin
    "ETHUSDT_4h": {"size": 0.01, "model_dir": "ETHUSDT_4h", "symbol": "ETHUSDT",
                   "interval": "240", "warmup": 300, "step": 0.01},
}

# Shared cross-symbol signal state for consensus scaling.
# Maps runner_key → current signal (+1, -1, 0). Updated by each AlphaRunner
# after process_bar(). Read by _get_consensus_scale() to adjust sizing.
# Thread safety: CPython GIL guarantees dict[key]=val is atomic. No lock needed
# for simple read/write. If switching to free-threaded Python, add a Lock.
_consensus_signals: dict[str, int] = {}
