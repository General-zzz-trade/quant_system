"""Shared configuration constants for the Bybit alpha runner."""
from __future__ import annotations

from pathlib import Path

MODEL_BASE = Path("models_v8")
INTERVAL = "60"  # Bybit: "60" = 1h
WARMUP_BARS = 800  # Must be > zscore_window(720) + zscore_warmup(180) for full z-score convergence
POLL_INTERVAL = 60  # seconds between checks

# Max order notional as fraction of equity (dynamic).
# Actual limit = equity × MAX_ORDER_NOTIONAL_PCT
# At $35K equity: 0.20 × $35K = $7K per order (matches cap=0.15 × lev=10 ≈ $5.3K for BTC)
# Hard floor: $100 (never go below this even if equity is tiny)
# Hard ceiling: $100K (absolute max regardless of equity)
MAX_ORDER_NOTIONAL_PCT = 0.20  # 20% of equity per order
MAX_ORDER_NOTIONAL_FLOOR = 100.0
MAX_ORDER_NOTIONAL_CEILING = 100_000.0


def get_max_order_notional(equity: float) -> float:
    """Dynamic max order notional based on current equity."""
    notional = equity * MAX_ORDER_NOTIONAL_PCT
    return max(MAX_ORDER_NOTIONAL_FLOOR, min(notional, MAX_ORDER_NOTIONAL_CEILING))


# Backward compat: static value used by code that imports MAX_ORDER_NOTIONAL directly
MAX_ORDER_NOTIONAL = 5_000.0  # fallback if equity unknown

# Default symbols + position sizes
# 2026-03-21: Focused on BTC + ETH only (altcoins removed due to poor liquidity)
SYMBOL_CONFIG = {
    # BTC: optimized dz=1.0, mh=24, maxh=144, monthly-gate (Sharpe 4.37, 20/22 PASS)
    "BTCUSDT": {"size": 0.001, "model_dir": "BTCUSDT_gate_v2", "max_qty": 1190, "step": 0.001,
                 "use_composite_regime": True},
    # ETH: production dz=0.4, mh=18 (Sharpe 4.67, 17/21 PASS)
    "ETHUSDT": {"size": 0.01, "model_dir": "ETHUSDT_gate_v2", "max_qty": 8000, "step": 0.01},
    # 15m alpha: separate model, different interval
    "ETHUSDT_15m": {"size": 0.01, "model_dir": "ETHUSDT_15m", "symbol": "ETHUSDT",
                    "interval": "15", "warmup": 800, "step": 0.01},
    # BTC 15m alpha: cross-market features unlock (Sharpe 17.59 WF validated)
    "BTCUSDT_15m": {"size": 0.001, "model_dir": "BTCUSDT_15m", "symbol": "BTCUSDT",
                    "interval": "15", "warmup": 800, "step": 0.001},
    # BTC 4h alpha (Strategy H primary): IC=0.29, Sharpe 6.34, 22/22 WF PASS, cap=0.15
    "BTCUSDT_4h": {"size": 0.001, "model_dir": "BTCUSDT_4h", "symbol": "BTCUSDT",
                   "interval": "240", "warmup": 200, "step": 0.001,
                   "use_composite_regime": True},
    # ETH 4h alpha (Strategy H primary): IC=0.42, Sharpe 5.92, 21/21 WF PASS, cap=0.10
    "ETHUSDT_4h": {"size": 0.01, "model_dir": "ETHUSDT_4h", "symbol": "ETHUSDT",
                   "interval": "240", "warmup": 200, "step": 0.01},
}

# Shared cross-symbol signal state for consensus scaling.
# Maps runner_key → current signal (+1, -1, 0). Updated by each AlphaRunner
# after process_bar(). Read by _get_consensus_scale() to adjust sizing.
# Thread safety: CPython GIL guarantees dict[key]=val is atomic. No lock needed
# for simple read/write. If switching to free-threaded Python, add a Lock.
_consensus_signals: dict[str, int] = {}
