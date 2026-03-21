"""Shared configuration constants for the Bybit alpha runner."""
from __future__ import annotations

from pathlib import Path

MODEL_BASE = Path("models_v8")
INTERVAL = "60"  # Bybit: "60" = 1h
WARMUP_BARS = 800  # Must be > zscore_window(720) + zscore_warmup(180) for full z-score convergence
POLL_INTERVAL = 60  # seconds between checks

# Hard safety limit — no single order can exceed this notional (USD)
# Increase manually as your account grows. Never let code auto-adjust this.
# Demo: $5,000 (enough for 10x on $35K account with 45% per-symbol cap)
# Real money: start at $500, raise gradually as you gain confidence.
MAX_ORDER_NOTIONAL = 5_000.0

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
}

# Shared cross-symbol signal state for consensus scaling.
# Maps runner_key → current signal (+1, -1, 0). Updated by each AlphaRunner
# after process_bar(). Read by _get_consensus_scale() to adjust sizing.
# Thread safety: CPython GIL guarantees dict[key]=val is atomic. No lock needed
# for simple read/write. If switching to free-threaded Python, add a Lock.
_consensus_signals: dict[str, int] = {}
