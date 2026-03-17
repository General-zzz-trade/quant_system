"""Shared configuration constants for the Bybit alpha runner."""
from __future__ import annotations

from pathlib import Path

MODEL_BASE = Path("models_v8")
INTERVAL = "60"  # Bybit: "60" = 1h
WARMUP_BARS = 800  # Must be > zscore_window(720) + zscore_warmup(180) for full z-score convergence
POLL_INTERVAL = 60  # seconds between checks

# Hard safety limit — no single order can exceed this notional (USD)
# Increase manually as your account grows. Never let code auto-adjust this.
# Demo account: $50K equity → $5K per order (10% of equity)
MAX_ORDER_NOTIONAL = 5000.0

# Default symbols + position sizes
SYMBOL_CONFIG = {
    "BTCUSDT": {"size": 0.001, "model_dir": "BTCUSDT_gate_v2", "max_qty": 1190, "step": 0.001,
                 "use_composite_regime": True},
    "ETHUSDT": {"size": 0.01, "model_dir": "ETHUSDT_gate_v2", "max_qty": 8000, "step": 0.01},
    # 15m alpha: separate model, different interval
    "ETHUSDT_15m": {"size": 0.01, "model_dir": "ETHUSDT_15m", "symbol": "ETHUSDT",
                    "interval": "15", "warmup": 800},
    # SUI 1h alpha (walk-forward 6/7 PASS, Sharpe 1.63, +150%)
    "SUIUSDT": {"size": 10, "model_dir": "SUIUSDT", "max_qty": 330000, "step": 10},
    # AXS 1h alpha (walk-forward 13/17 PASS, Sharpe 1.25, +241%)
    "AXSUSDT": {"size": 5.0, "model_dir": "AXSUSDT", "max_qty": 50000, "step": 0.1},  # min $5 notional → ~4 AXS
}

# Shared cross-symbol signal state for consensus scaling.
# Maps runner_key → current signal (+1, -1, 0). Updated by each AlphaRunner
# after process_bar(). Read by _get_consensus_scale() to adjust sizing.
_consensus_signals: dict[str, int] = {}
