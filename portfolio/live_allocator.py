"""Live portfolio allocator — bridges portfolio/allocator.py to live trading.

Provides periodic target weight recomputation and integrates with the
existing RiskAggregator (Direction 19).

Usage in live_runner.py:
    allocator = LivePortfolioAllocator(config=LiveAllocatorConfig(...))
    # On each decision cycle:
    targets = allocator.compute_targets(symbols, account, prices, signals)
    # targets is an AllocationPlan with per-symbol TargetPosition
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class LiveAllocatorConfig:
    """Configuration for live portfolio allocation."""
    # Max gross leverage across all symbols
    max_gross_leverage: float = 3.0
    # Max net leverage (long - short)
    max_net_leverage: float = 1.0
    # Max weight per symbol
    max_concentration: float = 0.4
    # Rebalance interval in seconds
    rebalance_interval_sec: float = 3600.0
    # Minimum rebalance delta (% of equity) to trigger trades
    min_rebalance_pct: float = 0.5
    # Enable vol-targeting at portfolio level
    vol_target: Optional[float] = None


@dataclass
class LivePortfolioAllocator:
    """Manages cross-asset allocation constraints for live trading.

    Sits between the decision layer (per-symbol signals) and the
    execution layer. Ensures that aggregate positions respect
    portfolio-level constraints (leverage, concentration).
    """

    config: LiveAllocatorConfig = field(default_factory=LiveAllocatorConfig)
    _last_rebalance_ts: float = field(default=0.0, init=False)
    _target_weights: Dict[str, float] = field(default_factory=dict)

    def should_rebalance(self) -> bool:
        """Check if enough time has passed for a rebalance."""
        now = time.time()
        return now - self._last_rebalance_ts >= self.config.rebalance_interval_sec

    def compute_signal_weights(
        self,
        symbols: Sequence[str],
        signals: Dict[str, float],
    ) -> Dict[str, float]:
        """Convert per-symbol ml_scores into target weights.

        Normalizes signals to target weights respecting concentration limits.

        Args:
            symbols: Active symbols
            signals: symbol -> ml_score (signed, -1 to 1)

        Returns:
            symbol -> target_weight (signed, respecting max_concentration)
        """
        if not signals:
            return {s: 0.0 for s in symbols}

        # Raw weights proportional to signal strength
        raw = {s: signals.get(s, 0.0) for s in symbols}
        total_abs = sum(abs(v) for v in raw.values())

        if total_abs < 1e-8:
            return {s: 0.0 for s in symbols}

        # Normalize to max_gross_leverage
        max_gross = self.config.max_gross_leverage
        scale = min(max_gross / total_abs, 1.0)

        weights = {}
        for s in symbols:
            w = raw[s] * scale
            # Apply concentration limit
            max_c = self.config.max_concentration
            w = max(-max_c, min(max_c, w))
            weights[s] = w

        # Apply net leverage constraint
        net = sum(weights.values())
        max_net = self.config.max_net_leverage
        if abs(net) > max_net:
            # Scale all weights proportionally
            net_scale = max_net / abs(net)
            weights = {s: w * net_scale for s, w in weights.items()}

        self._target_weights = dict(weights)
        self._last_rebalance_ts = time.time()

        return weights

    def scale_order(
        self,
        symbol: str,
        qty: float,
        equity: float,
        price: float,
    ) -> float:
        """Scale an order quantity to respect portfolio constraints.

        If the resulting position would breach constraints, scales down.

        Args:
            symbol: Order symbol
            qty: Proposed order quantity (signed)
            equity: Current equity
            price: Current price

        Returns:
            Scaled quantity (may be same, reduced, or zero)
        """
        if equity <= 0 or price <= 0:
            return qty

        target_w = self._target_weights.get(symbol)
        if target_w is None:
            # No target computed yet, pass through
            return qty

        # Max allowed notional for this symbol
        max_notional = abs(target_w) * equity
        proposed_notional = abs(qty * price)

        if proposed_notional <= max_notional:
            return qty

        # Scale down to fit
        if proposed_notional > 0:
            scale = max_notional / proposed_notional
            return qty * scale

        return qty

    def get_status(self) -> Dict[str, Any]:
        """Return current allocation status for health endpoint."""
        return {
            "target_weights": dict(self._target_weights),
            "last_rebalance_ts": self._last_rebalance_ts,
            "config": {
                "max_gross_leverage": self.config.max_gross_leverage,
                "max_net_leverage": self.config.max_net_leverage,
                "max_concentration": self.config.max_concentration,
            },
        }
