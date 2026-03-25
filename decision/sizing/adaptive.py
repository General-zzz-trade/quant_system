"""Adaptive position sizer with equity-tier weights, IC health, and regime awareness.

Replaces the monolithic sizing logic from AlphaRunner with a composable,
testable sizer that plugs into the framework's PositionSizer protocol.
"""
from __future__ import annotations

import logging
import math
from decimal import Decimal, ROUND_DOWN

from state.snapshot import StateSnapshot

logger = logging.getLogger(__name__)

try:
    from _quant_hotpath import rust_adaptive_target_qty
    _RUST_SIZER = True
except ImportError:
    _RUST_SIZER = False

# ── Equity-tier base weights per runner key ────────────────────────
# Keys match SYMBOL_CONFIG runner_key values.
_TIER_WEIGHTS: dict[str, dict[str, float]] = {
    "small": {  # equity < 500
        "BTCUSDT": 0.25,
        "ETHUSDT": 0.25,
        "BTCUSDT_4h": 0.35,
        "ETHUSDT_4h": 0.30,
    },
    "medium": {  # 500 <= equity < 10_000
        "BTCUSDT": 0.18,
        "ETHUSDT": 0.18,
        "BTCUSDT_4h": 0.25,
        "ETHUSDT_4h": 0.20,
    },
    "large": {  # equity >= 10_000
        "BTCUSDT": 0.12,
        "ETHUSDT": 0.12,
        "BTCUSDT_4h": 0.18,
        "ETHUSDT_4h": 0.15,
    },
}

# Fallback cap when runner_key is not in the tier table.
_DEFAULT_CAP = 0.15


class AdaptivePositionSizer:
    """Equity-tier + IC-health + regime-aware position sizer.

    Parameters
    ----------
    runner_key : str
        Runner identifier (e.g. ``"BTCUSDT_4h"``).
    step_size : float
        Minimum lot increment for rounding.
    min_size : float
        Minimum quantity returned (absolute floor).
    max_qty : float
        Hard upper clamp; 0 means unlimited.
    """

    def __init__(
        self,
        runner_key: str,
        step_size: float = 0.001,
        min_size: float = 0.001,
        max_qty: float = 0,
    ) -> None:
        self.runner_key = runner_key
        self.step_size = step_size
        self.min_size = min_size
        self.max_qty = max_qty

    # ── helpers ────────────────────────────────────────────────

    def _round_to_step(self, size: float) -> Decimal:
        """Floor *size* to the nearest step_size increment."""
        if self.step_size <= 0:
            return Decimal(str(size))
        # Number of decimal places implied by step_size
        decimals = max(0, -math.floor(math.log10(self.step_size)))
        quant = Decimal(10) ** -decimals
        return Decimal(str(size)).quantize(quant, rounding=ROUND_DOWN)

    @staticmethod
    def _equity_tier(equity: float) -> str:
        if equity < 500:
            return "small"
        if equity < 10_000:
            return "medium"
        return "large"

    # ── main entry point ──────────────────────────────────────

    def target_qty(
        self,
        snapshot: StateSnapshot,
        symbol: str,
        weight: Decimal = Decimal("1"),
        leverage: float = 10.0,
        ic_scale: float = 1.0,
        regime_active: bool = True,
        z_scale: float = 1.0,
    ) -> Decimal:
        """Compute target position quantity.

        Parameters
        ----------
        snapshot : StateSnapshot
            Current state (account balance + market prices).
        symbol : str
            Trading symbol.
        weight : Decimal
            External allocation weight (default 1).
        leverage : float
            Account leverage multiplier.
        ic_scale : float
            IC-health multiplier (GREEN=1.2, YELLOW=0.8, RED=0.4).
        regime_active : bool
            Whether the regime filter is active; inactive reduces cap by 40%.
        z_scale : float
            Z-score confidence scaler.
        """
        equity = float(snapshot.account.balance)
        market = snapshot.markets.get(symbol)
        price = float(market.close) if market is not None else 0.0

        if _RUST_SIZER:
            result = rust_adaptive_target_qty(
                self.runner_key, equity, price,
                self.step_size, self.min_size, self.max_qty,
                float(weight), leverage, ic_scale,
                regime_active, z_scale,
            )
            return Decimal(str(result))

        if equity <= 0 or price <= 0:
            return self._round_to_step(self.min_size)

        # 1. Tier-based cap
        tier = self._equity_tier(equity)
        base_cap = _TIER_WEIGHTS[tier].get(self.runner_key, _DEFAULT_CAP)

        # 2. Regime discount
        if not regime_active:
            base_cap *= 0.6

        # 3. IC health scaling
        per_sym_cap = base_cap * ic_scale

        # 4. Notional → quantity
        notional = equity * per_sym_cap * leverage * float(weight)
        size = notional / price * z_scale

        # 5. Clamp
        size = max(size, self.min_size)
        if self.max_qty > 0:
            size = min(size, self.max_qty)

        return self._round_to_step(size)
