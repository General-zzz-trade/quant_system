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
# Design: 10x total leverage at medium tier (BTC 5x + ETH 5x).
# Within each symbol: 4h gets 60% (higher conviction), 1h gets 40%.
# z_scale/IC/regime further adjust at runtime.
_TIER_WEIGHTS: dict[str, dict[str, float]] = {
    "micro": {  # equity < 500 — ETH 6.5x + BTC 2x (2026-04-13 portfolio backtest):
                # 3m: combo +98.1% vs pure-ETH +54.0%, all windows improved.
                # BTC active=23% + ETH active=6% = complementary signals.
        "BTCUSDT": 0.20,   # 2x effective: 0.20 × 10 = 2.0x
        "ETHUSDT": 0.65,   # 6.5x effective: 0.65 × 10 = 6.5x
        "SOLUSDT": 0.40,   # unchanged (dropped from active roster)
        "BTCUSDT_4h": 0.0,
        "ETHUSDT_4h": 0.0,
    },
    "medium": {  # 500 <= equity < 10_000
        # 14-day data: BTC $228/15=$15/trade, ETH $1749/8=$219/trade
        # BTC was under-sized; raise to match ETH exposure
        "BTCUSDT": 0.45,  # was 0.35 (1h only trades now that 4h is signal-only)
        "ETHUSDT": 0.45,  # was 0.40
        "SOLUSDT": 0.30,
        "BTCUSDT_4h": 0.0,  # signal_only — no orders
        "ETHUSDT_4h": 0.0,
    },
    "large": {  # equity >= 10_000 — 10x leverage safe
        "BTCUSDT": 0.10,  # was 0.08
        "ETHUSDT": 0.10,  # was 0.08
        "SOLUSDT": 0.08,
        "BTCUSDT_4h": 0.0,
        "ETHUSDT_4h": 0.0,
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
            return "micro"
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
        # Prefer _f (float) accessors over raw Fd8 i64
        acct = snapshot.account
        _bf = getattr(acct, "balance_f", None)
        if isinstance(_bf, (int, float)) and _bf > 0:
            equity = float(_bf)
        else:
            raw_b = float(acct.balance)
            equity = raw_b / 100_000_000 if raw_b > 1_000_000 else raw_b

        market = snapshot.markets.get(symbol)
        if market is not None:
            _cf = getattr(market, "close_f", None)
            if isinstance(_cf, (int, float)) and _cf > 0:
                price = float(_cf)
            else:
                raw_c = float(market.close)
                price = raw_c / 100_000_000 if raw_c > 1_000_000 else raw_c
        else:
            price = 0.0

        if _RUST_SIZER:
            try:
                result = rust_adaptive_target_qty(
                    self.runner_key, equity, price,
                    self.step_size, self.min_size, self.max_qty,
                    float(weight), leverage, ic_scale,
                    regime_active, z_scale,
                )
                logger.debug("sizer_path=rust runner=%s qty=%s", self.runner_key, result)
                return Decimal(str(result))
            except Exception:
                logger.warning(
                    "Rust sizer failed for %s, falling back to Python",
                    self.runner_key,
                    exc_info=True,
                )

        else:
            logger.debug("sizer_path=python runner=%s (Rust not available)", self.runner_key)

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

        # 5. Clamp — but respect cap=0.0 (disabled symbol in this tier)
        if base_cap > 0:
            size = max(size, self.min_size)
        else:
            size = 0.0
        if self.max_qty > 0:
            size = min(size, self.max_qty)

        return self._round_to_step(size)
