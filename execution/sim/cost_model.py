"""Realistic vectorized cost model for backtest PnL simulation.

Replaces the fixed 6bps cost assumption with a multi-component model:
  1. Trading fees (maker/taker weighted)
  2. Market impact (Almgren-Chriss sqrt model)
  3. Bid-ask spread (volatility-proportional)
  4. Volume participation constraint
  5. Funding costs (unchanged from existing pipeline)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CostBreakdown:
    """Per-bar cost decomposition."""
    fee_cost: np.ndarray        # Trading fees
    impact_cost: np.ndarray     # Market impact
    spread_cost: np.ndarray     # Bid-ask spread crossing
    total_cost: np.ndarray      # Sum of all components
    clipped_signal: np.ndarray  # Signal after volume participation clipping


@dataclass
class RealisticCostModel:
    """Multi-component cost model for realistic backtest PnL.

    All costs are computed as fractions of notional (not bps).
    """
    # Fee rates (in bps, converted internally)
    maker_fee_bps: float = 2.0     # Binance VIP0 maker
    taker_fee_bps: float = 4.0     # Binance VIP0 taker
    taker_ratio: float = 0.7       # Fraction of trades that cross the spread

    # Market impact (Almgren-Chriss)
    eta: float = 0.5               # Impact coefficient

    # Bid-ask spread
    spread_multiplier: float = 0.05  # spread ≈ multiplier × per-bar volatility (~1-2 bps for BTC)

    # Volume participation
    max_participation: float = 0.10  # Max fraction of bar volume we consume

    def compute_costs(
        self,
        signal: np.ndarray,
        closes: np.ndarray,
        volumes: np.ndarray,
        volatility: np.ndarray,
        capital: float = 10000.0,
        funding_rates: np.ndarray | None = None,
    ) -> CostBreakdown:
        """Compute per-bar cost breakdown.

        Parameters
        ----------
        signal : (N,) position signal in [-1, 1]
        closes : (N,) close prices
        volumes : (N,) bar volumes (base asset units)
        volatility : (N,) per-bar realized volatility (e.g., 20-bar rolling std of returns)
        capital : notional capital
        funding_rates : (N,) funding rates per settlement (optional)

        Returns
        -------
        CostBreakdown with per-bar costs as fractions of capital.
        """
        n = len(signal)
        signal = signal.copy()

        # --- Step 1: Volume participation clipping ---
        # Clip position changes to max_participation of bar volume
        turnover_raw = np.abs(np.diff(signal, prepend=0.0))
        # Position change in base units
        notional = capital  # Assume constant capital for simplicity
        max_position_change = np.where(
            closes > 0,
            (volumes * self.max_participation * closes) / notional,
            np.inf,
        )
        # Clip turnover
        excess = turnover_raw > max_position_change
        if excess.any():
            # Rebuild signal respecting participation limits
            clipped_signal = np.empty_like(signal)
            clipped_signal[0] = np.clip(signal[0], -max_position_change[0], max_position_change[0])
            for i in range(1, n):
                delta = signal[i] - clipped_signal[i - 1]
                max_delta = max_position_change[i]
                delta_clipped = np.clip(delta, -max_delta, max_delta)
                clipped_signal[i] = clipped_signal[i - 1] + delta_clipped
        else:
            clipped_signal = signal

        turnover = np.abs(np.diff(clipped_signal, prepend=0.0))

        # --- Step 2: Trading fees ---
        blended_fee = (
            self.taker_ratio * self.taker_fee_bps +
            (1.0 - self.taker_ratio) * self.maker_fee_bps
        ) / 10000.0  # bps to fraction
        fee_cost = turnover * blended_fee

        # --- Step 3: Market impact (Almgren-Chriss sqrt model) ---
        # impact = eta * sigma_daily * sqrt(qty / ADV)
        # qty/ADV ≈ turnover * capital / (volume * close)
        safe_vol_notional = np.maximum(volumes * closes, 1.0)
        participation = (turnover * notional) / safe_vol_notional
        # Use per-bar volatility as proxy for daily sigma
        # (hourly vol → daily: multiply by sqrt(24))
        sigma_daily = np.where(
            np.isnan(volatility) | (volatility <= 0),
            0.0,
            volatility * np.sqrt(24.0),
        )
        impact_cost = self.eta * sigma_daily * np.sqrt(np.maximum(participation, 0.0))

        # --- Step 4: Bid-ask spread ---
        # Half-spread cost on each side of the trade
        spread_bps = self.spread_multiplier * np.where(
            np.isnan(volatility), 0.0, volatility
        )
        spread_cost = turnover * spread_bps / 2.0

        # --- Total ---
        total_cost = fee_cost + impact_cost + spread_cost

        return CostBreakdown(
            fee_cost=fee_cost,
            impact_cost=impact_cost,
            spread_cost=spread_cost,
            total_cost=total_cost,
            clipped_signal=clipped_signal,
        )

    @staticmethod
    def flat_cost(signal: np.ndarray, cost_per_trade: float = 0.0006) -> np.ndarray:
        """Legacy flat cost model: turnover × fixed bps."""
        turnover = np.abs(np.diff(signal, prepend=0.0))
        return turnover * cost_per_trade
