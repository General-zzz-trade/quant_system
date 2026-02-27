# execution/algos/impact_model.py
"""Almgren-Chriss market impact model for optimal execution."""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from decimal import Decimal
from typing import List, Optional, Sequence

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ImpactEstimate:
    """Market impact estimation result."""
    permanent_impact_bps: float
    temporary_impact_bps: float
    total_impact_bps: float
    optimal_n_slices: int
    optimal_duration_sec: float
    expected_cost_bps: float


@dataclass(frozen=True, slots=True)
class AlmgrenChrissParams:
    """Parameters for the Almgren-Chriss model.

    sigma: Daily volatility of the asset
    eta: Temporary impact coefficient (price impact per unit traded per unit time)
    gamma: Permanent impact coefficient (permanent price shift per unit traded)
    daily_volume: Average daily trading volume
    risk_aversion: Lambda parameter — higher = more urgency, less market impact optimization
    """
    sigma: float = 0.02        # 2% daily vol
    eta: float = 2.5e-6        # temporary impact coefficient
    gamma: float = 2.5e-7      # permanent impact coefficient
    daily_volume: float = 1e8  # $100M daily volume
    risk_aversion: float = 1e-6


class AlmgrenChrissModel:
    """Almgren-Chriss optimal execution model.

    Computes the optimal trade schedule that minimizes the combination
    of market impact costs and timing risk.

    Reference: Almgren & Chriss (2000) "Optimal Execution of Portfolio Transactions"
    """

    def __init__(self, params: Optional[AlmgrenChrissParams] = None) -> None:
        self.params = params or AlmgrenChrissParams()

    def estimate_impact(
        self,
        total_qty: float,
        price: float,
        *,
        duration_sec: Optional[float] = None,
        n_slices: Optional[int] = None,
    ) -> ImpactEstimate:
        """Estimate market impact for a given order.

        Args:
            total_qty: Total quantity to execute
            price: Current market price
            duration_sec: Optional execution duration (auto-computed if None)
            n_slices: Optional number of slices (auto-computed if None)
        """
        p = self.params
        notional = total_qty * price

        # Participation rate
        participation = notional / max(p.daily_volume, 1.0)

        # Permanent impact: gamma * total_qty
        perm_impact = p.gamma * total_qty * price
        perm_bps = (perm_impact / notional) * 10000 if notional > 0 else 0.0

        # Optimal duration based on Almgren-Chriss
        if duration_sec is None:
            # T* proportional to sqrt(total_qty * risk_aversion / eta)
            kappa = math.sqrt(p.risk_aversion * p.sigma ** 2 / max(p.eta, 1e-12))
            optimal_t = max(1.0 / max(kappa, 1e-8), 60.0)
            # Scale by participation rate
            optimal_t = min(optimal_t * (1 + participation * 10), 7200.0)
            duration_sec = optimal_t
        else:
            duration_sec = max(duration_sec, 60.0)

        # Number of slices
        if n_slices is None:
            n_slices = max(int(duration_sec / 60.0), 3)
            n_slices = min(n_slices, 100)

        # Temporary impact per slice
        slice_qty = total_qty / max(n_slices, 1)
        slice_rate = slice_qty / max(duration_sec / n_slices, 1.0)
        temp_impact = p.eta * slice_rate * price
        temp_bps = (temp_impact / price) * 10000

        # Total expected cost
        total_bps = perm_bps + temp_bps
        expected_cost = perm_impact + temp_impact * n_slices

        return ImpactEstimate(
            permanent_impact_bps=round(perm_bps, 4),
            temporary_impact_bps=round(temp_bps, 4),
            total_impact_bps=round(total_bps, 4),
            optimal_n_slices=n_slices,
            optimal_duration_sec=round(duration_sec, 1),
            expected_cost_bps=round(
                (expected_cost / notional) * 10000 if notional > 0 else 0.0, 4,
            ),
        )

    def optimal_trajectory(
        self,
        total_qty: float,
        n_slices: int,
        duration_sec: float,
    ) -> List[float]:
        """Compute optimal execution trajectory (qty per slice).

        Returns a list of quantities to trade at each time step,
        following the Almgren-Chriss optimal execution schedule.
        """
        if n_slices <= 0:
            return []

        p = self.params
        tau = duration_sec / n_slices

        # kappa = sqrt(lambda * sigma^2 / eta)
        kappa = math.sqrt(
            p.risk_aversion * p.sigma ** 2 / max(p.eta, 1e-12),
        )

        # Optimal trajectory: x_j = X * sinh(kappa*(T-t_j)) / sinh(kappa*T)
        # where T = total duration, t_j = j * tau
        T = duration_sec
        kT = kappa * T

        if kT < 1e-6:
            # Linear trajectory for small kappa*T
            return [total_qty / n_slices] * n_slices

        trajectory = []
        remaining = total_qty
        for j in range(n_slices):
            t_j = j * tau
            t_next = (j + 1) * tau

            # Position at t_j and t_next
            pos_j = total_qty * math.sinh(kappa * (T - t_j)) / math.sinh(kT)
            pos_next = total_qty * math.sinh(kappa * (T - t_next)) / math.sinh(kT) if j < n_slices - 1 else 0.0

            qty = pos_j - pos_next
            trajectory.append(max(qty, 0.0))

        # Normalize to ensure total matches
        total_traj = sum(trajectory) or 1.0
        trajectory = [q * total_qty / total_traj for q in trajectory]

        return trajectory
