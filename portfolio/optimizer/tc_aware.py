# portfolio/optimizer/tc_aware.py
"""Transaction-cost-aware optimization objective."""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Mapping

from portfolio.optimizer.objectives import Objective

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class TransactionCostConfig:
    """Configuration for transaction cost modeling."""
    fee_bps: float = 4.0           # Trading fee in basis points
    impact_model: str = "sqrt"     # "linear" | "sqrt" | "almgren_chriss"
    impact_coefficient: float = 0.1  # Market impact scaling factor
    turnover_penalty: float = 0.01   # Lambda for TC penalty in objective


class TransactionCostAwareObjective:
    """Wraps another objective and penalizes transaction costs.

    obj_value = inner_objective(w) + lambda * TC(w, w_current)
    TC = sum of (fees + market_impact) for each asset's weight change.
    """

    name: str = "tc_aware"

    def __init__(
        self,
        inner: Objective,
        config: TransactionCostConfig | None = None,
    ) -> None:
        self._inner = inner
        self._config = config or TransactionCostConfig()

    @property
    def inner(self) -> Objective:
        return self._inner

    @property
    def config(self) -> TransactionCostConfig:
        return self._config

    def evaluate(self, weights: Mapping[str, float], input_data: object) -> float:
        """Evaluate the TC-penalized objective.

        Returns inner objective value plus turnover_penalty * transaction_cost.
        """
        base = self._inner.evaluate(weights, input_data)
        current = getattr(input_data, "current_weights", {})
        if not current:
            return base
        tc = self._estimate_transaction_cost(weights, current, input_data)
        return base + self._config.turnover_penalty * tc

    def _estimate_transaction_cost(
        self,
        new_w: Mapping[str, float],
        old_w: Mapping[str, float],
        input_data: object,
    ) -> float:
        """Estimate total transaction cost from weight changes.

        TC_i = |delta_w_i| * (fee_bps / 10_000 + impact_i)
        """
        all_symbols = set(new_w) | set(old_w)
        fee_frac = self._config.fee_bps / 10_000.0
        total_equity = getattr(input_data, "total_equity", 0.0)

        total_cost = 0.0
        for sym in all_symbols:
            delta = abs(new_w.get(sym, 0.0) - old_w.get(sym, 0.0))
            if delta < 1e-12:
                continue

            # Fee component
            delta * fee_frac

            # Market impact component
            impact = self._compute_impact(delta, total_equity)

            total_cost += delta * (fee_frac + impact)

        return total_cost

    def _compute_impact(self, delta_weight: float, total_equity: float) -> float:
        """Compute market impact for a single asset's weight change."""
        coeff = self._config.impact_coefficient
        model = self._config.impact_model

        if model == "linear":
            return coeff * delta_weight

        if model == "sqrt":
            return coeff * math.sqrt(delta_weight)

        if model == "almgren_chriss":
            # Simplified Almgren-Chriss: impact ~ coeff * delta^(3/5)
            return coeff * (delta_weight ** 0.6)

        # Default to linear
        return coeff * delta_weight

    def estimate_turnover(
        self,
        new_w: Mapping[str, float],
        old_w: Mapping[str, float],
    ) -> float:
        """Sum of absolute weight changes (one-way turnover)."""
        all_symbols = set(new_w) | set(old_w)
        return sum(
            abs(new_w.get(s, 0.0) - old_w.get(s, 0.0))
            for s in all_symbols
        )

    def estimate_cost_breakdown(
        self,
        new_w: Mapping[str, float],
        old_w: Mapping[str, float],
        input_data: object,
    ) -> dict[str, float]:
        """Breakdown of transaction costs by component."""
        all_symbols = set(new_w) | set(old_w)
        fee_frac = self._config.fee_bps / 10_000.0
        total_equity = getattr(input_data, "total_equity", 0.0)

        total_fees = 0.0
        total_impact = 0.0
        total_turnover = 0.0

        for sym in all_symbols:
            delta = abs(new_w.get(sym, 0.0) - old_w.get(sym, 0.0))
            if delta < 1e-12:
                continue
            total_turnover += delta
            total_fees += delta * fee_frac
            total_impact += delta * self._compute_impact(delta, total_equity)

        return {
            "turnover": total_turnover,
            "fees": total_fees,
            "impact": total_impact,
            "total": total_fees + total_impact,
        }
