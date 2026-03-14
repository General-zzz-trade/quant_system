# portfolio/risk_budget.py
"""Risk budget allocation and optimization objective."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Mapping

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RiskBudget:
    """Risk budget allocation across assets or strategies.

    Each budget entry maps an asset/strategy name to its target
    fraction of total portfolio risk. Budgets should sum to 1.0.
    """

    budgets: Mapping[str, float]  # symbol/strategy -> risk budget fraction

    def target_risk_contribution(self, asset: str) -> float:
        """Target risk contribution fraction for an asset."""
        return self.budgets.get(asset, 0.0)

    @property
    def total(self) -> float:
        """Sum of all budget fractions (should be ~1.0)."""
        return sum(self.budgets.values())

    def is_valid(self) -> bool:
        """Check that budgets are non-negative and sum to ~1.0."""
        if any(v < 0 for v in self.budgets.values()):
            return False
        return abs(self.total - 1.0) < 1e-6

    @staticmethod
    def equal(symbols: tuple[str, ...]) -> RiskBudget:
        """Create equal risk budget across all symbols."""
        n = len(symbols)
        if n == 0:
            return RiskBudget(budgets={})
        frac = 1.0 / n
        return RiskBudget(budgets={s: frac for s in symbols})


class RiskBudgetObjective:
    """Optimize weights to match target risk contributions.

    Minimizes: sum((RC_i / TotalRC - target_i)^2)

    Where RC_i = w_i * (Cov @ w)_i is the risk contribution of asset i.
    This generalizes RiskParity to non-equal budgets.
    """

    name: str = "risk_budget"

    def __init__(self, budget: RiskBudget) -> None:
        self._budget = budget

    @property
    def budget(self) -> RiskBudget:
        return self._budget

    def evaluate(self, weights: Mapping[str, float], input_data: object) -> float:
        """Evaluate the risk budget objective.

        Returns sum of squared deviations between actual and target
        risk contributions.
        """
        n = len(weights)
        if n == 0:
            return 0.0

        cov = getattr(input_data, "covariance", {})
        symbols = list(weights.keys())

        # Marginal risk contribution: (Cov @ w)_i
        marginal = []
        for s1 in symbols:
            row = cov.get(s1, {})
            mc = sum(weights.get(s2, 0.0) * row.get(s2, 0.0) for s2 in symbols)
            marginal.append(mc)

        # Risk contribution: RC_i = w_i * marginal_i
        risk_contrib = [weights[s] * m for s, m in zip(symbols, marginal)]
        total_risk = sum(risk_contrib)

        if abs(total_risk) < 1e-20:
            return 0.0

        # Deviation from target budget
        obj = 0.0
        for i, s in enumerate(symbols):
            actual_frac = risk_contrib[i] / total_risk
            target_frac = self._budget.target_risk_contribution(s)
            obj += (actual_frac - target_frac) ** 2

        return obj
