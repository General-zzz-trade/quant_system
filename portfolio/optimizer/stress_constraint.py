# portfolio/optimizer/stress_constraint.py
"""Stress loss constraint for portfolio optimization."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Mapping, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class StressConstraintConfig:
    """Configuration for stress loss constraint."""
    max_stress_loss_pct: float = 20.0  # Maximum loss under stress scenario (percentage)
    scenarios: tuple = ()               # Stress scenarios (from risk/stress.py)


class StressLossConstraint:
    """Constraint: portfolio stress loss must be within threshold.

    Uses a callable stress_fn(weights) -> float that returns the
    maximum stress loss percentage for the given weights.
    If stress_fn is None, the constraint is always satisfied.
    """

    name: str = "stress_loss"

    def __init__(
        self,
        config: StressConstraintConfig | None = None,
        stress_fn: Optional[Callable[[Mapping[str, float]], float]] = None,
    ) -> None:
        self._config = config or StressConstraintConfig()
        self._stress_fn = stress_fn

    @property
    def config(self) -> StressConstraintConfig:
        return self._config

    def stress_loss(self, weights: Mapping[str, float]) -> float:
        """Compute stress loss for the given weights. Returns 0 if no stress_fn."""
        if self._stress_fn is None:
            return 0.0
        return self._stress_fn(weights)

    def is_feasible(self, weights: Mapping[str, float]) -> bool:
        """Check if stress loss is within the threshold."""
        if self._stress_fn is None:
            return True
        loss = self._stress_fn(weights)
        return loss <= self._config.max_stress_loss_pct

    def project(self, weights: dict[str, float]) -> dict[str, float]:
        """Project weights to satisfy the stress loss constraint.

        If stress loss exceeds the threshold, scale down all positions
        proportionally using binary search for the right scaling factor.
        """
        if self.is_feasible(weights):
            return weights

        if self._stress_fn is None:
            return weights

        # Binary search for the largest scaling factor in [0, 1] such that
        # the scaled weights satisfy the stress constraint.
        lo = 0.0
        hi = 1.0
        best_scale = 0.0

        for _ in range(50):  # Max 50 iterations for convergence
            mid = (lo + hi) / 2.0
            scaled = {s: w * mid for s, w in weights.items()}
            loss = self._stress_fn(scaled)

            if loss <= self._config.max_stress_loss_pct:
                best_scale = mid
                lo = mid
            else:
                hi = mid

            if hi - lo < 1e-8:
                break

        return {s: w * best_scale for s, w in weights.items()}
