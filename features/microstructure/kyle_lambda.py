"""Kyle's Lambda — price impact per unit of signed order flow.

Estimates the permanent price impact coefficient via OLS regression
of price changes on signed trade volume.

Reference: Kyle (1985).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Optional, Sequence

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class KyleLambdaResult:
    """Result of Kyle's Lambda estimation."""

    kyle_lambda: float  # Price impact per unit volume
    r_squared: float  # Regression fit quality
    n_observations: int


class KyleLambdaEstimator:
    """Estimate Kyle's Lambda: price impact per unit of signed order flow.

    Model: delta_price = lambda * signed_volume + epsilon
    Lambda estimated via OLS regression.
    """

    def __init__(self, *, window: int = 100) -> None:
        self._window = window

    def estimate(self, ticks: Sequence[Any]) -> KyleLambdaResult:
        """Estimate Kyle's Lambda from tick data.

        Steps:
        1. Compute signed volume: +volume for buys, -volume for sells
        2. Compute price changes between consecutive ticks
        3. Regress price changes on signed volume
        4. Lambda = regression slope
        """
        if len(ticks) < 2:
            return KyleLambdaResult(
                kyle_lambda=0.0, r_squared=0.0, n_observations=0,
            )

        # Use last `window + 1` ticks to get `window` observations
        window_ticks = ticks[-(self._window + 1):]

        signed_volumes: list[float] = []
        price_changes: list[float] = []

        for i in range(1, len(window_ticks)):
            prev = window_ticks[i - 1]
            curr = window_ticks[i]

            p_prev = float(getattr(prev, "price", Decimal("0")))
            p_curr = float(getattr(curr, "price", Decimal("0")))
            qty = float(getattr(curr, "qty", Decimal("0")))

            side = getattr(curr, "side", "")
            if side == "sell":
                signed_vol = -qty
            else:
                signed_vol = qty

            price_changes.append(p_curr - p_prev)
            signed_volumes.append(signed_vol)

        n = len(signed_volumes)
        if n == 0:
            return KyleLambdaResult(
                kyle_lambda=0.0, r_squared=0.0, n_observations=0,
            )

        # Pure Python OLS: slope = cov(x,y) / var(x)
        slope, r_sq = _ols(signed_volumes, price_changes)

        return KyleLambdaResult(
            kyle_lambda=slope,
            r_squared=r_sq,
            n_observations=n,
        )


def _ols(x: list[float], y: list[float]) -> tuple[float, float]:
    """Simple OLS regression: y = slope * x + intercept.

    Returns (slope, r_squared).
    """
    n = len(x)
    if n == 0:
        return 0.0, 0.0

    mean_x = sum(x) / n
    mean_y = sum(y) / n

    cov_xy = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y)) / n
    var_x = sum((xi - mean_x) ** 2 for xi in x) / n
    var_y = sum((yi - mean_y) ** 2 for yi in y) / n

    if var_x < 1e-15:
        return 0.0, 0.0

    slope = cov_xy / var_x

    if var_y < 1e-15:
        r_squared = 1.0 if var_x < 1e-15 else 0.0
    else:
        r_squared = (cov_xy ** 2) / (var_x * var_y)

    return slope, r_squared
