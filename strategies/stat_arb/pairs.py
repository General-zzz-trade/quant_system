"""Pairs trading — cointegration-based statistical arbitrage.

Identifies cointegrated pairs and generates mean-reversion signals
when the spread deviates from its historical mean.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import List, Mapping, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PairResult:
    """Cointegration test result for a pair."""
    symbol_a: str
    symbol_b: str
    correlation: float
    hedge_ratio: float
    spread_mean: float
    spread_std: float
    half_life: float
    is_cointegrated: bool


@dataclass(frozen=True, slots=True)
class PairSignal:
    """Trading signal from pairs strategy."""
    symbol_a: str
    symbol_b: str
    side_a: str   # "buy" or "sell"
    side_b: str
    zscore: float
    strength: float


def _ols_simple(x: Sequence[float], y: Sequence[float]) -> Tuple[float, float]:
    """Simple OLS: y = alpha + beta * x. Returns (alpha, beta)."""
    n = len(x)
    sx = sum(x)
    sy = sum(y)
    sxx = sum(xi * xi for xi in x)
    sxy = sum(xi * yi for xi, yi in zip(x, y))

    denom = n * sxx - sx * sx
    if abs(denom) < 1e-12:
        return 0.0, 0.0

    beta = (n * sxy - sx * sy) / denom
    alpha = (sy - beta * sx) / n
    return alpha, beta


def _correlation(x: Sequence[float], y: Sequence[float]) -> float:
    n = len(x)
    if n < 2:
        return 0.0
    mx = sum(x) / n
    my = sum(y) / n
    cov = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y)) / n
    sx = math.sqrt(sum((xi - mx) ** 2 for xi in x) / n)
    sy = math.sqrt(sum((yi - my) ** 2 for yi in y) / n)
    if sx * sy == 0:
        return 0.0
    return cov / (sx * sy)


def _half_life(spread: Sequence[float]) -> float:
    """Estimate mean-reversion half-life via AR(1)."""
    if len(spread) < 3:
        return float("inf")
    y = [spread[i] - spread[i - 1] for i in range(1, len(spread))]
    x = list(spread[:-1])
    _, beta = _ols_simple(x, y)
    if beta >= 0:
        return float("inf")
    return -math.log(2) / beta


def find_pairs(
    prices: Mapping[str, Sequence[float]],
    *,
    min_correlation: float = 0.7,
    min_observations: int = 60,
) -> List[PairResult]:
    """Screen for cointegrated pairs.

    Args:
        prices: Symbol → price series mapping.
        min_correlation: Minimum correlation threshold.
        min_observations: Minimum number of price observations.
    """
    symbols = [s for s, p in prices.items() if len(p) >= min_observations]
    results: List[PairResult] = []

    for i in range(len(symbols)):
        for j in range(i + 1, len(symbols)):
            sa, sb = symbols[i], symbols[j]
            pa = list(prices[sa])
            pb = list(prices[sb])
            n = min(len(pa), len(pb))
            pa, pb = pa[:n], pb[:n]

            corr = _correlation(pa, pb)
            if abs(corr) < min_correlation:
                continue

            alpha, beta = _ols_simple(pa, pb)
            spread = [pb[k] - beta * pa[k] - alpha for k in range(n)]

            spread_mean = sum(spread) / len(spread)
            spread_std = math.sqrt(sum((s - spread_mean) ** 2 for s in spread) / len(spread))
            hl = _half_life(spread)

            is_coint = hl < len(spread) * 0.5 and spread_std > 0

            results.append(PairResult(
                symbol_a=sa,
                symbol_b=sb,
                correlation=corr,
                hedge_ratio=beta,
                spread_mean=spread_mean,
                spread_std=spread_std,
                half_life=hl,
                is_cointegrated=is_coint,
            ))

    results.sort(key=lambda r: r.half_life)
    return results


class PairsStrategy:
    """Live pairs trading strategy.

    Generates signals based on z-score of the pair spread.
    """

    def __init__(
        self,
        pair: PairResult,
        *,
        entry_zscore: float = 2.0,
        exit_zscore: float = 0.5,
        window: int = 60,
    ) -> None:
        self._pair = pair
        self._entry_z = entry_zscore
        self._exit_z = exit_zscore
        self._window = window
        self._spreads: List[float] = []
        self._position: str = "flat"  # "long_spread", "short_spread", "flat"

    def on_prices(self, price_a: float, price_b: float) -> Optional[PairSignal]:
        """Feed new prices, get signal."""
        spread = price_b - self._pair.hedge_ratio * price_a

        self._spreads.append(spread)
        if len(self._spreads) > self._window:
            self._spreads.pop(0)

        if len(self._spreads) < self._window:
            return None

        mean = sum(self._spreads) / len(self._spreads)
        std = math.sqrt(sum((s - mean) ** 2 for s in self._spreads) / len(self._spreads))
        if std == 0:
            return None

        z = (spread - mean) / std

        if self._position == "flat":
            if z > self._entry_z:
                self._position = "short_spread"
                return PairSignal(
                    symbol_a=self._pair.symbol_a,
                    symbol_b=self._pair.symbol_b,
                    side_a="buy", side_b="sell",
                    zscore=z, strength=min(abs(z) / self._entry_z, 1.0),
                )
            elif z < -self._entry_z:
                self._position = "long_spread"
                return PairSignal(
                    symbol_a=self._pair.symbol_a,
                    symbol_b=self._pair.symbol_b,
                    side_a="sell", side_b="buy",
                    zscore=z, strength=min(abs(z) / self._entry_z, 1.0),
                )

        elif self._position == "long_spread" and z > -self._exit_z:
            self._position = "flat"
            return PairSignal(
                symbol_a=self._pair.symbol_a,
                symbol_b=self._pair.symbol_b,
                side_a="buy", side_b="sell",
                zscore=z, strength=0.0,
            )

        elif self._position == "short_spread" and z < self._exit_z:
            self._position = "flat"
            return PairSignal(
                symbol_a=self._pair.symbol_a,
                symbol_b=self._pair.symbol_b,
                side_a="sell", side_b="buy",
                zscore=z, strength=0.0,
            )

        return None
