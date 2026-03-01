# research/monte_carlo.py
"""Monte Carlo path simulation for return sequences."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Tuple

try:
    from features._quant_rolling import cpp_simulate_paths as _cpp_simulate
    _MC_CPP = True
except ImportError:
    _MC_CPP = False


@dataclass(frozen=True, slots=True)
class MonteCarloResult:
    """Aggregate statistics from Monte Carlo simulation."""
    paths: int
    mean_final: float
    median_final: float
    percentile_5: float
    percentile_95: float
    prob_loss: float        # P(final < initial)
    prob_target: float      # P(final >= target)
    max_drawdown_mean: float
    max_drawdown_95: float


def _max_drawdown(equity_curve: List[float]) -> float:
    """Compute maximum drawdown from an equity curve."""
    peak = equity_curve[0]
    max_dd = 0.0
    for v in equity_curve:
        if v > peak:
            peak = v
        dd = (peak - v) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
    return max_dd


def _percentile(sorted_vals: List[float], p: float) -> float:
    """Linear-interpolation percentile on a pre-sorted list."""
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_vals[0]
    k = (n - 1) * p / 100.0
    lo = int(math.floor(k))
    hi = min(lo + 1, n - 1)
    frac = k - lo
    return sorted_vals[lo] + frac * (sorted_vals[hi] - sorted_vals[lo])


class MonteCarloSimulator:
    """Monte Carlo path simulation for return sequences."""

    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)

    def simulate_single_path(
        self,
        returns: List[float],
        horizon: int,
        method: str,
    ) -> List[float]:
        """Generate a single equity path (cumulative wealth, starting at 1.0).

        Args:
            returns: Historical return series to resample from.
            horizon: Number of steps to simulate.
            method: "bootstrap" (block bootstrap, block_size=5) or "parametric".

        Returns:
            Equity curve of length horizon+1 (including initial 1.0).
        """
        if not returns:
            return [1.0] * (horizon + 1)

        sampled: List[float]
        if method == "parametric":
            mu = sum(returns) / len(returns)
            var = sum((r - mu) ** 2 for r in returns) / max(len(returns) - 1, 1)
            sigma = math.sqrt(var)
            sampled = [self._rng.gauss(mu, sigma) for _ in range(horizon)]
        else:
            # Block bootstrap with block_size=5
            block_size = 5
            sampled = []
            n = len(returns)
            while len(sampled) < horizon:
                start = self._rng.randint(0, n - 1)
                for j in range(block_size):
                    if len(sampled) >= horizon:
                        break
                    sampled.append(returns[(start + j) % n])

        # Build equity curve
        equity = [1.0]
        for r in sampled:
            equity.append(equity[-1] * (1.0 + r))
        return equity

    def simulate_paths(
        self,
        returns: List[float],
        n_paths: int = 1000,
        horizon: int = 252,
        method: str = "bootstrap",
        target_return: float = 0.0,
    ) -> MonteCarloResult:
        """Simulate equity paths and compute aggregate statistics.

        Args:
            returns: Historical return series.
            n_paths: Number of paths to simulate.
            horizon: Number of steps per path.
            method: "bootstrap" or "parametric".
            target_return: Target cumulative return for prob_target.

        Returns:
            MonteCarloResult with path statistics.
        """
        if not returns or n_paths < 1:
            return MonteCarloResult(
                paths=0, mean_final=1.0, median_final=1.0,
                percentile_5=1.0, percentile_95=1.0,
                prob_loss=0.0, prob_target=0.0,
                max_drawdown_mean=0.0, max_drawdown_95=0.0,
            )

        if _MC_CPP:
            parametric = (method == "parametric")
            r = _cpp_simulate(
                returns, n_paths, horizon, parametric,
                target_return, 5, self._rng.randint(0, 2**63),
            )
            return MonteCarloResult(
                paths=r.paths, mean_final=r.mean_final,
                median_final=r.median_final,
                percentile_5=r.percentile_5,
                percentile_95=r.percentile_95,
                prob_loss=r.prob_loss, prob_target=r.prob_target,
                max_drawdown_mean=r.max_drawdown_mean,
                max_drawdown_95=r.max_drawdown_95,
            )

        finals: List[float] = []
        drawdowns: List[float] = []
        target_wealth = 1.0 + target_return

        for _ in range(n_paths):
            equity = self.simulate_single_path(returns, horizon, method)
            finals.append(equity[-1])
            drawdowns.append(_max_drawdown(equity))

        finals_sorted = sorted(finals)
        dd_sorted = sorted(drawdowns)

        mean_final = sum(finals) / n_paths
        median_final = _percentile(finals_sorted, 50)
        p5 = _percentile(finals_sorted, 5)
        p95 = _percentile(finals_sorted, 95)
        prob_loss = sum(1 for f in finals if f < 1.0) / n_paths
        prob_target = sum(1 for f in finals if f >= target_wealth) / n_paths
        dd_mean = sum(drawdowns) / n_paths
        dd_95 = _percentile(dd_sorted, 95)

        return MonteCarloResult(
            paths=n_paths,
            mean_final=mean_final,
            median_final=median_final,
            percentile_5=p5,
            percentile_95=p95,
            prob_loss=prob_loss,
            prob_target=prob_target,
            max_drawdown_mean=dd_mean,
            max_drawdown_95=dd_95,
        )
