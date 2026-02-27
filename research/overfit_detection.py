# research/overfit_detection.py
"""Overfit detection — Deflated Sharpe Ratio, parameter stability, probability of backtest overfitting."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from math import erf, exp, log, pi, sqrt
from typing import Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


def _norm_cdf(x: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    """Standard normal PDF."""
    return exp(-0.5 * x * x) / sqrt(2.0 * pi)


# ── Deflated Sharpe Ratio ────────────────────────────────────

@dataclass(frozen=True, slots=True)
class DeflatedSharpeResult:
    """Result of Deflated Sharpe Ratio test."""
    observed_sharpe: float
    deflated_sharpe: float
    p_value: float
    is_significant: bool
    n_trials: int
    expected_max_sharpe: float


def deflated_sharpe_ratio(
    observed_sharpe: float,
    n_trials: int,
    n_observations: int,
    variance_of_sharpes: float = 1.0,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    significance: float = 0.05,
) -> DeflatedSharpeResult:
    """Compute the Deflated Sharpe Ratio (Bailey & López de Prado).

    Adjusts for multiple testing: the more strategies tested, the higher
    the expected maximum Sharpe just from random chance.

    Args:
        observed_sharpe: The Sharpe ratio of the selected strategy.
        n_trials: Number of strategy configurations tested.
        n_observations: Number of return observations.
        variance_of_sharpes: Variance across all tested Sharpe ratios.
        skewness: Skewness of returns.
        kurtosis: Kurtosis of returns (3 = normal).
        significance: Significance level for the test.

    Returns:
        DeflatedSharpeResult with p-value and significance flag.
    """
    if n_trials < 1 or n_observations < 2:
        return DeflatedSharpeResult(
            observed_sharpe=observed_sharpe,
            deflated_sharpe=0.0,
            p_value=1.0,
            is_significant=False,
            n_trials=n_trials,
            expected_max_sharpe=0.0,
        )

    # Expected maximum Sharpe under null (Euler-Mascheroni approximation)
    euler_mascheroni = 0.5772156649
    if n_trials > 1:
        e_max = sqrt(variance_of_sharpes) * (
            (1 - euler_mascheroni) * _approx_inverse_norm(1 - 1.0 / n_trials)
            + euler_mascheroni * _approx_inverse_norm(1 - 1.0 / (n_trials * exp(1)))
        )
    else:
        e_max = 0.0

    # Standard error of Sharpe estimate
    sr_std = sqrt(
        (1 - skewness * observed_sharpe + (kurtosis - 1) / 4.0 * observed_sharpe ** 2)
        / max(n_observations - 1, 1)
    )

    if sr_std < 1e-12:
        return DeflatedSharpeResult(
            observed_sharpe=observed_sharpe,
            deflated_sharpe=0.0,
            p_value=1.0,
            is_significant=False,
            n_trials=n_trials,
            expected_max_sharpe=e_max,
        )

    # Deflated Sharpe
    dsr = (observed_sharpe - e_max) / sr_std
    p_value = 1.0 - _norm_cdf(dsr)

    return DeflatedSharpeResult(
        observed_sharpe=observed_sharpe,
        deflated_sharpe=dsr,
        p_value=p_value,
        is_significant=p_value < significance,
        n_trials=n_trials,
        expected_max_sharpe=e_max,
    )


def _approx_inverse_norm(p: float) -> float:
    """Approximate inverse normal CDF (Beasley-Springer-Moro)."""
    if p <= 0:
        return -6.0
    if p >= 1:
        return 6.0
    if p == 0.5:
        return 0.0

    # Rational approximation
    t = sqrt(-2.0 * log(min(p, 1.0 - p)))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    result = t - (c0 + c1 * t + c2 * t * t) / (1 + d1 * t + d2 * t * t + d3 * t ** 3)
    return result if p > 0.5 else -result


# ── Parameter Stability ──────────────────────────────────────

@dataclass(frozen=True, slots=True)
class ParameterStabilityResult:
    """Result of parameter stability analysis."""
    param_name: str
    values_tested: int
    best_value: float
    neighbor_metrics: List[float]
    stability_score: float  # 0-1, higher = more stable
    is_stable: bool


def parameter_stability(
    param_values: Sequence[float],
    metrics: Sequence[float],
    stability_threshold: float = 0.5,
) -> ParameterStabilityResult:
    """Analyze parameter stability — are nearby parameter values similarly good?

    A stable parameter has similar performance for nearby values.
    An unstable parameter shows sharp performance cliffs.

    Args:
        param_values: Parameter values tested (sorted).
        metrics: Corresponding performance metrics.
        stability_threshold: Minimum stability score to consider stable.

    Returns:
        ParameterStabilityResult with stability analysis.
    """
    if len(param_values) < 3 or len(param_values) != len(metrics):
        return ParameterStabilityResult(
            param_name="", values_tested=len(param_values),
            best_value=0.0, neighbor_metrics=[],
            stability_score=0.0, is_stable=False,
        )

    best_idx = max(range(len(metrics)), key=lambda i: metrics[i])
    best_value = param_values[best_idx]
    best_metric = metrics[best_idx]

    # Collect neighbor metrics (within 1 step)
    neighbors: List[float] = []
    for offset in [-1, 1]:
        ni = best_idx + offset
        if 0 <= ni < len(metrics):
            neighbors.append(metrics[ni])

    # Stability = average neighbor metric / best metric
    if best_metric != 0 and neighbors:
        avg_neighbor = sum(neighbors) / len(neighbors)
        stability = avg_neighbor / best_metric if best_metric > 0 else 0.0
        stability = min(max(stability, 0.0), 1.0)
    else:
        stability = 0.0

    return ParameterStabilityResult(
        param_name="",
        values_tested=len(param_values),
        best_value=best_value,
        neighbor_metrics=neighbors,
        stability_score=stability,
        is_stable=stability >= stability_threshold,
    )


# ── Probability of Backtest Overfitting (PBO) ────────────────

def probability_of_backtest_overfitting(
    in_sample_sharpes: Sequence[float],
    out_of_sample_sharpes: Sequence[float],
) -> float:
    """Estimate probability of backtest overfitting.

    PBO = fraction of cases where the IS-best strategy
    underperforms the median OOS.

    Uses rank correlation between IS and OOS performance.
    Low PBO (< 0.5) suggests the strategy selection process
    has predictive power beyond random chance.
    """
    if len(in_sample_sharpes) != len(out_of_sample_sharpes):
        raise ValueError("IS and OOS arrays must have same length")
    if len(in_sample_sharpes) < 2:
        return 0.5  # uninformative

    n = len(in_sample_sharpes)

    # Find IS-best strategy index
    best_is_idx = max(range(n), key=lambda i: in_sample_sharpes[i])

    # Check if IS-best also performs well OOS
    oos_best = out_of_sample_sharpes[best_is_idx]
    oos_median = sorted(out_of_sample_sharpes)[n // 2]

    # Simple PBO: what fraction of random picks would beat the IS-best OOS?
    n_better = sum(1 for s in out_of_sample_sharpes if s >= oos_best)
    pbo = n_better / n

    return pbo
