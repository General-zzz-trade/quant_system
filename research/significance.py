# research/significance.py
"""Factor significance tests — t-test, multiple testing correction, minimum OOS period."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Normal CDF (reuse Abramowitz & Stegun via erf from math)
# ---------------------------------------------------------------------------

def _norm_cdf(x: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _approx_inverse_norm(p: float) -> float:
    """Approximate inverse normal CDF (rational approximation)."""
    if p <= 0:
        return -6.0
    if p >= 1:
        return 6.0
    if p == 0.5:
        return 0.0
    t = math.sqrt(-2.0 * math.log(min(p, 1.0 - p)))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    result = t - (c0 + c1 * t + c2 * t * t) / (1 + d1 * t + d2 * t * t + d3 * t ** 3)
    return result if p > 0.5 else -result


# ---------------------------------------------------------------------------
# t-distribution CDF approximation
# ---------------------------------------------------------------------------

def _t_cdf(t_val: float, df: int) -> float:
    """CDF of the t-distribution.

    For df >= 30, use the normal approximation.
    For small df, use the regularized incomplete beta function.
    """
    if df >= 30:
        return _norm_cdf(t_val)
    # Use incomplete beta: P(T <= t) = 1 - 0.5 * I_x(df/2, 0.5)
    # where x = df / (df + t^2)
    x = df / (df + t_val * t_val)
    a = df / 2.0
    b = 0.5
    ibeta = _regularized_incomplete_beta(x, a, b)
    if t_val >= 0:
        return 1.0 - 0.5 * ibeta
    else:
        return 0.5 * ibeta


def _regularized_incomplete_beta(x: float, a: float, b: float) -> float:
    """Regularized incomplete beta function I_x(a, b) via continued fraction.

    Uses the Lentz algorithm for the continued fraction expansion.
    """
    if x < 0.0 or x > 1.0:
        return 0.0
    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0

    # Use symmetry relation when x > (a+1)/(a+b+2) for better convergence
    if x > (a + 1.0) / (a + b + 2.0):
        return 1.0 - _regularized_incomplete_beta(1.0 - x, b, a)

    # Log of the prefactor: x^a * (1-x)^b / (a * B(a,b))
    ln_prefix = (
        a * math.log(x) + b * math.log(1.0 - x)
        - math.log(a)
        - _log_beta(a, b)
    )

    # Continued fraction (Lentz method)
    cf = _beta_cf(x, a, b)
    return math.exp(ln_prefix) * cf


def _log_beta(a: float, b: float) -> float:
    """Log of the Beta function: log(B(a,b)) = lgamma(a) + lgamma(b) - lgamma(a+b)."""
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def _beta_cf(x: float, a: float, b: float) -> float:
    """Continued fraction for incomplete beta (Lentz algorithm)."""
    max_iter = 200
    tiny = 1e-30
    eps = 1e-10

    # Modified Lentz method
    f = 1.0
    c = 1.0
    d = 1.0 - (a + b) * x / (a + 1.0)
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    f = d

    for m in range(1, max_iter + 1):
        # Even step: d_{2m}
        m2 = 2 * m
        num = m * (b - m) * x / ((a + m2 - 1) * (a + m2))
        d = 1.0 + num * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + num / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        f *= c * d

        # Odd step: d_{2m+1}
        num = -(a + m) * (a + b + m) * x / ((a + m2) * (a + m2 + 1))
        d = 1.0 + num * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + num / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = c * d
        f *= delta

        if abs(delta - 1.0) < eps:
            break

    return f


# ---------------------------------------------------------------------------
# t-test
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class TTestResult:
    """Result of a one-sample t-test on factor IC series."""
    factor_name: str
    t_stat: float
    p_value: float
    mean_ic: float
    std_ic: float
    is_significant: bool
    n_observations: int


def ic_ttest(
    factor_name: str,
    ic_series: List[float],
    alpha: float = 0.05,
) -> TTestResult:
    """One-sample t-test: H0: mean(IC) = 0.

    Args:
        factor_name: Name of the factor.
        ic_series: Series of IC values.
        alpha: Significance level.

    Returns:
        TTestResult with t-statistic, p-value, and significance flag.
    """
    n = len(ic_series)
    if n < 2:
        return TTestResult(
            factor_name=factor_name,
            t_stat=0.0,
            p_value=1.0,
            mean_ic=ic_series[0] if n == 1 else 0.0,
            std_ic=0.0,
            is_significant=False,
            n_observations=n,
        )

    mean_ic = sum(ic_series) / n
    var = sum((x - mean_ic) ** 2 for x in ic_series) / (n - 1)
    std_ic = math.sqrt(var)

    if std_ic < 1e-15:
        # All values identical — if mean != 0, infinitely significant
        p = 0.0 if abs(mean_ic) > 1e-15 else 1.0
        return TTestResult(
            factor_name=factor_name,
            t_stat=float("inf") if mean_ic > 0 else float("-inf") if mean_ic < 0 else 0.0,
            p_value=p,
            mean_ic=mean_ic,
            std_ic=std_ic,
            is_significant=p < alpha,
            n_observations=n,
        )

    t_stat = mean_ic / (std_ic / math.sqrt(n))
    # Two-sided p-value
    df = n - 1
    p_value = 2.0 * (1.0 - _t_cdf(abs(t_stat), df))
    p_value = max(0.0, min(1.0, p_value))

    return TTestResult(
        factor_name=factor_name,
        t_stat=t_stat,
        p_value=p_value,
        mean_ic=mean_ic,
        std_ic=std_ic,
        is_significant=p_value < alpha,
        n_observations=n,
    )


# ---------------------------------------------------------------------------
# Multiple testing correction
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class MultipleTestingResult:
    """Result after applying multiple testing correction."""
    results: Tuple[TTestResult, ...]
    method: str
    n_significant: int
    significant_factors: Tuple[str, ...]


def multiple_testing_correction(
    results: List[TTestResult],
    method: str = "holm",
    alpha: float = 0.05,
) -> MultipleTestingResult:
    """Apply multiple testing correction to a set of t-test results.

    Args:
        results: List of TTestResult from ic_ttest.
        method: "bonferroni", "holm", or "fdr_bh".
        alpha: Family-wise error rate.

    Returns:
        MultipleTestingResult with corrected significance flags.
    """
    n = len(results)
    if n == 0:
        return MultipleTestingResult(
            results=(), method=method, n_significant=0, significant_factors=(),
        )

    if method == "bonferroni":
        adjusted = _bonferroni(results, alpha)
    elif method == "holm":
        adjusted = _holm(results, alpha)
    elif method == "fdr_bh":
        adjusted = _fdr_bh(results, alpha)
    else:
        raise ValueError(f"Unknown method: {method}")

    sig_factors = tuple(r.factor_name for r in adjusted if r.is_significant)
    return MultipleTestingResult(
        results=tuple(adjusted),
        method=method,
        n_significant=len(sig_factors),
        significant_factors=sig_factors,
    )


def _bonferroni(results: List[TTestResult], alpha: float) -> List[TTestResult]:
    """Bonferroni correction: reject if p < alpha/n."""
    n = len(results)
    adjusted: List[TTestResult] = []
    for r in results:
        adj_p = min(r.p_value * n, 1.0)
        adjusted.append(TTestResult(
            factor_name=r.factor_name,
            t_stat=r.t_stat,
            p_value=adj_p,
            mean_ic=r.mean_ic,
            std_ic=r.std_ic,
            is_significant=adj_p < alpha,
            n_observations=r.n_observations,
        ))
    return adjusted


def _holm(results: List[TTestResult], alpha: float) -> List[TTestResult]:
    """Holm step-down correction."""
    n = len(results)
    indexed = sorted(range(n), key=lambda i: results[i].p_value)
    significant = [False] * n

    for rank, idx in enumerate(indexed):
        adj_alpha = alpha / (n - rank)
        if results[idx].p_value < adj_alpha:
            significant[idx] = True
        else:
            break  # Step-down: stop at first non-rejection

    adjusted: List[TTestResult] = []
    for i, r in enumerate(results):
        adjusted.append(TTestResult(
            factor_name=r.factor_name,
            t_stat=r.t_stat,
            p_value=r.p_value,
            mean_ic=r.mean_ic,
            std_ic=r.std_ic,
            is_significant=significant[i],
            n_observations=r.n_observations,
        ))
    return adjusted


def _fdr_bh(results: List[TTestResult], alpha: float) -> List[TTestResult]:
    """Benjamini-Hochberg FDR correction."""
    n = len(results)
    indexed = sorted(range(n), key=lambda i: results[i].p_value)
    significant = [False] * n

    # Step-up: find largest k where p_(k) <= k/n * alpha
    max_k = -1
    for rank, idx in enumerate(indexed):
        k = rank + 1
        if results[idx].p_value <= k / n * alpha:
            max_k = rank

    # All with rank <= max_k are significant
    if max_k >= 0:
        for rank in range(max_k + 1):
            significant[indexed[rank]] = True

    adjusted: List[TTestResult] = []
    for i, r in enumerate(results):
        adjusted.append(TTestResult(
            factor_name=r.factor_name,
            t_stat=r.t_stat,
            p_value=r.p_value,
            mean_ic=r.mean_ic,
            std_ic=r.std_ic,
            is_significant=significant[i],
            n_observations=r.n_observations,
        ))
    return adjusted


# ---------------------------------------------------------------------------
# Minimum OOS period
# ---------------------------------------------------------------------------

def minimum_oos_period(
    n_trials: int,
    target_sharpe: float = 1.0,
    alpha: float = 0.05,
    periods_per_year: int = 252,
) -> int:
    """Minimum out-of-sample period to distinguish signal from noise.

    Based on Bailey & Lopez de Prado: the minimum track record length
    needed to reject the null of zero Sharpe at significance level alpha,
    accounting for the multiple testing from n_trials.

    Args:
        n_trials: Number of strategies/factors tested.
        target_sharpe: Annualized Sharpe ratio to detect.
        alpha: Significance level.
        periods_per_year: Trading periods per year.

    Returns:
        Minimum number of trading days.
    """
    if target_sharpe <= 0 or n_trials < 1:
        return 0

    # Effective alpha after Bonferroni-like adjustment
    effective_alpha = alpha / max(n_trials, 1)
    z = _approx_inverse_norm(1.0 - effective_alpha / 2.0)

    # Sharpe per period
    sharpe_per_period = target_sharpe / math.sqrt(periods_per_year)

    # min_n = (z / sharpe_per_period)^2
    min_n = (z / sharpe_per_period) ** 2

    return max(int(math.ceil(min_n)), 1)
