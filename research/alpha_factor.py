"""Alpha factor definition and quality evaluation.

Core module: define what a factor is, evaluate how good it is.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from runner.backtest.csv_io import OhlcvBar


@dataclass(frozen=True)
class AlphaFactor:
    """A named factor with a compute function over OHLCV bars."""

    name: str
    compute_fn: Callable[[Sequence[OhlcvBar]], List[Optional[float]]]
    category: str = "custom"


@dataclass(frozen=True)
class FactorReport:
    """Comprehensive single-factor quality report."""

    name: str
    ic_mean: float
    ic_std: float
    ic_ir: float
    ic_t_stat: float
    rank_ic_mean: float
    rank_ic_std: float
    rank_ic_ir: float
    pct_positive_ic: float
    decay_profile: List[float]
    avg_turnover: float
    factor_autocorr: float
    n_observations: int


@dataclass(frozen=True)
class ComparisonReport:
    """Multi-factor comparison report."""

    factor_reports: List[FactorReport]
    correlation_matrix: Dict[str, Dict[str, float]]
    marginal_ic: Dict[str, float]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pearson_corr(x: List[float], y: List[float]) -> float:
    """Pearson correlation coefficient. Returns 0.0 for degenerate inputs."""
    n = len(x)
    if n < 2 or len(y) != n:
        return 0.0
    mx = sum(x) / n
    my = sum(y) / n
    vx = sum((xi - mx) ** 2 for xi in x)
    vy = sum((yi - my) ** 2 for yi in y)
    if vx < 1e-12 or vy < 1e-12:
        return 0.0
    cov = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    return cov / math.sqrt(vx * vy)


def _rank(vals: List[float]) -> List[float]:
    """Average rank (1-based) for tie handling."""
    n = len(vals)
    indexed = sorted(range(n), key=lambda i: vals[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n - 1 and vals[indexed[j + 1]] == vals[indexed[j]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[indexed[k]] = avg_rank
        i = j + 1
    return ranks


def _spearman_rank_corr(x: List[float], y: List[float]) -> float:
    """Spearman rank correlation."""
    if len(x) < 2 or len(x) != len(y):
        return 0.0
    return _pearson_corr(_rank(x), _rank(y))


def _rolling_ic(
    factor_vals: List[Optional[float]],
    fwd_rets: List[Optional[float]],
    window: int,
) -> List[float]:
    """Compute rolling Pearson IC over windows."""
    n = len(factor_vals)
    ics: List[float] = []
    for end in range(window, n + 1):
        xs: List[float] = []
        ys: List[float] = []
        for i in range(end - window, end):
            fv = factor_vals[i]
            fr = fwd_rets[i]
            if fv is not None and fr is not None:
                xs.append(fv)
                ys.append(fr)
        if len(xs) >= 20:
            ics.append(_pearson_corr(xs, ys))
    return ics


def _factor_turnover(values: List[Optional[float]]) -> float:
    """Average absolute change in factor values (normalized)."""
    clean = [v for v in values if v is not None]
    if len(clean) < 2:
        return 0.0
    diffs = [abs(clean[i] - clean[i - 1]) for i in range(1, len(clean))]
    mean_abs = sum(abs(v) for v in clean) / len(clean)
    if mean_abs < 1e-12:
        return 0.0
    return sum(diffs) / len(diffs) / mean_abs


# ---------------------------------------------------------------------------
# Core API
# ---------------------------------------------------------------------------

def compute_forward_returns(
    bars: Sequence[OhlcvBar], horizon: int,
) -> List[Optional[float]]:
    """Compute forward returns: ret[i] = close[i+horizon]/close[i] - 1."""
    n = len(bars)
    result: List[Optional[float]] = [None] * n
    for i in range(n - horizon):
        c_now = float(bars[i].c)
        c_fwd = float(bars[i + horizon].c)
        if c_now != 0:
            result[i] = c_fwd / c_now - 1.0
    return result


def evaluate_factor(
    factor: AlphaFactor,
    bars: Sequence[OhlcvBar],
    horizons: Sequence[int] = (1, 5, 10, 24),
    ic_window: int = 100,
) -> FactorReport:
    """Full factor evaluation → FactorReport."""
    factor_vals = factor.compute_fn(bars)
    primary_horizon = horizons[0]
    fwd_rets = compute_forward_returns(bars, primary_horizon)

    # Paired valid observations
    xs: List[float] = []
    ys: List[float] = []
    for fv, fr in zip(factor_vals, fwd_rets):
        if fv is not None and fr is not None:
            xs.append(fv)
            ys.append(fr)

    n_obs = len(xs)

    if n_obs < 20:
        return FactorReport(
            name=factor.name, ic_mean=0.0, ic_std=0.0, ic_ir=0.0,
            ic_t_stat=0.0, rank_ic_mean=0.0, rank_ic_std=0.0,
            rank_ic_ir=0.0, pct_positive_ic=0.0, decay_profile=[],
            avg_turnover=0.0, factor_autocorr=0.0, n_observations=n_obs,
        )

    # Rolling IC series
    ics = _rolling_ic(factor_vals, fwd_rets, ic_window)
    if len(ics) < 2:
        ic_mean = _pearson_corr(xs, ys)
        ic_std = 0.0
    else:
        ic_mean = sum(ics) / len(ics)
        ic_std = math.sqrt(sum((ic - ic_mean) ** 2 for ic in ics) / (len(ics) - 1))

    ic_ir = ic_mean / ic_std if ic_std > 1e-12 else 0.0
    ic_t = ic_ir * math.sqrt(len(ics)) if ics else 0.0

    pct_pos = sum(1 for ic in ics if ic > 0) / len(ics) if ics else 0.0

    # Rank IC
    rank_ics: List[float] = []
    for end in range(ic_window, len(factor_vals) + 1):
        rxs: List[float] = []
        rys: List[float] = []
        for i in range(end - ic_window, end):
            fv = factor_vals[i]
            fr = fwd_rets[i]
            if fv is not None and fr is not None:
                rxs.append(fv)
                rys.append(fr)
        if len(rxs) >= 20:
            rank_ics.append(_spearman_rank_corr(rxs, rys))

    if len(rank_ics) < 2:
        rank_ic_mean = _spearman_rank_corr(xs, ys)
        rank_ic_std = 0.0
    else:
        rank_ic_mean = sum(rank_ics) / len(rank_ics)
        rank_ic_std = math.sqrt(
            sum((r - rank_ic_mean) ** 2 for r in rank_ics) / (len(rank_ics) - 1)
        )
    rank_ic_ir = rank_ic_mean / rank_ic_std if rank_ic_std > 1e-12 else 0.0

    # Decay profile
    decay: List[float] = []
    for h in horizons:
        h_rets = compute_forward_returns(bars, h)
        hxs: List[float] = []
        hys: List[float] = []
        for fv, fr in zip(factor_vals, h_rets):
            if fv is not None and fr is not None:
                hxs.append(fv)
                hys.append(fr)
        decay.append(_pearson_corr(hxs, hys) if len(hxs) >= 20 else 0.0)

    # Turnover & autocorrelation
    turnover = _factor_turnover(factor_vals)
    clean_vals = [v for v in factor_vals if v is not None]
    if len(clean_vals) >= 2:
        autocorr = _pearson_corr(clean_vals[:-1], clean_vals[1:])
    else:
        autocorr = 0.0

    return FactorReport(
        name=factor.name,
        ic_mean=ic_mean,
        ic_std=ic_std,
        ic_ir=ic_ir,
        ic_t_stat=ic_t,
        rank_ic_mean=rank_ic_mean,
        rank_ic_std=rank_ic_std,
        rank_ic_ir=rank_ic_ir,
        pct_positive_ic=pct_pos,
        decay_profile=decay,
        avg_turnover=turnover,
        factor_autocorr=autocorr,
        n_observations=n_obs,
    )


def compare_factors(
    factors: Sequence[AlphaFactor],
    bars: Sequence[OhlcvBar],
    horizons: Sequence[int] = (1, 5, 10, 24),
    ic_window: int = 100,
) -> ComparisonReport:
    """Compare multiple factors — reports, cross-correlation, marginal IC."""
    reports: List[FactorReport] = []
    all_vals: Dict[str, List[Optional[float]]] = {}

    for f in factors:
        report = evaluate_factor(f, bars, horizons, ic_window)
        reports.append(report)
        all_vals[f.name] = f.compute_fn(bars)

    # Cross-correlation matrix
    names = [f.name for f in factors]
    corr_matrix: Dict[str, Dict[str, float]] = {}
    for a in names:
        corr_matrix[a] = {}
        for b in names:
            va = all_vals[a]
            vb = all_vals[b]
            xs: List[float] = []
            ys: List[float] = []
            for fa, fb in zip(va, vb):
                if fa is not None and fb is not None:
                    xs.append(fa)
                    ys.append(fb)
            corr_matrix[a][b] = _pearson_corr(xs, ys) if len(xs) >= 20 else 0.0

    # Marginal IC via OLS orthogonalization
    fwd_rets = compute_forward_returns(bars, horizons[0])
    from research.orthogonalize import marginal_ic_ols
    marginal = marginal_ic_ols(all_vals, fwd_rets)

    return ComparisonReport(
        factor_reports=reports,
        correlation_matrix=corr_matrix,
        marginal_ic=marginal,
    )
