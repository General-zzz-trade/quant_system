"""Alpha factor factory — parametric sweep generation and quality screening.

Generates factor variants by sweeping parameter grids, evaluates quality,
and selects the best uncorrelated subset.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from research.alpha_factor import (
    AlphaFactor,
    FactorReport,
    _pearson_corr,
    evaluate_factor,
)
from runner.backtest.csv_io import OhlcvBar


@dataclass(frozen=True)
class ScreeningConfig:
    """Quality gate thresholds for factor screening."""

    min_ic_ir: float = 0.3
    min_abs_ic: float = 0.02
    max_autocorr: float = 0.95
    min_observations: int = 100
    max_factor_correlation: float = 0.85


@dataclass(frozen=True)
class ScreeningResult:
    """Result of screening a single factor."""

    factor: AlphaFactor
    report: FactorReport
    passed: bool
    reject_reasons: Tuple[str, ...] = ()


class FactorFactory:
    """Generate factor sweeps and screen for quality."""

    def __init__(self, generators: Optional[Mapping[str, Callable]] = None) -> None:
        self._generators = dict(generators) if generators else {}

    def register(self, family: str, generator: Callable) -> None:
        """Register a factor generator function.

        The generator must accept keyword arguments matching the parameter grid
        and return a Callable[[Sequence[OhlcvBar]], List[Optional[float]]].
        """
        self._generators[family] = generator

    def generate_sweep(
        self,
        family: str,
        param_grid: Mapping[str, Sequence],
    ) -> List[AlphaFactor]:
        """Generate all combinations of parameters for a factor family.

        Parameters
        ----------
        family : str
            Factor family name (must be registered).
        param_grid : dict
            Mapping of parameter names to sequences of values.

        Returns
        -------
        List of AlphaFactor instances, one per parameter combination.
        """
        if family not in self._generators:
            raise KeyError(f"Unknown factor family: {family}. "
                           f"Available: {list(self._generators.keys())}")

        generator = self._generators[family]
        keys = list(param_grid.keys())
        values = [param_grid[k] for k in keys]

        factors: List[AlphaFactor] = []
        for combo in product(*values):
            params = dict(zip(keys, combo))
            name = f"{family}_{'_'.join(str(v) for v in combo)}"
            compute_fn = generator(**params)
            factors.append(AlphaFactor(name=name, compute_fn=compute_fn, category=family))

        return factors

    def screen(
        self,
        factors: Sequence[AlphaFactor],
        bars: Sequence[OhlcvBar],
        config: ScreeningConfig = ScreeningConfig(),
        horizons: Sequence[int] = (1, 5, 10, 24),
        ic_window: int = 100,
    ) -> List[ScreeningResult]:
        """Evaluate and screen factors against quality thresholds.

        Returns all results sorted by IC_IR descending (stability-first).
        """
        results: List[ScreeningResult] = []

        for factor in factors:
            report = evaluate_factor(factor, bars, horizons, ic_window)
            reasons: List[str] = []

            if report.n_observations < config.min_observations:
                reasons.append(f"n_obs={report.n_observations} < {config.min_observations}")
            if abs(report.ic_mean) < config.min_abs_ic:
                reasons.append(f"|ic_mean|={abs(report.ic_mean):.4f} < {config.min_abs_ic}")
            if abs(report.ic_ir) < config.min_ic_ir:
                reasons.append(f"|ic_ir|={abs(report.ic_ir):.2f} < {config.min_ic_ir}")
            if report.factor_autocorr > config.max_autocorr:
                reasons.append(f"autocorr={report.factor_autocorr:.2f} > {config.max_autocorr}")

            results.append(ScreeningResult(
                factor=factor,
                report=report,
                passed=len(reasons) == 0,
                reject_reasons=tuple(reasons),
            ))

        # Sort by abs(IC_IR) descending — stability first
        results.sort(key=lambda r: abs(r.report.ic_ir), reverse=True)
        return results

    def select_uncorrelated(
        self,
        results: Sequence[ScreeningResult],
        bars: Sequence[OhlcvBar],
        max_correlation: float = 0.85,
    ) -> List[ScreeningResult]:
        """Greedy decorrelation: keep factors in IC_IR order, skip if too correlated.

        Only considers results that passed screening.
        """
        passed = [r for r in results if r.passed]
        if not passed:
            return []

        # Pre-compute factor values
        factor_values: Dict[str, List[Optional[float]]] = {}
        for r in passed:
            factor_values[r.factor.name] = r.factor.compute_fn(bars)

        selected: List[ScreeningResult] = []

        for candidate in passed:
            cand_vals = factor_values[candidate.factor.name]
            too_correlated = False

            for chosen in selected:
                chosen_vals = factor_values[chosen.factor.name]
                corr = _compute_cross_correlation(cand_vals, chosen_vals)
                if abs(corr) > max_correlation:
                    too_correlated = True
                    break

            if not too_correlated:
                selected.append(candidate)

        return selected


def _compute_cross_correlation(
    a: List[Optional[float]], b: List[Optional[float]],
) -> float:
    """Pearson correlation between two factor value series (skipping None)."""
    xs: List[float] = []
    ys: List[float] = []
    for va, vb in zip(a, b):
        if va is not None and vb is not None:
            xs.append(va)
            ys.append(vb)
    if len(xs) < 20:
        return 0.0
    return _pearson_corr(xs, ys)
