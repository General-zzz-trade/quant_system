# research/sensitivity.py
"""One-at-a-time parameter sensitivity analysis."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple


@dataclass(frozen=True, slots=True)
class SensitivityResult:
    """Result of sweeping a single parameter."""
    param_name: str
    param_values: Tuple[Any, ...]
    metric_values: Tuple[float, ...]
    best_value: Any
    best_metric: float
    sensitivity_score: float  # std(metrics) / |mean(metrics)| — higher = more sensitive


class SensitivityAnalyzer:
    """One-at-a-time parameter sensitivity analysis.

    For each parameter, hold all others at their base values, sweep the
    parameter through its range, and record the resulting metric.
    """

    def analyze(
        self,
        base_params: Dict[str, Any],
        param_ranges: Dict[str, List[Any]],
        evaluate_fn: Callable[[Dict[str, Any]], float],
    ) -> List[SensitivityResult]:
        """Run sensitivity analysis.

        Args:
            base_params: Default parameter values.
            param_ranges: For each param name, the list of values to try.
            evaluate_fn: Callable(params_dict) -> scalar metric.

        Returns:
            List of SensitivityResult, one per parameter.
        """
        results: List[SensitivityResult] = []

        for param_name, values in param_ranges.items():
            if not values:
                continue

            metrics: List[float] = []
            for val in values:
                params = dict(base_params)
                params[param_name] = val
                metrics.append(evaluate_fn(params))

            best_idx = max(range(len(metrics)), key=lambda i: metrics[i])
            best_value = values[best_idx]
            best_metric = metrics[best_idx]

            sensitivity = _sensitivity_score(metrics)

            results.append(SensitivityResult(
                param_name=param_name,
                param_values=tuple(values),
                metric_values=tuple(metrics),
                best_value=best_value,
                best_metric=best_metric,
                sensitivity_score=sensitivity,
            ))

        return results

    def rank_parameters(
        self, results: List[SensitivityResult],
    ) -> List[Tuple[str, float]]:
        """Rank parameters by sensitivity score (most sensitive first)."""
        ranked = sorted(results, key=lambda r: r.sensitivity_score, reverse=True)
        return [(r.param_name, r.sensitivity_score) for r in ranked]


def _sensitivity_score(metrics: List[float]) -> float:
    """Compute sensitivity as coefficient of variation: std / |mean|."""
    n = len(metrics)
    if n < 2:
        return 0.0
    mean = sum(metrics) / n
    var = sum((m - mean) ** 2 for m in metrics) / (n - 1)
    std = math.sqrt(var)
    if abs(mean) < 1e-12:
        return 0.0
    return std / abs(mean)
