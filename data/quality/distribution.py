"""Rolling distribution tracker for data pipeline quality monitoring."""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence


@dataclass(frozen=True, slots=True)
class DistributionSnapshot:
    """Point-in-time distribution statistics."""
    feature: str
    count: int
    mean: float
    std: float
    min_val: float
    max_val: float
    skew: float
    kurtosis: float


@dataclass(frozen=True, slots=True)
class ShiftAlert:
    """Distribution shift detection result."""
    feature: str
    metric: str  # "mean", "std", "range"
    baseline_value: float
    current_value: float
    z_score: float
    severity: str  # "warning", "critical"


class RollingDistribution:
    """Tracks rolling distribution statistics for a single feature.

    Computes mean, std, skew, kurtosis using online algorithms.
    Detects distribution shifts between baseline and current window.
    """

    def __init__(
        self,
        window: int = 500,
        baseline_window: int = 1000,
        shift_z_threshold: float = 3.0,
        critical_z_threshold: float = 5.0,
    ) -> None:
        self._window = window
        self._baseline_window = baseline_window
        self._shift_z = shift_z_threshold
        self._critical_z = critical_z_threshold

        self._current: deque[float] = deque(maxlen=window)
        self._baseline: deque[float] = deque(maxlen=baseline_window)
        self._baseline_frozen = False
        self._count = 0

    def push(self, value: float) -> list[ShiftAlert]:
        """Push a value and return any detected shifts."""
        self._count += 1
        self._current.append(value)

        if not self._baseline_frozen:
            self._baseline.append(value)
            if len(self._baseline) == self._baseline_window:
                self._baseline_frozen = True
            return []

        if len(self._current) < self._window:
            return []

        return self._check_shifts()

    def _check_shifts(self) -> list[ShiftAlert]:
        alerts: list[ShiftAlert] = []
        baseline_stats = _compute_stats(list(self._baseline))
        current_stats = _compute_stats(list(self._current))

        if baseline_stats is None or current_stats is None:
            return alerts

        # Mean shift
        if baseline_stats["std"] > 1e-12:
            n = len(self._current)
            z_mean = abs(current_stats["mean"] - baseline_stats["mean"]) / (baseline_stats["std"] / math.sqrt(n))
            if z_mean > self._critical_z:
                alerts.append(ShiftAlert(
                    feature="", metric="mean",
                    baseline_value=baseline_stats["mean"],
                    current_value=current_stats["mean"],
                    z_score=z_mean, severity="critical",
                ))
            elif z_mean > self._shift_z:
                alerts.append(ShiftAlert(
                    feature="", metric="mean",
                    baseline_value=baseline_stats["mean"],
                    current_value=current_stats["mean"],
                    z_score=z_mean, severity="warning",
                ))

        # Std shift (F-test approximation)
        if baseline_stats["std"] > 1e-12 and current_stats["std"] > 1e-15:
            ratio = current_stats["std"] / baseline_stats["std"]
            if ratio > 2.0 or ratio < 0.5:
                log_ratio = abs(math.log(max(ratio, 1e-10))) * 3
                alerts.append(ShiftAlert(
                    feature="", metric="std",
                    baseline_value=baseline_stats["std"],
                    current_value=current_stats["std"],
                    z_score=log_ratio,
                    severity="critical" if ratio > 3.0 or ratio < 0.33 else "warning",
                ))

        return alerts

    def snapshot(self, feature: str = "") -> Optional[DistributionSnapshot]:
        if len(self._current) < 3:
            return None
        stats = _compute_stats(list(self._current))
        if stats is None:
            return None
        return DistributionSnapshot(
            feature=feature,
            count=len(self._current),
            mean=stats["mean"],
            std=stats["std"],
            min_val=stats["min"],
            max_val=stats["max"],
            skew=stats["skew"],
            kurtosis=stats["kurtosis"],
        )


class DistributionTracker:
    """Tracks multiple features' distributions simultaneously."""

    def __init__(
        self,
        window: int = 500,
        baseline_window: int = 1000,
        shift_z_threshold: float = 3.0,
    ) -> None:
        self._window = window
        self._baseline_window = baseline_window
        self._shift_z = shift_z_threshold
        self._trackers: Dict[str, RollingDistribution] = {}

    def on_observation(self, features: Dict[str, float]) -> list[ShiftAlert]:
        """Push an observation for all features. Returns detected shifts."""
        all_alerts: list[ShiftAlert] = []
        for name, value in features.items():
            if not isinstance(value, (int, float)) or math.isnan(value):
                continue
            if name not in self._trackers:
                self._trackers[name] = RollingDistribution(
                    window=self._window,
                    baseline_window=self._baseline_window,
                    shift_z_threshold=self._shift_z,
                )
            alerts = self._trackers[name].push(value)
            for a in alerts:
                all_alerts.append(ShiftAlert(
                    feature=name,
                    metric=a.metric,
                    baseline_value=a.baseline_value,
                    current_value=a.current_value,
                    z_score=a.z_score,
                    severity=a.severity,
                ))
        return all_alerts

    def snapshots(self) -> Dict[str, DistributionSnapshot]:
        result: Dict[str, DistributionSnapshot] = {}
        for name, tracker in self._trackers.items():
            snap = tracker.snapshot(name)
            if snap is not None:
                result[name] = snap
        return result


def _compute_stats(values: list[float]) -> Optional[Dict[str, float]]:
    """Compute mean, std, skew, kurtosis from a list of values."""
    n = len(values)
    if n < 3:
        return None

    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    std = math.sqrt(max(var, 0.0))

    skew = 0.0
    kurtosis = 0.0
    if std > 1e-12:
        m3 = sum((v - mean) ** 3 for v in values) / n
        m4 = sum((v - mean) ** 4 for v in values) / n
        skew = m3 / (std ** 3)
        kurtosis = m4 / (std ** 4) - 3.0  # excess kurtosis

    return {
        "mean": mean,
        "std": std,
        "min": min(values),
        "max": max(values),
        "skew": skew,
        "kurtosis": kurtosis,
    }
