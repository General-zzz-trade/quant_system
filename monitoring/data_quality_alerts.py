"""Data quality alerting — bridges BarValidator and DriftMonitor into the alert system."""
from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Optional

from monitoring.alerts.base import Alert, AlertSink, Severity

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DataQualityConfig:
    """Configuration for data quality monitoring."""
    # NaN/missing threshold: alert if fraction exceeds this
    nan_fraction_threshold: float = 0.05
    # Distribution shift: z-score threshold for mean shift detection
    mean_shift_z_threshold: float = 3.0
    # Minimum observations before checking distribution
    min_observations: int = 50
    # Rolling window for distribution tracking
    window_size: int = 200


class DataQualityMonitor:
    """Monitors incoming feature data for quality issues.

    Checks:
    1. NaN/missing values in features
    2. Distribution shift (mean/std change) via z-score
    3. Constant features (zero variance)
    """

    def __init__(
        self,
        config: Optional[DataQualityConfig] = None,
        sink: Optional[AlertSink] = None,
    ) -> None:
        self._cfg = config or DataQualityConfig()
        self._sink = sink
        self._feature_stats: Dict[str, _FeatureTracker] = {}
        self._nan_counts: Dict[str, int] = {}
        self._total_counts: Dict[str, int] = {}

    def on_features(self, features: Mapping[str, Any], symbol: str = "") -> list[str]:
        """Check a feature dict for quality issues. Returns list of issue descriptions."""
        issues: list[str] = []

        for name, value in features.items():
            self._total_counts[name] = self._total_counts.get(name, 0) + 1

            # Check NaN/None
            if value is None or (isinstance(value, float) and math.isnan(value)):
                self._nan_counts[name] = self._nan_counts.get(name, 0) + 1
                total = self._total_counts[name]
                nan_frac = self._nan_counts[name] / total
                if total >= self._cfg.min_observations and nan_frac > self._cfg.nan_fraction_threshold:
                    issues.append(f"feature '{name}' NaN rate {nan_frac:.1%} exceeds threshold")
                continue

            # Track distribution
            try:
                val_f = float(value)
            except (TypeError, ValueError):
                continue

            if name not in self._feature_stats:
                self._feature_stats[name] = _FeatureTracker(self._cfg.window_size)
            tracker = self._feature_stats[name]
            shift = tracker.push(val_f)

            if shift is not None and abs(shift) > self._cfg.mean_shift_z_threshold:
                issues.append(
                    f"feature '{name}' distribution shift z={shift:.2f} "
                    f"(mean {tracker.baseline_mean:.4f} → {tracker.current_mean:.4f})"
                )

            if tracker.is_constant and tracker.count >= self._cfg.min_observations:
                issues.append(f"feature '{name}' has zero variance (constant)")

        if issues and self._sink is not None:
            self._sink.emit(Alert(
                title="Data quality issue",
                message=f"{symbol}: " + "; ".join(issues[:3]),
                severity=Severity.WARNING,
                source="data_quality_monitor",
                ts=datetime.now(timezone.utc),
                meta={"symbol": symbol, "issues": issues},
            ))

        return issues

    def get_stats(self) -> Dict[str, Dict[str, Any]]:
        """Return current statistics for all tracked features."""
        result: Dict[str, Dict[str, Any]] = {}
        for name, tracker in self._feature_stats.items():
            total = self._total_counts.get(name, 0)
            nans = self._nan_counts.get(name, 0)
            result[name] = {
                "count": tracker.count,
                "mean": tracker.current_mean,
                "std": tracker.current_std,
                "nan_rate": nans / total if total > 0 else 0.0,
            }
        return result


class _FeatureTracker:
    """Tracks rolling mean/std for a single feature with baseline comparison."""

    __slots__ = ("_window", "_buf", "_sum", "_sumsq", "_baseline_mean", "_baseline_std", "_baseline_set", "count")

    def __init__(self, window: int) -> None:
        self._window = window
        self._buf: deque[float] = deque(maxlen=window)
        self._sum = 0.0
        self._sumsq = 0.0
        self._baseline_mean: Optional[float] = None
        self._baseline_std: Optional[float] = None
        self._baseline_set = False
        self.count = 0

    def push(self, value: float) -> Optional[float]:
        """Push a value and return z-score of mean shift if baseline is established."""
        if len(self._buf) == self._window:
            old = self._buf[0]
            self._sum -= old
            self._sumsq -= old * old
        self._buf.append(value)
        self._sum += value
        self._sumsq += value * value
        self.count += 1

        n = len(self._buf)
        if n < self._window:
            return None

        # Set baseline on first full window
        if not self._baseline_set:
            self._baseline_mean = self._sum / n
            var = self._sumsq / n - (self._sum / n) ** 2
            self._baseline_std = math.sqrt(max(var, 0.0))
            self._baseline_set = True
            return None

        # Compare current mean to baseline
        if self._baseline_std is not None and self._baseline_std > 1e-12:
            current_mean = self._sum / n
            z = (current_mean - self._baseline_mean) / (self._baseline_std / math.sqrt(n))
            return z
        return None

    @property
    def baseline_mean(self) -> float:
        return self._baseline_mean or 0.0

    @property
    def current_mean(self) -> float:
        n = len(self._buf)
        return self._sum / n if n > 0 else 0.0

    @property
    def current_std(self) -> float:
        n = len(self._buf)
        if n < 2:
            return 0.0
        var = self._sumsq / n - (self._sum / n) ** 2
        return math.sqrt(max(var, 0.0))

    @property
    def is_constant(self) -> bool:
        return len(self._buf) >= 10 and self.current_std < 1e-12
