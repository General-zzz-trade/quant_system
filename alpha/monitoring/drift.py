# alpha/monitoring/drift.py
"""ModelDriftMonitor — detect feature distribution shift and prediction degradation."""
from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DriftAlert:
    feature: str
    baseline_mean: float
    baseline_std: float
    current_mean: float
    z_score: float


@dataclass
class ModelDriftMonitor:
    """Tracks rolling feature statistics and alerts when they diverge from training baseline.

    Usage:
        monitor = ModelDriftMonitor(
            baseline_stats={"ma_fast": (100.0, 5.0), "vol": (0.02, 0.005)},
            alert_fn=alert_manager.fire,
        )
        monitor.on_features({"ma_fast": 120.0, "vol": 0.02})
        alerts = monitor.check()
    """

    baseline_stats: Dict[str, tuple[float, float]]  # {feature: (mean, std)}
    window_size: int = 100
    z_threshold: float = 2.0
    alert_fn: Optional[Callable[[str, str], None]] = None

    _windows: Dict[str, Deque[float]] = field(default_factory=dict, init=False)
    _alert_history: List[DriftAlert] = field(default_factory=list, init=False)

    def on_features(self, features: Dict[str, Any]) -> None:
        """Record a feature observation."""
        for name, value in features.items():
            if name not in self.baseline_stats:
                continue
            if value is None:
                continue
            try:
                v = float(value)
            except (TypeError, ValueError):
                continue
            if name not in self._windows:
                self._windows[name] = deque(maxlen=self.window_size)
            self._windows[name].append(v)

    def check(self) -> List[DriftAlert]:
        """Check for drift in all tracked features. Returns list of triggered alerts."""
        alerts = []
        for name, (base_mean, base_std) in self.baseline_stats.items():
            if name not in self._windows:
                continue
            window = self._windows[name]
            if len(window) < self.window_size // 2:
                continue  # not enough data

            current_mean = sum(window) / len(window)
            if base_std <= 0:
                continue

            z = abs(current_mean - base_mean) / base_std
            if z > self.z_threshold:
                alert = DriftAlert(
                    feature=name,
                    baseline_mean=base_mean,
                    baseline_std=base_std,
                    current_mean=current_mean,
                    z_score=z,
                )
                alerts.append(alert)
                self._alert_history.append(alert)
                logger.warning(
                    "Feature drift: %s current_mean=%.4f baseline=%.4f±%.4f z=%.2f",
                    name, current_mean, base_mean, base_std, z,
                )
                if self.alert_fn is not None:
                    self.alert_fn(
                        "model_drift",
                        f"Feature '{name}' drifted: z={z:.2f} (threshold={self.z_threshold})",
                    )

        return alerts

    @property
    def alert_count(self) -> int:
        return len(self._alert_history)
