# alpha/monitoring/ood_detector.py
"""Out-of-Distribution detector using Mahalanobis distance (diagonal covariance)."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass(frozen=True, slots=True)
class OODResult:
    is_ood: bool
    score: float  # higher = more OOD
    method: str
    threshold: float


class OODDetector:
    """Detects out-of-distribution inputs using Mahalanobis distance.

    Uses diagonal covariance for efficiency — each feature is scored
    independently via (x_i - mu_i)^2 / var_i, then summed and sqrt'd.
    """

    def __init__(self, z_threshold: float = 3.0) -> None:
        self._z_threshold = z_threshold
        self._mean: Dict[str, float] = {}
        self._var: Dict[str, float] = {}
        self._fitted = False

    @property
    def fitted(self) -> bool:
        return self._fitted

    def fit(self, features: List[Dict[str, float]]) -> None:
        """Compute mean and variance from training data."""
        if not features:
            raise ValueError("Cannot fit on empty data")

        # Collect per-feature values
        accum: Dict[str, List[float]] = {}
        for obs in features:
            for k, v in obs.items():
                if not isinstance(v, (int, float)) or math.isnan(v):
                    continue
                accum.setdefault(k, []).append(float(v))

        if not accum:
            raise ValueError("No valid numeric features found")

        self._mean = {}
        self._var = {}
        for k, vals in accum.items():
            n = len(vals)
            if n < 2:
                continue
            mu = sum(vals) / n
            var = sum((x - mu) ** 2 for x in vals) / n
            self._mean[k] = mu
            # Floor variance to avoid division by zero
            self._var[k] = max(var, 1e-12)

        if not self._mean:
            raise ValueError("Need at least 2 observations per feature to fit")

        self._fitted = True

    def score(self, features: Dict[str, float]) -> OODResult:
        """Score a single observation. Higher = more OOD."""
        if not self._fitted:
            raise RuntimeError("OODDetector not fitted — call fit() first")

        # Only score features we were trained on
        matched = 0
        sq_sum = 0.0
        for k, mu in self._mean.items():
            v = features.get(k)
            if v is None:
                continue
            if not isinstance(v, (int, float)) or math.isnan(v):
                continue
            matched += 1
            sq_sum += (float(v) - mu) ** 2 / self._var[k]

        if matched == 0:
            # No overlap — treat as maximally OOD
            return OODResult(
                is_ood=True,
                score=float("inf"),
                method="mahalanobis_diag",
                threshold=self._z_threshold,
            )

        # Normalize by number of features to get a per-feature z-like score
        distance = math.sqrt(sq_sum / matched)

        return OODResult(
            is_ood=distance > self._z_threshold,
            score=distance,
            method="mahalanobis_diag",
            threshold=self._z_threshold,
        )

    def save(self, path: Path) -> None:
        """Persist fitted parameters to JSON."""
        if not self._fitted:
            raise RuntimeError("Cannot save unfitted detector")
        data = {
            "z_threshold": self._z_threshold,
            "mean": self._mean,
            "var": self._var,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2))

    def load(self, path: Path) -> None:
        """Load fitted parameters from JSON."""
        raw = json.loads(path.read_text())
        self._z_threshold = raw["z_threshold"]
        self._mean = raw["mean"]
        self._var = raw["var"]
        self._fitted = True
