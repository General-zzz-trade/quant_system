"""Feature selection based on predictive power."""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Mapping, Sequence

try:
    from _quant_hotpath import (
        cpp_correlation_select as _cpp_correlation_select,
        cpp_mutual_info_select as _cpp_mutual_info_select,
    )
    _USING_CPP = True
except ImportError:
    _USING_CPP = False

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class FeatureScore:
    """Score assigned to a feature by a selection method.

    Attributes:
        name: Feature identifier.
        score: Numeric relevance score (higher = more predictive).
        method: Selection method used (``mutual_info`` or ``correlation``).
    """
    name: str
    score: float
    method: str


class FeatureSelector:
    """Select top features based on predictive power.

    Supports mutual-information and absolute-correlation ranking.
    All computations are pure Python (no sklearn/numpy required).

    Usage:
        selector = FeatureSelector(method="correlation", top_k=5)
        scores = selector.select(features={"sma_20": [...], ...}, target=[...])
    """

    def __init__(self, *, method: str = "mutual_info", top_k: int = 10) -> None:
        self._method = method
        self._top_k = top_k

    def select(
        self,
        features: Mapping[str, Sequence[float]],
        target: Sequence[float],
    ) -> list[FeatureScore]:
        """Score and rank features, return top-k."""
        if self._method == "mutual_info":
            return self._mutual_info_select(features, target)
        elif self._method == "correlation":
            return self._correlation_select(features, target)
        raise ValueError(f"Unknown method: {self._method}")

    def _mutual_info_select(
        self,
        features: Mapping[str, Sequence[float]],
        target: Sequence[float],
    ) -> list[FeatureScore]:
        """Binned mutual information estimation."""
        n_bins = max(5, int(math.sqrt(len(target))))
        names = list(features.keys())
        target_list = list(target)

        if _USING_CPP and names:
            valid_names = [n for n in names if len(features[n]) == len(target_list)]
            if valid_names:
                feat_mat = [features[n] if isinstance(features[n], list) else list(features[n])
                            for n in valid_names]
                mi_vec = _cpp_mutual_info_select(feat_mat, target_list, n_bins)
                scores = [FeatureScore(name=n, score=mi_vec[i], method="mutual_info")
                          for i, n in enumerate(valid_names)]
                scores.sort(key=lambda s: s.score, reverse=True)
                return scores[: self._top_k]

        target_binned = self._bin_values(target_list, n_bins)
        scores: list[FeatureScore] = []
        for name, values in features.items():
            vals = list(values)
            if len(vals) != len(target):
                logger.warning("Feature %s length mismatch, skipping", name)
                continue
            feat_binned = self._bin_values(vals, n_bins)
            mi = self._compute_mi(feat_binned, target_binned, n_bins)
            scores.append(FeatureScore(name=name, score=mi, method="mutual_info"))

        scores.sort(key=lambda s: s.score, reverse=True)
        return scores[: self._top_k]

    def _correlation_select(
        self,
        features: Mapping[str, Sequence[float]],
        target: Sequence[float],
    ) -> list[FeatureScore]:
        """Absolute Pearson correlation ranking."""
        target_list = list(target)
        n = len(target_list)
        if n < 2:
            return []

        names = list(features.keys())

        if _USING_CPP and names:
            valid_names = [nm for nm in names if len(features[nm]) == n]
            if valid_names:
                feat_mat = [features[nm] if isinstance(features[nm], list) else list(features[nm])
                            for nm in valid_names]
                corr_vec = _cpp_correlation_select(feat_mat, target_list)
                scores = [FeatureScore(name=nm, score=corr_vec[i], method="correlation")
                          for i, nm in enumerate(valid_names)]
                scores.sort(key=lambda s: s.score, reverse=True)
                return scores[: self._top_k]

        t_mean = sum(target_list) / n
        t_var = sum((v - t_mean) ** 2 for v in target_list)

        scores: list[FeatureScore] = []
        for name, values in features.items():
            vals = list(values)
            if len(vals) != n:
                logger.warning("Feature %s length mismatch, skipping", name)
                continue

            f_mean = sum(vals) / n
            f_var = sum((v - f_mean) ** 2 for v in vals)

            if f_var < 1e-12 or t_var < 1e-12:
                corr = 0.0
            else:
                cov = sum((v - f_mean) * (t - t_mean) for v, t in zip(vals, target_list))
                corr = cov / math.sqrt(f_var * t_var)

            scores.append(
                FeatureScore(name=name, score=abs(corr), method="correlation"),
            )

        scores.sort(key=lambda s: s.score, reverse=True)
        return scores[: self._top_k]

    @staticmethod
    def _bin_values(values: list[float], n_bins: int) -> list[int]:
        """Bin continuous values into integer bin indices."""
        if not values:
            return []
        v_min = min(values)
        v_max = max(values)
        if v_max - v_min < 1e-12:
            return [0] * len(values)
        scale = (n_bins - 1) / (v_max - v_min)
        return [min(int((v - v_min) * scale), n_bins - 1) for v in values]

    @staticmethod
    def _compute_mi(x_bins: list[int], y_bins: list[int], n_bins: int) -> float:
        """Compute mutual information from binned values."""
        n = len(x_bins)
        if n == 0:
            return 0.0

        joint: dict[tuple[int, int], int] = {}
        x_counts: dict[int, int] = {}
        y_counts: dict[int, int] = {}

        for xi, yi in zip(x_bins, y_bins):
            joint[(xi, yi)] = joint.get((xi, yi), 0) + 1
            x_counts[xi] = x_counts.get(xi, 0) + 1
            y_counts[yi] = y_counts.get(yi, 0) + 1

        mi = 0.0
        for (xi, yi), count in joint.items():
            p_xy = count / n
            p_x = x_counts[xi] / n
            p_y = y_counts[yi] / n
            if p_xy > 0 and p_x > 0 and p_y > 0:
                mi += p_xy * math.log(p_xy / (p_x * p_y))

        return max(mi, 0.0)
