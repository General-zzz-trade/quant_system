"""Feature selection based on predictive power."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Mapping, Sequence

from _quant_hotpath import (
    cpp_correlation_select as _cpp_correlation_select,
    cpp_mutual_info_select as _cpp_mutual_info_select,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class FeatureScore:
    """Score assigned to a feature by a selection method."""
    name: str
    score: float
    method: str


class FeatureSelector:
    """Select top features based on predictive power.

    Supports mutual-information and absolute-correlation ranking.
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
        """Binned mutual information estimation via Rust."""
        import math
        n_bins = max(5, int(math.sqrt(len(target))))
        names = list(features.keys())
        target_list = list(target)

        valid_names = [n for n in names if len(features[n]) == len(target_list)]
        if not valid_names:
            return []

        feat_mat = [features[n] if isinstance(features[n], list) else list(features[n])
                    for n in valid_names]
        mi_vec = _cpp_mutual_info_select(feat_mat, target_list, n_bins)
        scores = [FeatureScore(name=n, score=mi_vec[i], method="mutual_info")
                  for i, n in enumerate(valid_names)]
        scores.sort(key=lambda s: s.score, reverse=True)
        return scores[: self._top_k]

    def _correlation_select(
        self,
        features: Mapping[str, Sequence[float]],
        target: Sequence[float],
    ) -> list[FeatureScore]:
        """Absolute Pearson correlation ranking via Rust."""
        target_list = list(target)
        n = len(target_list)
        if n < 2:
            return []

        names = list(features.keys())
        valid_names = [nm for nm in names if len(features[nm]) == n]
        if not valid_names:
            return []

        feat_mat = [features[nm] if isinstance(features[nm], list) else list(features[nm])
                    for nm in valid_names]
        corr_vec = _cpp_correlation_select(feat_mat, target_list)
        scores = [FeatureScore(name=nm, score=corr_vec[i], method="correlation")
                  for i, nm in enumerate(valid_names)]
        scores.sort(key=lambda s: s.score, reverse=True)
        return scores[: self._top_k]
