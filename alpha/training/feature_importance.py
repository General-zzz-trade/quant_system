"""Feature importance analysis and automatic feature selection.

Works with any model that has a feature_importances_ attribute
(XGBoost, LightGBM, sklearn tree-based models).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class FeatureImportance:
    """Single feature importance score."""
    name: str
    importance: float
    rank: int


def analyze_importance(
    model: Any,
    feature_names: Sequence[str],
) -> List[FeatureImportance]:
    """Extract and rank feature importances from a trained model.

    Supports any model with feature_importances_ attribute.
    """
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        # Try sklearn-style
        importances = getattr(model, "coef_", None)
        if importances is not None:
            importances = [abs(c) for c in importances]

    if importances is None:
        raise ValueError("Model does not expose feature importances")

    pairs = list(zip(feature_names, importances))
    pairs.sort(key=lambda x: x[1], reverse=True)

    return [
        FeatureImportance(name=name, importance=float(imp), rank=rank + 1)
        for rank, (name, imp) in enumerate(pairs)
    ]


def select_top_features(
    model: Any,
    feature_names: Sequence[str],
    *,
    top_k: int = 0,
    min_importance: float = 0.0,
) -> List[str]:
    """Select top features by importance.

    Args:
        top_k: Keep top K features (0 = use min_importance threshold).
        min_importance: Minimum importance threshold.

    Returns:
        List of selected feature names.
    """
    ranked = analyze_importance(model, feature_names)

    if top_k > 0:
        return [f.name for f in ranked[:top_k]]

    return [f.name for f in ranked if f.importance >= min_importance]
