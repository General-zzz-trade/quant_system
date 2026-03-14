"""Centralized feature catalog — single source of truth for production features.

All production models must declare features from this catalog.
Training scripts, model loaders, and feature hooks should validate against it.
"""
from __future__ import annotations

from features.enriched_computer import ENRICHED_FEATURE_NAMES

# Try to import cross-asset feature names
try:
    from features.cross_asset_computer import CROSS_ASSET_FEATURE_NAMES
except ImportError:
    CROSS_ASSET_FEATURE_NAMES = ()

# Canonical set of features available in production
PRODUCTION_FEATURES: frozenset[str] = frozenset(
    list(ENRICHED_FEATURE_NAMES) + list(CROSS_ASSET_FEATURE_NAMES)
)


def validate_model_features(
    declared: list[str] | tuple[str, ...],
    *,
    model_name: str = "",
    strict: bool = False,
) -> list[str]:
    """Validate that all declared model features exist in the production catalog.

    Args:
        declared: Feature names declared by a model.
        model_name: Optional model name for error messages.
        strict: If True, raise ValueError on unknown features. If False, return warnings.

    Returns:
        List of warning messages (empty if all features are valid).

    Raises:
        ValueError: If strict=True and unknown features are found.
    """
    unknown = set(declared) - PRODUCTION_FEATURES
    if not unknown:
        return []

    prefix = f"Model {model_name}: " if model_name else ""
    msgs = [f"{prefix}Unknown feature '{f}' not in production catalog" for f in sorted(unknown)]

    if strict:
        raise ValueError(
            f"{prefix}{len(unknown)} features not in production catalog: {sorted(unknown)}"
        )

    return msgs
