"""Backward-compat: redirects to alpha.retrain.pipeline."""
from alpha.retrain.pipeline import (  # noqa: F401
    _check_model_age_hours,
    retrain_symbol,
    _model_dir_for,
)
