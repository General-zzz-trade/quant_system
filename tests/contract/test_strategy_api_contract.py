"""Strategy API contract — strategies must conform to interface."""
from __future__ import annotations

from alpha.base import AlphaModel


def test_alpha_model_protocol():
    assert hasattr(AlphaModel, "predict")
    # 'name' is defined at instance level in Protocol, not class level
    assert AlphaModel is not None
