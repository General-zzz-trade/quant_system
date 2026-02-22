"""Tests for context validators."""
from __future__ import annotations

from context.validators import validate_context


def test_validate_context_importable():
    """validate_context is importable and callable."""
    assert callable(validate_context)
