"""Tests for context views."""
from __future__ import annotations

from context.views.base import ContextView


def test_context_view_protocol():
    """ContextView is a Protocol — verify it's importable."""
    assert ContextView is not None
