"""State view contract — views must be read-only."""
from __future__ import annotations

from context.views.base import ContextView


def test_context_view_protocol():
    assert ContextView is not None
