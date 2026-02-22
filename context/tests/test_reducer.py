"""Tests for context reducer."""
from __future__ import annotations

from context.reducer import reduce_context


def test_reducer_importable():
    assert callable(reduce_context)
