"""Tests for state snapshot."""
from __future__ import annotations

from state.snapshot import StateSnapshot


def test_state_snapshot_importable():
    """StateSnapshot is importable and has required API."""
    assert hasattr(StateSnapshot, "of")
    assert callable(StateSnapshot.of)
