"""Tests for engine clock."""
from __future__ import annotations

from engine.clock import Clock


def test_clock_protocol_exists():
    """Clock is a Protocol — verify it's importable."""
    assert Clock is not None
