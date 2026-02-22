"""Tests for engine scheduler."""
from __future__ import annotations

from engine.scheduler import Scheduler


def test_scheduler_protocol_exists():
    """Scheduler is a Protocol — verify it's importable."""
    assert Scheduler is not None
