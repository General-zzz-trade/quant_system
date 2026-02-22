"""Tests for context snapshot."""
from __future__ import annotations

from context.snapshot import SnapshotHistory


def test_snapshot_history_empty():
    history = SnapshotHistory()
    assert history.latest is None


def test_snapshot_history_max_size():
    history = SnapshotHistory(max_size=5)
    assert history.latest is None
