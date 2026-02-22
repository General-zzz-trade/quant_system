"""Tests for rate limiting and retry."""
from __future__ import annotations

from execution.bridge.rate_limit import SlidingWindowCounter


def test_sliding_window_allows_initial():
    sw = SlidingWindowCounter(max_count=5, window_sec=1.0)
    assert sw.allow()


def test_sliding_window_blocks_when_full():
    sw = SlidingWindowCounter(max_count=2, window_sec=60.0)
    assert sw.allow()
    assert sw.allow()
    assert not sw.allow()
