"""Unit tests for OrderTimeoutTracker with injected clock."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from execution.safety.timeout_tracker import OrderTimeoutTracker


class TestOrderTimeoutTracker:

    def _make_tracker(self, timeout=30.0, start=0.0):
        clock = [start]

        def clock_fn():
            return clock[0]

        tracker = OrderTimeoutTracker(
            timeout_sec=timeout, clock_fn=clock_fn,
        )
        return tracker, clock

    def test_submit_tracks_order(self):
        tracker, _ = self._make_tracker()
        tracker.on_submit("o1", SimpleNamespace())
        assert tracker.pending_count == 1
        assert "o1" in tracker.pending_order_ids

    def test_fill_removes_order(self):
        tracker, _ = self._make_tracker()
        tracker.on_submit("o1")
        tracker.on_fill("o1")
        assert tracker.pending_count == 0

    def test_cancel_removes_order(self):
        tracker, _ = self._make_tracker()
        tracker.on_submit("o1")
        tracker.on_cancel("o1")
        assert tracker.pending_count == 0

    def test_timeout_detected(self):
        tracker, clock = self._make_tracker(timeout=10.0)
        tracker.on_submit("o1", SimpleNamespace())
        clock[0] = 11.0
        timed_out = tracker.check_timeouts()
        assert timed_out == ["o1"]
        assert tracker.pending_count == 0

    def test_no_timeout_before_deadline(self):
        tracker, clock = self._make_tracker(timeout=10.0)
        tracker.on_submit("o1")
        clock[0] = 9.0
        timed_out = tracker.check_timeouts()
        assert timed_out == []
        assert tracker.pending_count == 1

    def test_cancel_fn_called_on_timeout(self):
        cancelled = []
        tracker, clock = self._make_tracker(timeout=5.0)
        tracker.cancel_fn = lambda cmd: cancelled.append(cmd)
        cmd = SimpleNamespace(order_id="o1")
        tracker.on_submit("o1", cmd)
        clock[0] = 6.0
        tracker.check_timeouts()
        assert len(cancelled) == 1
        assert cancelled[0] is cmd

    def test_cancel_fn_exception_handled(self):
        def bad_cancel(cmd):
            raise RuntimeError("cancel failed")

        tracker, clock = self._make_tracker(timeout=5.0)
        tracker.cancel_fn = bad_cancel
        tracker.on_submit("o1", SimpleNamespace())
        clock[0] = 6.0
        # Should not raise
        timed_out = tracker.check_timeouts()
        assert timed_out == ["o1"]

    def test_multiple_orders_partial_timeout(self):
        tracker, clock = self._make_tracker(timeout=10.0, start=0.0)
        tracker.on_submit("o1")
        clock[0] = 5.0
        tracker.on_submit("o2")
        clock[0] = 11.0  # o1 timed out (11-0>10), o2 not (11-5<10)
        timed_out = tracker.check_timeouts()
        assert timed_out == ["o1"]
        assert tracker.pending_count == 1
        assert "o2" in tracker.pending_order_ids

    def test_empty_check_returns_empty(self):
        tracker, _ = self._make_tracker()
        assert tracker.check_timeouts() == []
