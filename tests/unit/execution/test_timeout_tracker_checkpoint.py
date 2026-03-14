"""Tests for OrderTimeoutTracker checkpoint/restore — especially clock skew resilience."""
from __future__ import annotations

from execution.safety.timeout_tracker import OrderTimeoutTracker


class _FakeClock:
    """Controllable monotonic clock for testing."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _make_tracker(clock: _FakeClock, timeout: float = 30.0) -> OrderTimeoutTracker:
    return OrderTimeoutTracker(timeout_sec=timeout, clock_fn=clock)


class TestCheckpointRestore:
    def test_roundtrip_preserves_elapsed(self):
        """Checkpoint → restore preserves elapsed time, not wall-clock."""
        clock = _FakeClock(1000.0)
        tracker = _make_tracker(clock)

        tracker.on_submit("ord-1", cmd="cmd-1")
        clock.advance(10.0)  # 10s elapsed

        cp = tracker.checkpoint()
        assert "elapsed_sec" in cp["pending"]["ord-1"]
        assert abs(cp["pending"]["ord-1"]["elapsed_sec"] - 10.0) < 0.01

        # Restore into fresh tracker (simulating restart)
        clock2 = _FakeClock(5000.0)  # Different monotonic base
        tracker2 = _make_tracker(clock2)
        tracker2.restore(cp)

        # After 21 more seconds (total 31s > 30s timeout), should time out
        clock2.advance(21.0)
        timed_out = tracker2.check_timeouts()
        assert "ord-1" in timed_out

    def test_no_spurious_timeout_after_restore(self):
        """Order that hasn't timed out shouldn't timeout after restore."""
        clock = _FakeClock(1000.0)
        tracker = _make_tracker(clock, timeout=30.0)

        tracker.on_submit("ord-1", cmd="cmd-1")
        clock.advance(5.0)  # 5s elapsed

        cp = tracker.checkpoint()

        clock2 = _FakeClock(9999.0)
        tracker2 = _make_tracker(clock2, timeout=30.0)
        tracker2.restore(cp)

        # Only 5s elapsed; need 25 more for timeout
        clock2.advance(20.0)  # total 25s < 30s
        assert tracker2.check_timeouts() == []

        clock2.advance(6.0)  # total 31s > 30s
        assert "ord-1" in tracker2.check_timeouts()

    def test_clock_skew_does_not_affect_timeout(self):
        """NTP clock adjustment between checkpoint and restore should NOT
        affect timeout calculation (this was the original bug)."""
        clock = _FakeClock(1000.0)
        tracker = _make_tracker(clock, timeout=30.0)

        tracker.on_submit("ord-1", cmd="cmd-1")
        clock.advance(10.0)  # 10s elapsed

        cp = tracker.checkpoint()

        # Simulate restart with DIFFERENT monotonic base
        # (monotonic always resets, this is normal)
        clock2 = _FakeClock(0.0)  # Fresh monotonic after restart
        tracker2 = _make_tracker(clock2, timeout=30.0)
        tracker2.restore(cp)

        # Need 20 more seconds (30 - 10 already elapsed)
        clock2.advance(19.0)
        assert tracker2.check_timeouts() == []

        clock2.advance(2.0)  # 10 + 21 = 31 > 30
        assert "ord-1" in tracker2.check_timeouts()

    def test_backward_compat_legacy_format(self):
        """Legacy checkpoint with wall_submit_ts should still work."""
        import time

        clock = _FakeClock(1000.0)
        tracker = _make_tracker(clock, timeout=30.0)

        # Simulate legacy checkpoint format — 10s ago in wall-clock
        wall_now = time.time()
        legacy_cp = {
            "pending": {
                "ord-legacy": {"wall_submit_ts": wall_now - 10.0}
            },
            "timeout_sec": 30.0,
        }

        tracker.restore(legacy_cp)
        assert tracker.pending_count == 1

        # Legacy restore computes elapsed from wall-clock (~10s).
        # With fake clock at 1000.0, mono_submit = 1000.0 - 10.0 = 990.0.
        # Need to advance past (990.0 + 30.0) = 1020.0.
        # Current clock is 1000.0, so need >20s.
        clock.advance(21.0)  # clock=1021 > 1020
        assert "ord-legacy" in tracker.check_timeouts()

    def test_negative_elapsed_clamped_to_zero(self):
        """Corrupted checkpoint with negative elapsed should be clamped."""
        clock = _FakeClock(1000.0)
        tracker = _make_tracker(clock, timeout=30.0)

        corrupted_cp = {
            "pending": {
                "ord-corrupt": {"elapsed_sec": -100.0}  # Negative = corrupted
            },
            "timeout_sec": 30.0,
        }

        tracker.restore(corrupted_cp)
        # Should be treated as 0 elapsed (just submitted)
        assert tracker.pending_count == 1
        assert tracker.check_timeouts() == []  # Not timed out yet

    def test_multiple_orders_checkpoint_restore(self):
        """Multiple pending orders survive checkpoint/restore."""
        clock = _FakeClock(1000.0)
        tracker = _make_tracker(clock, timeout=30.0)

        tracker.on_submit("ord-1", cmd="c1")
        clock.advance(5.0)
        tracker.on_submit("ord-2", cmd="c2")
        clock.advance(5.0)
        # ord-1: 10s elapsed, ord-2: 5s elapsed

        cp = tracker.checkpoint()
        assert len(cp["pending"]) == 2
        assert abs(cp["pending"]["ord-1"]["elapsed_sec"] - 10.0) < 0.01
        assert abs(cp["pending"]["ord-2"]["elapsed_sec"] - 5.0) < 0.01

        clock2 = _FakeClock(0.0)
        tracker2 = _make_tracker(clock2, timeout=30.0)
        tracker2.restore(cp)

        # ord-1 needs 20 more, ord-2 needs 25 more
        clock2.advance(22.0)
        timed_out = tracker2.check_timeouts()
        assert "ord-1" in timed_out
        assert "ord-2" not in timed_out

    def test_empty_checkpoint_restore(self):
        """Empty checkpoint restores to empty state."""
        clock = _FakeClock(1000.0)
        tracker = _make_tracker(clock)

        cp = tracker.checkpoint()
        assert cp["pending"] == {}

        tracker.on_submit("ord-1")
        assert tracker.pending_count == 1

        tracker.restore(cp)
        assert tracker.pending_count == 0
