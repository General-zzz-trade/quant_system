"""Tests for engine clock."""
from __future__ import annotations

from engine.clock import Clock


def test_clock_protocol_exists():
    """Clock is a Protocol — verify it's importable."""
    assert Clock is not None


class TestReplayClockMonotonicity:
    """Tests for non-monotonic feed() warning in core.clock.ReplayClock."""

    def test_backward_time_raises_warning(self, caplog):
        import logging
        from core.clock import ReplayClock
        t0 = __import__("datetime").datetime(2026, 1, 1, 12, 0, 0,
                                              tzinfo=__import__("datetime").timezone.utc)
        t1 = __import__("datetime").datetime(2026, 1, 1, 11, 0, 0,
                                              tzinfo=__import__("datetime").timezone.utc)
        clock = ReplayClock()
        clock.feed(t0)
        with caplog.at_level(logging.WARNING, logger="core.clock"):
            clock.feed(t1)  # backward
        warnings = [r for r in caplog.records
                    if r.levelno >= logging.WARNING and "monoton" in r.message.lower()]
        assert len(warnings) >= 1

    def test_forward_time_no_warning(self, caplog):
        import logging
        from core.clock import ReplayClock
        t0 = __import__("datetime").datetime(2026, 1, 1, 12, 0, 0,
                                              tzinfo=__import__("datetime").timezone.utc)
        t1 = __import__("datetime").datetime(2026, 1, 1, 13, 0, 0,
                                              tzinfo=__import__("datetime").timezone.utc)
        clock = ReplayClock()
        clock.feed(t0)
        with caplog.at_level(logging.WARNING, logger="core.clock"):
            clock.feed(t1)
        warnings = [r for r in caplog.records
                    if r.levelno >= logging.WARNING and "monoton" in r.message.lower()]
        assert len(warnings) == 0
