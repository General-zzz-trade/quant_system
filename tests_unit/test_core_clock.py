"""Tests for core.clock — SystemClock, SimulatedClock, ReplayClock."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from core.clock import ReplayClock, SimulatedClock, SystemClock


class TestSystemClock:
    def test_now_returns_utc(self) -> None:
        c = SystemClock()
        now = c.now()
        assert now.tzinfo is not None
        assert now.tzinfo == timezone.utc

    def test_monotonic_increases(self) -> None:
        c = SystemClock()
        t1 = c.monotonic()
        t2 = c.monotonic()
        assert t2 >= t1


class TestSimulatedClock:
    def test_initial_time(self) -> None:
        c = SimulatedClock()
        assert c.now() == datetime(2024, 1, 1, tzinfo=timezone.utc)
        assert c.monotonic() == 0.0

    def test_advance(self) -> None:
        c = SimulatedClock()
        c.advance(timedelta(hours=1))
        assert c.now() == datetime(2024, 1, 1, 1, 0, tzinfo=timezone.utc)
        assert c.monotonic() == 3600.0

    def test_set_absolute(self) -> None:
        c = SimulatedClock()
        target = datetime(2025, 6, 15, 12, 0, tzinfo=timezone.utc)
        c.set(target)
        assert c.now() == target
        assert c.monotonic() > 0

    def test_set_naive_becomes_utc(self) -> None:
        c = SimulatedClock()
        c.set(datetime(2025, 1, 1))
        assert c.now().tzinfo == timezone.utc

    def test_sleep_advances_time(self) -> None:
        c = SimulatedClock()
        c.sleep(10.0)
        assert c.monotonic() == 10.0
        assert c.now() == datetime(2024, 1, 1, 0, 0, 10, tzinfo=timezone.utc)

    def test_multiple_advances_accumulate(self) -> None:
        c = SimulatedClock()
        c.advance(timedelta(seconds=5))
        c.advance(timedelta(seconds=3))
        assert c.monotonic() == 8.0


class TestReplayClock:
    def test_initial_state(self) -> None:
        c = ReplayClock()
        assert c.now() == datetime(2024, 1, 1, tzinfo=timezone.utc)

    def test_feed_advances_time(self) -> None:
        c = ReplayClock()
        t1 = datetime(2024, 3, 15, 10, 0, tzinfo=timezone.utc)
        c.feed(t1)
        assert c.now() == t1
        assert c.monotonic() > 0

    def test_feed_earlier_time_no_regression(self) -> None:
        c = ReplayClock()
        t1 = datetime(2024, 6, 1, tzinfo=timezone.utc)
        t2 = datetime(2024, 1, 1, tzinfo=timezone.utc)
        c.feed(t1)
        mono_after_t1 = c.monotonic()
        c.feed(t2)  # earlier — should not regress
        assert c.now() == t1  # stays at latest
        assert c.monotonic() == mono_after_t1

    def test_sleep_is_noop(self) -> None:
        c = ReplayClock()
        mono_before = c.monotonic()
        c.sleep(100.0)
        assert c.monotonic() == mono_before

    def test_feed_naive_becomes_utc(self) -> None:
        c = ReplayClock()
        c.feed(datetime(2025, 1, 1))
        assert c.now().tzinfo == timezone.utc
