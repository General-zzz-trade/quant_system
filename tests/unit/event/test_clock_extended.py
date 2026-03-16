"""Extended tests for event.clock — Clock modes, bar_index, monotonicity, reset."""
from __future__ import annotations

from datetime import datetime, timezone, timedelta

import pytest

from event.clock import (
    Clock,
    ClockError,
    ClockMode,
    ClockSnapshot,
    EventClock,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_T0 = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
_T1 = datetime(2024, 6, 1, 13, 0, 0, tzinfo=timezone.utc)
_T2 = datetime(2024, 6, 1, 14, 0, 0, tzinfo=timezone.utc)
_T3 = datetime(2024, 6, 1, 15, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Tests: Clock modes (REPLAY vs LIVE)
# ---------------------------------------------------------------------------


class TestClockModes:
    def test_replay_mode(self) -> None:
        clk = Clock(mode=ClockMode.REPLAY)
        assert clk.mode == ClockMode.REPLAY

    def test_live_mode(self) -> None:
        clk = Clock(mode=ClockMode.LIVE)
        assert clk.mode == ClockMode.LIVE

    def test_mode_enum_values(self) -> None:
        assert ClockMode.REPLAY.value == "replay"
        assert ClockMode.LIVE.value == "live"


# ---------------------------------------------------------------------------
# Tests: bar_index tracking
# ---------------------------------------------------------------------------


class TestBarIndex:
    def test_bar_index_auto_increment(self) -> None:
        clk = Clock(mode=ClockMode.REPLAY)
        clk.update_from_event_time(event_time=_T0)
        assert clk.bar_index == 1
        clk.update_from_event_time(event_time=_T1)
        assert clk.bar_index == 2
        clk.update_from_event_time(event_time=_T2)
        assert clk.bar_index == 3

    def test_bar_index_explicit(self) -> None:
        clk = Clock(mode=ClockMode.REPLAY)
        clk.update_from_event_time(event_time=_T0, bar_index=42)
        assert clk.bar_index == 42

    def test_bar_index_starts_at_zero(self) -> None:
        clk = Clock(mode=ClockMode.REPLAY)
        assert clk.bar_index == 0


# ---------------------------------------------------------------------------
# Tests: Time monotonicity
# ---------------------------------------------------------------------------


class TestMonotonicity:
    def test_forward_time_ok(self) -> None:
        clk = Clock(mode=ClockMode.REPLAY, strict_monotonic=True)
        clk.update_from_event_time(event_time=_T0)
        clk.update_from_event_time(event_time=_T1)
        assert clk.ts == _T1

    def test_backward_time_raises_strict(self) -> None:
        clk = Clock(mode=ClockMode.REPLAY, strict_monotonic=True)
        clk.update_from_event_time(event_time=_T1)
        with pytest.raises(ClockError, match="回退"):
            clk.update_from_event_time(event_time=_T0)

    def test_backward_time_ok_non_strict(self) -> None:
        clk = Clock(mode=ClockMode.REPLAY, strict_monotonic=False)
        clk.update_from_event_time(event_time=_T1)
        clk.update_from_event_time(event_time=_T0)
        assert clk.ts == _T0

    def test_equal_time_raises_by_default(self) -> None:
        clk = Clock(mode=ClockMode.REPLAY, strict_monotonic=True)
        clk.update_from_event_time(event_time=_T0)
        with pytest.raises(ClockError, match="重复"):
            clk.update_from_event_time(event_time=_T0)

    def test_equal_time_ok_when_allowed(self) -> None:
        clk = Clock(mode=ClockMode.REPLAY, strict_monotonic=True, allow_equal_timestamps=True)
        clk.update_from_event_time(event_time=_T0)
        clk.update_from_event_time(event_time=_T0)
        assert clk.ts == _T0

    def test_naive_datetime_raises(self) -> None:
        clk = Clock(mode=ClockMode.REPLAY)
        naive = datetime(2024, 6, 1, 12, 0, 0)
        with pytest.raises(ClockError, match="naive"):
            clk.update_from_event_time(event_time=naive)


# ---------------------------------------------------------------------------
# Tests: reset() behavior
# ---------------------------------------------------------------------------


class TestClockReset:
    def test_reset_replay_mode(self) -> None:
        clk = Clock(mode=ClockMode.REPLAY)
        clk.update_from_event_time(event_time=_T1)
        clk.reset(ts=_T0, bar_index=0)
        assert clk.ts == _T0
        assert clk.bar_index == 0

    def test_reset_live_mode_without_force_raises(self) -> None:
        clk = Clock(mode=ClockMode.LIVE)
        clk.update_from_event_time(event_time=_T0)
        with pytest.raises(ClockError, match="force"):
            clk.reset(ts=_T0)

    def test_reset_live_mode_with_force(self) -> None:
        clk = Clock(mode=ClockMode.LIVE)
        clk.update_from_event_time(event_time=_T1)
        clk.reset(ts=_T0, bar_index=5, force=True)
        assert clk.ts == _T0
        assert clk.bar_index == 5

    def test_reset_allows_forward_update_after(self) -> None:
        clk = Clock(mode=ClockMode.REPLAY)
        clk.update_from_event_time(event_time=_T2)
        clk.reset(ts=_T0)
        # After reset, we should be able to update forward from _T0
        clk.update_from_event_time(event_time=_T1)
        assert clk.ts == _T1


# ---------------------------------------------------------------------------
# Tests: ts access before initialization
# ---------------------------------------------------------------------------


class TestClockUninitialized:
    def test_ts_before_update_raises(self) -> None:
        clk = Clock(mode=ClockMode.REPLAY)
        with pytest.raises(ClockError, match="初始化"):
            _ = clk.ts


# ---------------------------------------------------------------------------
# Tests: Snapshot
# ---------------------------------------------------------------------------


class TestClockSnapshot:
    def test_snapshot_replay(self) -> None:
        clk = Clock(mode=ClockMode.REPLAY)
        clk.update_from_event_time(event_time=_T0)
        snap = clk.snapshot()
        assert isinstance(snap, ClockSnapshot)
        assert snap.mode == ClockMode.REPLAY
        assert snap.ts == _T0
        assert snap.bar_index == 1
        assert snap.drift_ms is None  # No drift in replay

    def test_snapshot_live_with_wall_time(self) -> None:
        clk = Clock(mode=ClockMode.LIVE)
        wall = _T0 + timedelta(milliseconds=150)
        clk.update_from_event_time(event_time=_T0, wall_time=wall)
        snap = clk.snapshot()
        assert snap.mode == ClockMode.LIVE
        assert snap.drift_ms == 150

    def test_snapshot_frozen(self) -> None:
        snap = ClockSnapshot(
            mode=ClockMode.REPLAY,
            ts=_T0,
            bar_index=10,
            last_event_time=_T0,
            last_wall_time=None,
            drift_ms=None,
        )
        assert snap.bar_index == 10


# ---------------------------------------------------------------------------
# Tests: is_after / is_before
# ---------------------------------------------------------------------------


class TestClockComparisons:
    def test_is_after(self) -> None:
        clk = Clock(mode=ClockMode.REPLAY)
        clk.update_from_event_time(event_time=_T1)
        assert clk.is_after(_T0) is True
        assert clk.is_after(_T2) is False

    def test_is_before(self) -> None:
        clk = Clock(mode=ClockMode.REPLAY)
        clk.update_from_event_time(event_time=_T1)
        assert clk.is_before(_T2) is True
        assert clk.is_before(_T0) is False


# ---------------------------------------------------------------------------
# Tests: EventClock (simple dataclass)
# ---------------------------------------------------------------------------


class TestEventClock:
    def test_default_values(self) -> None:
        ec = EventClock()
        assert ec.current_time is None
        assert ec.bar_index == 0

    def test_update_from_event_time(self) -> None:
        ec = EventClock()
        ec.update_from_event_time(event_time=_T0, bar_index=5)
        assert ec.current_time == _T0
        assert ec.bar_index == 5

    def test_non_utc_timezone(self) -> None:
        """Clock converts non-UTC to UTC internally."""
        tz_est = timezone(timedelta(hours=-5))
        t_est = datetime(2024, 6, 1, 7, 0, 0, tzinfo=tz_est)  # 7am EST = 12pm UTC
        clk = Clock(mode=ClockMode.REPLAY)
        clk.update_from_event_time(event_time=t_est)
        assert clk.ts == _T0
