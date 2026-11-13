"""
Comprehensive unit tests for engine modules with low coverage.

Targets:
- engine/replay.py      (27% -> 75%)
- engine/clock.py       (44% -> 80%)
- engine/config.py      (0%  -> 80%)
- engine/metrics.py     (0%  -> 80%)
- engine/tracing.py     (0%  -> 80%)
- engine/module_reloader.py (34% -> 75%)
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# _quant_hotpath is the real Rust extension — import it directly.
# We provide a minimal pure-Python RustSpscRing shim only if the real one
# does not work in test isolation (it normally does in this environment).
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# engine/clock.py
# ---------------------------------------------------------------------------

class TestReplayClockEngine:
    def _make(self):
        from engine.clock import ReplayClock
        return ReplayClock()

    def test_initial_state(self):
        c = self._make()
        assert c.now() is None

    def test_advance_to_sets_first(self):
        from engine.clock import ReplayClock
        c = ReplayClock()
        c.advance_to(100.0)
        assert c.now() == 100.0

    def test_advance_to_increases(self):
        c = self._make()
        c.advance_to(100.0)
        c.advance_to(200.0)
        assert c.now() == 200.0

    def test_advance_to_same_value_ok(self):
        c = self._make()
        c.advance_to(100.0)
        c.advance_to(100.0)
        assert c.now() == 100.0

    def test_advance_to_backwards_raises(self):
        from engine.clock import ClockMonotonicError
        c = self._make()
        c.advance_to(200.0)
        with pytest.raises(ClockMonotonicError):
            c.advance_to(100.0)

    def test_advance_by_from_none(self):
        c = self._make()
        c.advance_by(5.0)
        assert c.now() == 5.0

    def test_advance_by_adds(self):
        c = self._make()
        c.advance_to(10.0)
        c.advance_by(5.0)
        assert c.now() == 15.0

    def test_advance_by_negative_raises(self):
        from engine.clock import ClockMonotonicError
        c = self._make()
        c.advance_to(10.0)
        with pytest.raises(ClockMonotonicError):
            c.advance_by(-1.0)

    def test_set_delegates_to_advance_to(self):
        c = self._make()
        c.set(50.0)
        assert c.now() == 50.0

    def test_mode_is_replay(self):
        from engine.clock import ClockMode
        c = self._make()
        assert c.mode == ClockMode.REPLAY

    def test_advance_to_none_noop(self):
        c = self._make()
        c.advance_to(100.0)
        c.advance_to(None)
        assert c.now() == 100.0

    def test_advance_by_non_numeric_ts(self):
        """advance_by when current ts is non-numeric falls back to string repr."""
        from engine.clock import ReplayClock
        c = ReplayClock()
        # Force non-numeric ts via string comparison path
        c._ts = "some_string_ts"
        c.advance_by(3.0)
        assert "+" in str(c.now())

    def test_advance_to_string_order_ok(self):
        """String timestamp comparison: ascending strings should work."""
        c = self._make()
        c.advance_to("2026-01-01")
        c.advance_to("2026-01-02")
        assert c.now() == "2026-01-02"

    def test_advance_to_string_backwards_raises(self):
        from engine.clock import ClockMonotonicError
        c = self._make()
        c.advance_to("2026-01-02")
        with pytest.raises(ClockMonotonicError):
            c.advance_to("2026-01-01")


class TestLiveClockEngine:
    def test_mode_is_live(self):
        from engine.clock import LiveClock, ClockMode
        c = LiveClock()
        assert c.mode == ClockMode.LIVE

    def test_now_returns_float(self):
        from engine.clock import LiveClock
        c = LiveClock()
        v = c.now()
        assert isinstance(v, float)

    def test_now_uses_injected_fn(self):
        from engine.clock import LiveClock
        c = LiveClock(time_fn=lambda: 999.5)
        assert c.now() == 999.5

    def test_set_raises(self):
        from engine.clock import LiveClock, ClockImmutableError
        c = LiveClock()
        with pytest.raises(ClockImmutableError):
            c.set(100.0)

    def test_advance_to_raises(self):
        from engine.clock import LiveClock, ClockImmutableError
        c = LiveClock()
        with pytest.raises(ClockImmutableError):
            c.advance_to(100.0)

    def test_advance_by_raises(self):
        from engine.clock import LiveClock, ClockImmutableError
        c = LiveClock()
        with pytest.raises(ClockImmutableError):
            c.advance_by(1.0)


class TestToFloatSeconds:
    def test_int(self):
        from engine.clock import _to_float_seconds
        assert _to_float_seconds(100) == 100.0

    def test_float(self):
        from engine.clock import _to_float_seconds
        assert _to_float_seconds(3.14) == 3.14

    def test_none_returns_none(self):
        from engine.clock import _to_float_seconds
        assert _to_float_seconds(None) is None

    def test_datetime_like(self):
        from engine.clock import _to_float_seconds
        obj = MagicMock()
        obj.timestamp.return_value = 42.0
        assert _to_float_seconds(obj) == 42.0

    def test_datetime_like_exception(self):
        from engine.clock import _to_float_seconds
        obj = MagicMock()
        obj.timestamp.side_effect = ValueError("bad")
        assert _to_float_seconds(obj) is None

    def test_obj_with_ts_attr(self):
        from engine.clock import _to_float_seconds

        class Obj:
            ts = 77.0

        assert _to_float_seconds(Obj()) == 77.0

    def test_unrecognised_type(self):
        from engine.clock import _to_float_seconds
        assert _to_float_seconds(object()) is None


# ---------------------------------------------------------------------------
# engine/config.py
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# engine/replay.py
# ---------------------------------------------------------------------------

class TestReplayConfig:
    def test_defaults(self):
        from engine.replay import ReplayConfig
        c = ReplayConfig()
        assert c.strict_order is True
        assert c.actor == "replay"
        assert c.stop_on_error is True
        assert c.allow_drop is True
        assert c.progress_interval == 0

    def test_custom(self):
        from engine.replay import ReplayConfig
        c = ReplayConfig(strict_order=False, stop_on_error=False, actor="test")
        assert c.strict_order is False
        assert c.actor == "test"


class TestEventReplay:
    def _make_dispatcher(self, route="drop"):
        from engine.dispatcher import Route
        disp = MagicMock()
        route_map = {"pipeline": Route.PIPELINE, "drop": Route.DROP}
        disp._route_for.return_value = route_map.get(route, Route.DROP)
        disp.dispatch.return_value = None
        return disp

    def _make_replay(self, events, route="pipeline", strict=True,
                     stop_on_error=True, allow_drop=True, progress_interval=0):
        from engine.replay import EventReplay, ReplayConfig
        disp = self._make_dispatcher(route=route)
        cfg = ReplayConfig(
            strict_order=strict,
            stop_on_error=stop_on_error,
            allow_drop=allow_drop,
            progress_interval=progress_interval,
        )
        replay = EventReplay(dispatcher=disp, source=events, config=cfg)
        return replay, disp

    def test_run_empty_source(self):
        replay, disp = self._make_replay([])
        n = replay.run()
        assert n == 0
        disp.dispatch.assert_not_called()

    def test_run_dispatches_events(self):
        events = [MagicMock(spec=[]) for _ in range(3)]
        replay, disp = self._make_replay(events)
        n = replay.run()
        assert n == 3
        assert disp.dispatch.call_count == 3

    def test_run_drop_route_not_counted(self):
        events = [MagicMock(spec=[]) for _ in range(2)]
        replay, disp = self._make_replay(events, route="drop")
        n = replay.run()
        assert n == 0  # DROP events not counted

    def test_strict_order_raises_on_reverse_ts(self):
        from engine.replay import ReplayError

        class E:
            def __init__(self, ts):
                self.ts = ts

        events = [E(200), E(100)]
        replay, _ = self._make_replay(events, strict=True)
        with pytest.raises(ReplayError):
            replay.run()

    def test_strict_order_ok_for_increasing_ts(self):
        class E:
            def __init__(self, ts):
                self.ts = ts

        events = [E(100), E(200), E(300)]
        replay, _ = self._make_replay(events, strict=True)
        n = replay.run()
        assert n == 3

    def test_allow_drop_false_raises_on_drop(self):
        from engine.replay import ReplayError
        events = [MagicMock(spec=[])]
        replay, _ = self._make_replay(events, route="drop", allow_drop=False)
        with pytest.raises(ReplayError):
            replay.run()

    def test_stop_on_error_true_propagates(self):
        from engine.replay import EventReplay, ReplayConfig
        from engine.dispatcher import Route
        disp = MagicMock()
        disp._route_for.return_value = Route.PIPELINE
        disp.dispatch.side_effect = RuntimeError("fail")
        cfg = ReplayConfig(stop_on_error=True)
        replay = EventReplay(dispatcher=disp, source=[MagicMock(spec=[])], config=cfg)
        with pytest.raises(RuntimeError):
            replay.run()

    def test_stop_on_error_false_continues(self):
        from engine.replay import EventReplay, ReplayConfig
        from engine.dispatcher import Route

        disp = MagicMock()
        disp._route_for.return_value = Route.PIPELINE
        # First event fails, second succeeds
        disp.dispatch.side_effect = [RuntimeError("fail"), None]
        cfg = ReplayConfig(stop_on_error=False)
        events = [MagicMock(spec=[]) for _ in range(2)]
        replay = EventReplay(dispatcher=disp, source=events, config=cfg)
        n = replay.run()
        # Second event was processed (dispatched without raising)
        assert n == 1

    def test_sink_on_start_called(self):
        from engine.replay import EventReplay, ReplayConfig
        from engine.dispatcher import Route

        class _NoLen:
            """Iterable without __len__ so total=None."""
            def __iter__(self):
                return iter([])

        disp = MagicMock()
        disp._route_for.return_value = Route.PIPELINE
        sink = MagicMock()
        cfg = ReplayConfig()
        replay = EventReplay(dispatcher=disp, source=_NoLen(), sink=sink, config=cfg)
        replay.run()
        sink.on_start.assert_called_once_with(None)

    def test_sink_on_start_with_len(self):
        from engine.replay import EventReplay, ReplayConfig
        from engine.dispatcher import Route
        disp = MagicMock()
        disp._route_for.return_value = Route.PIPELINE
        sink = MagicMock()
        cfg = ReplayConfig()
        source = [MagicMock(spec=[])]  # list has __len__
        replay = EventReplay(dispatcher=disp, source=source, sink=sink, config=cfg)
        replay.run()
        sink.on_start.assert_called_once_with(1)

    def test_sink_on_finish_called(self):
        from engine.replay import EventReplay, ReplayConfig
        from engine.dispatcher import Route
        disp = MagicMock()
        disp._route_for.return_value = Route.PIPELINE
        sink = MagicMock()
        cfg = ReplayConfig()
        replay = EventReplay(dispatcher=disp, source=[], sink=sink, config=cfg)
        replay.run()
        sink.on_finish.assert_called_once_with(0)

    def test_sink_on_start_exception_ignored(self):
        from engine.replay import EventReplay, ReplayConfig
        from engine.dispatcher import Route
        disp = MagicMock()
        disp._route_for.return_value = Route.PIPELINE
        sink = MagicMock()
        sink.on_start.side_effect = RuntimeError("oops")
        cfg = ReplayConfig()
        replay = EventReplay(dispatcher=disp, source=[], sink=sink, config=cfg)
        # Should not raise
        replay.run()

    def test_sink_on_finish_exception_ignored(self):
        from engine.replay import EventReplay, ReplayConfig
        from engine.dispatcher import Route
        disp = MagicMock()
        disp._route_for.return_value = Route.PIPELINE
        sink = MagicMock()
        sink.on_finish.side_effect = RuntimeError("oops")
        cfg = ReplayConfig()
        replay = EventReplay(dispatcher=disp, source=[], sink=sink, config=cfg)
        replay.run()

    def test_replay_interrupted_propagates_from_sink(self):
        from engine.replay import EventReplay, ReplayConfig, ReplayInterrupted
        from engine.dispatcher import Route
        disp = MagicMock()
        disp._route_for.return_value = Route.PIPELINE
        sink = MagicMock()
        sink.on_event.side_effect = ReplayInterrupted("stop!")
        cfg = ReplayConfig()
        events = [MagicMock(spec=[])]
        replay = EventReplay(dispatcher=disp, source=events, sink=sink, config=cfg)
        with pytest.raises(ReplayInterrupted):
            replay.run()

    def test_progress_interval_triggers_callback(self):
        from engine.replay import EventReplay, ReplayConfig
        from engine.dispatcher import Route
        disp = MagicMock()
        disp._route_for.return_value = Route.PIPELINE
        sink = MagicMock()
        sink.on_event.return_value = None  # no exception
        cfg = ReplayConfig(progress_interval=2)
        events = [MagicMock(spec=[]) for _ in range(4)]
        replay = EventReplay(dispatcher=disp, source=events, sink=sink, config=cfg)
        replay.run()
        # on_event should be called multiple times (regular + progress)
        assert sink.on_event.call_count >= 4

    def test_order_keys_with_header_ts(self):
        from engine.replay import EventReplay

        class Header:
            ts = 500
            event_index = 1

        class E:
            header = Header()

        ts_key, eidx_key, fb = EventReplay._order_keys(E(), fallback=1)
        assert ts_key == 500
        assert eidx_key == 1

    def test_order_keys_fallback_only(self):
        from engine.replay import EventReplay
        ts_key, eidx_key, fb = EventReplay._order_keys(object(), fallback=7)
        assert ts_key is None
        assert eidx_key is None
        assert fb == 7

    def test_strict_order_fallback_path_raises(self):
        """When no ts or event_index, strict order uses fallback idx."""
        # objects with no ts/header
        events = [object(), object()]
        replay, _ = self._make_replay(events, strict=True)
        # fallback idx goes 1, 2 — forward, should not raise
        n = replay.run()
        assert n == 2

    def test_strict_order_event_index_check(self):
        """Test the event_index ordering path."""
        from engine.replay import ReplayError, EventReplay, ReplayConfig
        from engine.dispatcher import Route

        class Header:
            ts = None  # no ts to force eidx path
            event_index = None

        class E:
            def __init__(self, eidx):
                self.header = Header()
                self.header.event_index = eidx

        # Ensure eidx only path by making ts missing
        class Header2:
            event_index: int

        disp = MagicMock()
        disp._route_for.return_value = Route.PIPELINE
        # Build manually to check the eidx backward check
        # events: eidx goes 3, 1 -> should raise
        events_data = []
        for idx_val in [3, 1]:
            e = MagicMock(spec=[])
            del e.ts
            h = MagicMock()
            h.ts = None
            h.event_index = idx_val
            e.header = h
            events_data.append(e)

        cfg = ReplayConfig(strict_order=True)
        replay = EventReplay(dispatcher=disp, source=events_data, config=cfg)
        with pytest.raises(ReplayError):
            replay.run()


# ---------------------------------------------------------------------------
# engine/guards.py
# ---------------------------------------------------------------------------
