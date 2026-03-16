"""Tests for event.lifecycle — LifecycleState transitions, terminal states, tracking."""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from event.lifecycle import (
    EventLifecycle,
    EventLifecycleError,
    LifecycleState,
    _ALLOWED_TRANSITIONS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _FakeHeader:
    event_id: str


@dataclass(frozen=True)
class _FakeEvent:
    header: _FakeHeader


def _evt(eid: str = "evt-001") -> _FakeEvent:
    return _FakeEvent(header=_FakeHeader(event_id=eid))


# ---------------------------------------------------------------------------
# Tests: Valid transitions
# ---------------------------------------------------------------------------


class TestValidTransitions:
    def test_created_to_enqueued(self) -> None:
        lc = EventLifecycle()
        e = _evt()
        lc.ensure_created(e)
        result = lc.transition(e, LifecycleState.ENQUEUED)
        assert result == LifecycleState.ENQUEUED

    def test_full_happy_path(self) -> None:
        """CREATED -> ENQUEUED -> DISPATCH_START -> HANDLED -> DISPATCHED"""
        lc = EventLifecycle()
        e = _evt()
        lc.ensure_created(e)
        lc.transition(e, LifecycleState.ENQUEUED)
        lc.transition(e, LifecycleState.DISPATCH_START)
        lc.transition(e, LifecycleState.HANDLED)
        result = lc.transition(e, LifecycleState.DISPATCHED)
        assert result == LifecycleState.DISPATCHED

    def test_dispatch_start_to_retry(self) -> None:
        lc = EventLifecycle()
        e = _evt()
        lc.ensure_created(e)
        lc.transition(e, LifecycleState.ENQUEUED)
        lc.transition(e, LifecycleState.DISPATCH_START)
        result = lc.transition(e, LifecycleState.RETRY)
        assert result == LifecycleState.RETRY

    def test_retry_to_enqueued(self) -> None:
        lc = EventLifecycle()
        e = _evt()
        lc.ensure_created(e)
        lc.transition(e, LifecycleState.ENQUEUED)
        lc.transition(e, LifecycleState.DISPATCH_START)
        lc.transition(e, LifecycleState.RETRY)
        result = lc.transition(e, LifecycleState.ENQUEUED)
        assert result == LifecycleState.ENQUEUED

    def test_retry_to_failed(self) -> None:
        lc = EventLifecycle()
        e = _evt()
        lc.ensure_created(e)
        lc.transition(e, LifecycleState.ENQUEUED)
        lc.transition(e, LifecycleState.DISPATCH_START)
        lc.transition(e, LifecycleState.RETRY)
        result = lc.transition(e, LifecycleState.FAILED)
        assert result == LifecycleState.FAILED

    def test_retry_to_dropped(self) -> None:
        lc = EventLifecycle()
        e = _evt()
        lc.ensure_created(e)
        lc.transition(e, LifecycleState.ENQUEUED)
        lc.transition(e, LifecycleState.DISPATCH_START)
        lc.transition(e, LifecycleState.RETRY)
        result = lc.transition(e, LifecycleState.DROPPED)
        assert result == LifecycleState.DROPPED

    def test_dispatch_start_to_dropped(self) -> None:
        lc = EventLifecycle()
        e = _evt()
        lc.ensure_created(e)
        lc.transition(e, LifecycleState.ENQUEUED)
        lc.transition(e, LifecycleState.DISPATCH_START)
        result = lc.transition(e, LifecycleState.DROPPED)
        assert result == LifecycleState.DROPPED

    def test_dispatch_start_to_failed(self) -> None:
        lc = EventLifecycle()
        e = _evt()
        lc.ensure_created(e)
        lc.transition(e, LifecycleState.ENQUEUED)
        lc.transition(e, LifecycleState.DISPATCH_START)
        result = lc.transition(e, LifecycleState.FAILED)
        assert result == LifecycleState.FAILED


# ---------------------------------------------------------------------------
# Tests: Invalid transitions
# ---------------------------------------------------------------------------


class TestInvalidTransitions:
    def test_created_to_dispatched_raises(self) -> None:
        lc = EventLifecycle()
        e = _evt()
        lc.ensure_created(e)
        with pytest.raises(EventLifecycleError):
            lc.transition(e, LifecycleState.DISPATCHED)

    def test_enqueued_to_handled_raises(self) -> None:
        lc = EventLifecycle()
        e = _evt()
        lc.ensure_created(e)
        lc.transition(e, LifecycleState.ENQUEUED)
        with pytest.raises(EventLifecycleError):
            lc.transition(e, LifecycleState.HANDLED)

    def test_created_to_failed_raises(self) -> None:
        lc = EventLifecycle()
        e = _evt()
        lc.ensure_created(e)
        with pytest.raises(EventLifecycleError):
            lc.transition(e, LifecycleState.FAILED)

    def test_handled_to_retry_raises(self) -> None:
        lc = EventLifecycle()
        e = _evt()
        lc.ensure_created(e)
        lc.transition(e, LifecycleState.ENQUEUED)
        lc.transition(e, LifecycleState.DISPATCH_START)
        lc.transition(e, LifecycleState.HANDLED)
        with pytest.raises(EventLifecycleError):
            lc.transition(e, LifecycleState.RETRY)


# ---------------------------------------------------------------------------
# Tests: Terminal states
# ---------------------------------------------------------------------------


class TestTerminalStates:
    def test_dispatched_is_terminal(self) -> None:
        assert _ALLOWED_TRANSITIONS[LifecycleState.DISPATCHED] == set()

    def test_failed_is_terminal(self) -> None:
        assert _ALLOWED_TRANSITIONS[LifecycleState.FAILED] == set()

    def test_dropped_is_terminal(self) -> None:
        assert _ALLOWED_TRANSITIONS[LifecycleState.DROPPED] == set()

    def test_no_transition_from_dispatched(self) -> None:
        lc = EventLifecycle()
        e = _evt()
        lc.ensure_created(e)
        lc.transition(e, LifecycleState.ENQUEUED)
        lc.transition(e, LifecycleState.DISPATCH_START)
        lc.transition(e, LifecycleState.HANDLED)
        lc.transition(e, LifecycleState.DISPATCHED)
        with pytest.raises(EventLifecycleError):
            lc.transition(e, LifecycleState.ENQUEUED)

    def test_no_transition_from_failed(self) -> None:
        lc = EventLifecycle()
        e = _evt()
        lc.ensure_created(e)
        lc.transition(e, LifecycleState.ENQUEUED)
        lc.transition(e, LifecycleState.DISPATCH_START)
        lc.transition(e, LifecycleState.FAILED)
        with pytest.raises(EventLifecycleError):
            lc.transition(e, LifecycleState.RETRY)


# ---------------------------------------------------------------------------
# Tests: State tracking
# ---------------------------------------------------------------------------


class TestStateTracking:
    def test_state_of_unknown_event_returns_none(self) -> None:
        lc = EventLifecycle()
        e = _evt("unknown-event")
        assert lc.state_of(e) is None

    def test_state_of_after_ensure_created(self) -> None:
        lc = EventLifecycle()
        e = _evt()
        lc.ensure_created(e)
        assert lc.state_of(e) == LifecycleState.CREATED

    def test_ensure_created_is_idempotent(self) -> None:
        lc = EventLifecycle()
        e = _evt()
        lc.ensure_created(e)
        lc.transition(e, LifecycleState.ENQUEUED)
        # Calling ensure_created again should NOT reset to CREATED
        lc.ensure_created(e)
        assert lc.state_of(e) == LifecycleState.ENQUEUED

    def test_reset_clears_state(self) -> None:
        lc = EventLifecycle()
        e = _evt()
        lc.ensure_created(e)
        lc.transition(e, LifecycleState.ENQUEUED)
        lc.reset(e)
        assert lc.state_of(e) is None

    def test_snapshot_returns_all_states(self) -> None:
        lc = EventLifecycle()
        e1 = _evt("e1")
        e2 = _evt("e2")
        lc.ensure_created(e1)
        lc.ensure_created(e2)
        lc.transition(e1, LifecycleState.ENQUEUED)

        snap = lc.snapshot()
        assert snap["e1"] == "enqueued"
        assert snap["e2"] == "created"

    def test_multiple_events_independent(self) -> None:
        lc = EventLifecycle()
        e1 = _evt("e1")
        e2 = _evt("e2")
        lc.ensure_created(e1)
        lc.ensure_created(e2)
        lc.transition(e1, LifecycleState.ENQUEUED)
        # e2 should still be CREATED
        assert lc.state_of(e2) == LifecycleState.CREATED

    def test_lifecycle_error_str(self) -> None:
        err = EventLifecycleError(message="bad transition", event_id="evt-x")
        s = str(err)
        assert "bad transition" in s
        assert "evt-x" in s


# ---------------------------------------------------------------------------
# Tests: Fallback event_key (no header)
# ---------------------------------------------------------------------------


class TestEventKeyFallback:
    def test_event_without_header_uses_id_fallback(self) -> None:
        """When event has no header.event_id, lifecycle uses id() as key."""
        lc = EventLifecycle()
        obj = object()
        lc.ensure_created(obj)
        assert lc.state_of(obj) == LifecycleState.CREATED
