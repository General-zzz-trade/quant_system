"""Property-based tests for Saga state machine."""
from __future__ import annotations

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from engine.saga import SagaError, SagaManager, SagaState, TERMINAL_STATES, _TRANSITIONS

from tests_unit.properties.strategies import valid_saga_transition_sequences


@given(transitions=valid_saga_transition_sequences())
@settings(max_examples=200)
def test_valid_transitions_always_succeed(transitions):
    """Valid transition sequences always produce valid SagaState values."""
    assume(len(transitions) > 0)
    mgr = SagaManager()
    mgr.create("o1", "i1", symbol="BTCUSDT", side="buy", qty=1.0)

    for to_state in transitions:
        saga = mgr.transition("o1", to_state)
        assert isinstance(saga.state, SagaState)
        assert saga.state == to_state
        if saga.is_terminal:
            break


@given(transitions=valid_saga_transition_sequences(max_length=15))
@settings(max_examples=200)
def test_terminal_state_has_no_valid_transitions(transitions):
    """Once a saga reaches a terminal state, _TRANSITIONS has no further targets (except COMPENSATING for some)."""
    assume(len(transitions) > 0)
    mgr = SagaManager()
    mgr.create("o1", "i1", symbol="BTCUSDT", side="buy", qty=1.0)

    final_state = SagaState.PENDING
    for to_state in transitions:
        saga = mgr.transition("o1", to_state)
        final_state = saga.state
        if saga.is_terminal:
            break

    if final_state in TERMINAL_STATES:
        allowed = _TRANSITIONS.get(final_state, frozenset())
        # FILLED, COMPENSATED, FAILED have no outgoing transitions
        # But REJECTED, CANCELLED, EXPIRED can go to COMPENSATING
        if final_state in (SagaState.FILLED, SagaState.COMPENSATED, SagaState.FAILED):
            assert len(allowed) == 0


@given(
    from_state=st.sampled_from(list(SagaState)),
    to_state=st.sampled_from(list(SagaState)),
)
@settings(max_examples=500)
def test_invalid_transitions_raise_saga_error(from_state, to_state):
    """Invalid transitions always raise SagaError."""
    valid_targets = _TRANSITIONS.get(from_state, frozenset())
    assume(to_state not in valid_targets)

    # We need a path from PENDING to from_state
    path = _find_path(SagaState.PENDING, from_state)
    assume(path is not None)

    mgr = SagaManager()
    mgr.create("o1", "i1", symbol="BTCUSDT", side="buy", qty=1.0)
    for step in path:
        mgr.transition("o1", step)

    with pytest.raises(SagaError):
        mgr.transition("o1", to_state)


@given(transitions=valid_saga_transition_sequences(max_length=15))
@settings(max_examples=200)
def test_history_length_matches_transitions(transitions):
    """Saga history records every transition."""
    assume(len(transitions) > 0)
    mgr = SagaManager()
    mgr.create("o1", "i1", symbol="BTCUSDT", side="buy", qty=1.0)

    count = 0
    for to_state in transitions:
        saga = mgr.transition("o1", to_state)
        count += 1
        if saga.is_terminal:
            break

    saga = mgr.get("o1")
    assert saga is not None
    assert len(saga.history) == count


# ── Helper ────────────────────────────────────────────────

def _find_path(start: SagaState, target: SagaState) -> list[SagaState] | None:
    """BFS to find a valid path from start to target."""
    if start == target:
        return []
    from collections import deque
    queue: deque[tuple[SagaState, list[SagaState]]] = deque([(start, [])])
    visited = {start}
    while queue:
        current, path = queue.popleft()
        for next_state in _TRANSITIONS.get(current, frozenset()):
            if next_state == target:
                return path + [next_state]
            if next_state not in visited:
                visited.add(next_state)
                queue.append((next_state, path + [next_state]))
    return None
