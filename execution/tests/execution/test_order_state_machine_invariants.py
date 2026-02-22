"""Tests for order state machine invariants."""
from __future__ import annotations

from execution.state_machine.transitions import OrderStatus, VALID_TRANSITIONS, TERMINAL_STATUSES


def test_pending_to_new():
    assert OrderStatus.NEW in VALID_TRANSITIONS.get(OrderStatus.PENDING_NEW, set())


def test_new_to_filled():
    assert OrderStatus.FILLED in VALID_TRANSITIONS.get(OrderStatus.NEW, set())


def test_filled_is_terminal():
    assert OrderStatus.FILLED in TERMINAL_STATUSES


def test_rejected_is_terminal():
    assert OrderStatus.REJECTED in TERMINAL_STATUSES


def test_terminal_no_transitions():
    for status in TERMINAL_STATUSES:
        allowed = VALID_TRANSITIONS.get(status, set())
        assert len(allowed) == 0, f"{status} should have no valid transitions"
