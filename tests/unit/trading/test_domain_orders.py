"""Order status table in trading.domain.orders: every pair is either allowed or refused, never ignored."""
from __future__ import annotations

import itertools

import pytest

from trading.domain.orders import SETTLED, TRANSITIONS, UNSETTLED, IllegalTransition, OrderStatus, check_transition

ALL = list(OrderStatus)


@pytest.mark.parametrize("source,target", list(itertools.product([None, *ALL], ALL)))
def test_every_status_pair_is_decided_by_the_table(source, target):
    allowed = source == target or target in TRANSITIONS.get(source, frozenset())
    if allowed:
        check_transition(None if source is None else source.value, target.value)
    else:
        with pytest.raises(IllegalTransition):
            check_transition(None if source is None else source.value, target.value)


def test_settled_orders_never_change_and_unsettled_orders_can_always_settle():
    for settled, other in itertools.product(SETTLED, ALL):
        if other != settled:
            with pytest.raises(IllegalTransition):
                check_transition(settled.value, other.value)
    for unsettled, settled in itertools.product(UNSETTLED, SETTLED):
        check_transition(unsettled.value, settled.value)


def test_new_rows_start_reserved_or_adopted_and_unknown_is_sticky():
    with pytest.raises(IllegalTransition):
        check_transition(None, "unknown")
    with pytest.raises(IllegalTransition):
        check_transition("unknown", "submitted")
    with pytest.raises(IllegalTransition):
        check_transition("submitted", "reserved")


@pytest.mark.parametrize("pair", [("bogus", "filled"), ("reserved", "bogus"), (None, "blocked")])
def test_unknown_status_strings_are_refused(pair):
    with pytest.raises(IllegalTransition):
        check_transition(*pair)


def test_status_values_are_the_stored_strings():
    assert {s.value for s in UNSETTLED} == {"reserved", "submitted", "unknown", "rejected_unconfirmed"}
    assert {s.value for s in SETTLED} == {"filled", "canceled", "mmp_canceled", "rejected", "expired", "not_found"}
