"""Root conftest: the project root on sys.path and the order-transition guard every test runs under.

(The event, engine and snapshot factories went with those packages, 2026-11-15.)
"""
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


@pytest.fixture(autouse=True)
def _order_transitions_stay_in_the_table():
    """Migration step 5: an order status change outside trading.domain.orders.TRANSITIONS fails the test.

    The executor only logs such a change in production (observation mode); here it must never happen.
    """
    from trading.application.execution import lifecycle
    lifecycle.TRANSITION_VIOLATIONS.clear()
    yield
    violations = list(lifecycle.TRANSITION_VIOLATIONS)
    lifecycle.TRANSITION_VIOLATIONS.clear()
    assert violations == [], f"order status transitions outside the table: {violations}"
