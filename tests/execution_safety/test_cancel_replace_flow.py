"""Cancel-replace safety tests."""
from __future__ import annotations

from decimal import Decimal

import pytest

from execution.state_machine.machine import OrderStateMachine, OrderStateMachineError
from execution.state_machine.transitions import OrderStatus


def test_replacement_order_survives_after_original_cancelled() -> None:
    """A replacement order must remain valid after the original order is cancelled."""
    sm = OrderStateMachine()

    sm.register(
        order_id="old-1",
        symbol="BTCUSDT",
        side="buy",
        order_type="LIMIT",
        qty=Decimal("1.0"),
        price=Decimal("50000"),
    )
    sm.transition(order_id="old-1", new_status=OrderStatus.NEW, ts_ms=1_000)
    sm.transition(order_id="old-1", new_status=OrderStatus.PENDING_CANCEL, ts_ms=2_000)
    old_state = sm.transition(
        order_id="old-1",
        new_status=OrderStatus.CANCELED,
        ts_ms=3_000,
        reason="cancel_replace",
    )

    sm.register(
        order_id="new-1",
        symbol="BTCUSDT",
        side="buy",
        order_type="LIMIT",
        qty=Decimal("1.0"),
        price=Decimal("49990"),
    )
    new_state = sm.transition(order_id="new-1", new_status=OrderStatus.NEW, ts_ms=4_000)

    assert old_state.status == OrderStatus.CANCELED
    assert new_state.status == OrderStatus.NEW
    assert sm.get("new-1").price == Decimal("49990")
    assert sm.active_count() == 1


def test_late_fill_on_cancelled_original_must_fail() -> None:
    """Once the original order is cancelled, a later fill report must not revive it."""
    sm = OrderStateMachine()

    sm.register(
        order_id="old-2",
        symbol="BTCUSDT",
        side="buy",
        order_type="LIMIT",
        qty=Decimal("1.0"),
        price=Decimal("50000"),
    )
    sm.transition(order_id="old-2", new_status=OrderStatus.NEW, ts_ms=1_000)
    sm.transition(order_id="old-2", new_status=OrderStatus.PENDING_CANCEL, ts_ms=2_000)
    sm.transition(order_id="old-2", new_status=OrderStatus.CANCELED, ts_ms=3_000)

    sm.register(
        order_id="new-2",
        symbol="BTCUSDT",
        side="buy",
        order_type="LIMIT",
        qty=Decimal("1.0"),
        price=Decimal("49980"),
    )
    sm.transition(order_id="new-2", new_status=OrderStatus.NEW, ts_ms=4_000)

    with pytest.raises(OrderStateMachineError):
        sm.transition(
            order_id="old-2",
            new_status=OrderStatus.FILLED,
            filled_qty=Decimal("1.0"),
            avg_price=Decimal("50000"),
            ts_ms=5_000,
            reason="late_fill_after_cancel_replace",
        )

    assert sm.get("old-2").status == OrderStatus.CANCELED
    assert sm.get("new-2").status == OrderStatus.NEW
    assert sm.active_count() == 1
