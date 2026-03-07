from __future__ import annotations

from decimal import Decimal

import pytest

from execution.state_machine.machine import (
    InvalidTransitionError,
    OrderStateMachine,
    OrderStateMachineError,
)
from execution.state_machine.transitions import OrderStatus


def test_register_and_get_roundtrip() -> None:
    sm = OrderStateMachine()

    state = sm.register(
        order_id="o-1",
        client_order_id="c-1",
        symbol="BTCUSDT",
        side="buy",
        order_type="LIMIT",
        qty=Decimal("1.25"),
        price=Decimal("100.5"),
    )

    assert state.order_id == "o-1"
    assert state.client_order_id == "c-1"
    assert state.status == OrderStatus.PENDING_NEW
    assert state.qty == Decimal("1.25")
    assert state.price == Decimal("100.5")
    assert state.remaining_qty == Decimal("1.25")

    stored = sm.get("o-1")
    assert stored is not None
    assert stored.symbol == "BTCUSDT"
    assert stored.status == OrderStatus.PENDING_NEW


def test_transition_archives_terminal_order() -> None:
    sm = OrderStateMachine()
    sm.register(
        order_id="o-2",
        symbol="ETHUSDT",
        side="sell",
        order_type="MARKET",
        qty=Decimal("2"),
    )

    sm.transition(order_id="o-2", new_status=OrderStatus.NEW, ts_ms=1)
    state = sm.transition(
        order_id="o-2",
        new_status=OrderStatus.FILLED,
        filled_qty=Decimal("2"),
        avg_price=Decimal("2500"),
        ts_ms=2,
        reason="fill",
    )

    assert state.status == OrderStatus.FILLED
    assert state.filled_qty == Decimal("2")
    assert state.avg_price == Decimal("2500")
    assert state.is_terminal is True
    assert len(state.transitions) == 2
    assert state.transitions[0].to_status == OrderStatus.NEW
    assert state.transitions[1].to_status == OrderStatus.FILLED
    assert sm.active_count() == 0

    archived = sm.get("o-2")
    assert archived is not None
    assert archived.status == OrderStatus.FILLED


def test_invalid_transition_raises_custom_error() -> None:
    sm = OrderStateMachine()
    sm.register(
        order_id="o-3",
        symbol="BTCUSDT",
        side="buy",
        order_type="LIMIT",
        qty=Decimal("1"),
    )

    with pytest.raises(InvalidTransitionError):
        sm.transition(order_id="o-3", new_status=OrderStatus.CANCELED)



def test_duplicate_register_raises_custom_error() -> None:
    sm = OrderStateMachine()
    sm.register(
        order_id="o-4",
        symbol="BTCUSDT",
        side="buy",
        order_type="LIMIT",
        qty=Decimal("1"),
    )

    with pytest.raises(OrderStateMachineError):
        sm.register(
            order_id="o-4",
            symbol="BTCUSDT",
            side="buy",
            order_type="LIMIT",
            qty=Decimal("1"),
        )
