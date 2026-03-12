"""Late execution report safety tests."""
from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace

from execution.safety.timeout_tracker import OrderTimeoutTracker
from execution.state_machine.machine import OrderStateMachine
from execution.state_machine.transitions import OrderStatus


def test_late_fill_after_timeout_cancel_marks_order_filled() -> None:
    """A late fill after cancel timeout must converge to FILLED, not drift forever."""
    now = [100.0]
    sm = OrderStateMachine()
    cancel_calls: list[str] = []

    def _cancel(cmd: SimpleNamespace) -> None:
        cancel_calls.append(cmd.order_id)
        sm.transition(
            order_id=cmd.order_id,
            new_status=OrderStatus.PENDING_CANCEL,
            ts_ms=2_000,
            reason="timeout_cancel",
        )

    tracker = OrderTimeoutTracker(
        timeout_sec=5.0,
        cancel_fn=_cancel,
        clock_fn=lambda: now[0],
    )

    sm.register(
        order_id="o-1",
        symbol="BTCUSDT",
        side="buy",
        order_type="LIMIT",
        qty=Decimal("1.0"),
        price=Decimal("50000"),
    )
    sm.transition(order_id="o-1", new_status=OrderStatus.NEW, ts_ms=1_000)
    tracker.on_submit("o-1", SimpleNamespace(order_id="o-1"))

    now[0] = 106.0
    timed_out = tracker.check_timeouts()

    assert timed_out == ["o-1"]
    assert cancel_calls == ["o-1"]
    assert tracker.pending_count == 0
    assert sm.get("o-1").status == OrderStatus.PENDING_CANCEL

    # Venue execution report arrives after cancel request; state machine must
    # converge to FILLED because pending_cancel -> filled is explicitly allowed.
    state = sm.transition(
        order_id="o-1",
        new_status=OrderStatus.FILLED,
        filled_qty=Decimal("1.0"),
        avg_price=Decimal("50010"),
        ts_ms=3_000,
        reason="late_fill_after_cancel",
    )
    tracker.on_fill("o-1")

    assert state.status == OrderStatus.FILLED
    assert state.filled_qty == Decimal("1.0")
    assert state.avg_price == Decimal("50010")
    assert sm.active_count() == 0
    assert tracker.pending_count == 0
