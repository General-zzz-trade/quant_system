from __future__ import annotations

from dataclasses import asdict
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
import json

from execution.safety.timeout_tracker import OrderTimeoutTracker
from execution.state_machine.machine import OrderStateMachine
from execution.state_machine.transitions import OrderStatus


def _snapshot_state(sm: OrderStateMachine, order_id: str) -> dict:
    state = sm.get(order_id)
    assert state is not None
    return {
        "order_id": state.order_id,
        "client_order_id": state.client_order_id,
        "symbol": state.symbol,
        "side": state.side,
        "order_type": state.order_type,
        "qty": str(state.qty),
        "price": str(state.price) if state.price is not None else None,
        "transitions": [
            {
                "to_status": t.to_status.value,
                "ts_ms": t.ts_ms,
                "reason": t.reason,
            }
            for t in state.transitions
        ],
    }


def _restore_state(snapshot: dict) -> OrderStateMachine:
    sm = OrderStateMachine()
    sm.register(
        order_id=snapshot["order_id"],
        client_order_id=snapshot["client_order_id"],
        symbol=snapshot["symbol"],
        side=snapshot["side"],
        order_type=snapshot["order_type"],
        qty=Decimal(snapshot["qty"]),
        price=Decimal(snapshot["price"]) if snapshot["price"] is not None else None,
    )
    for item in snapshot["transitions"]:
        sm.transition(
            order_id=snapshot["order_id"],
            new_status=OrderStatus(item["to_status"]),
            ts_ms=int(item["ts_ms"]),
            reason=item["reason"],
        )
    return sm


def test_timeout_cancel_restart_then_late_fill_converges_to_filled(tmp_path: Path) -> None:
    now = [100.0]
    original = OrderStateMachine()

    def _cancel(cmd: SimpleNamespace) -> None:
        original.transition(
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

    original.register(
        order_id="o-timeout-1",
        symbol="BTCUSDT",
        side="buy",
        order_type="LIMIT",
        qty=Decimal("1.0"),
        price=Decimal("50000"),
    )
    original.transition(order_id="o-timeout-1", new_status=OrderStatus.NEW, ts_ms=1_000)
    tracker.on_submit("o-timeout-1", SimpleNamespace(order_id="o-timeout-1"))

    now[0] = 106.0
    assert tracker.check_timeouts() == ["o-timeout-1"]
    assert original.get("o-timeout-1").status == OrderStatus.PENDING_CANCEL

    checkpoint_path = tmp_path / "timeout_restart.json"
    checkpoint_path.write_text(json.dumps(_snapshot_state(original, "o-timeout-1")))

    restored = _restore_state(json.loads(checkpoint_path.read_text()))
    restored.transition(
        order_id="o-timeout-1",
        new_status=OrderStatus.FILLED,
        filled_qty=Decimal("1.0"),
        avg_price=Decimal("50010"),
        ts_ms=3_000,
        reason="late_fill_after_restart",
    )

    state = restored.get("o-timeout-1")
    assert state is not None
    assert state.status == OrderStatus.FILLED
    assert state.filled_qty == Decimal("1.0")
    assert state.avg_price == Decimal("50010")
    assert restored.active_count() == 0

