from __future__ import annotations

from decimal import Decimal

import pytest

_quant_hotpath = pytest.importorskip("_quant_hotpath")
from _quant_hotpath import RustOrderStateMachine  # noqa: E402

from execution.state_machine.projection import project_order  # noqa: E402
from execution.state_machine.transitions import OrderStatus  # noqa: E402


def test_project_order_matches_rust_order_state_machine_terminal_state() -> None:
    events = [
        {"order_id": "ord-1", "status": "new", "qty": "1.0", "filled_qty": "0", "avg_price": None, "ts_ms": 1000},
        {"order_id": "ord-1", "status": "partially_filled", "qty": "1.0", "filled_qty": "0.4", "avg_price": "42000",
            "ts_ms": 1100},
        {"order_id": "ord-1", "status": "filled", "qty": "1.0", "filled_qty": "1.0", "avg_price": "42100",
            "ts_ms": 1200},
    ]
    projection = project_order(events)
    assert projection is not None

    sm = RustOrderStateMachine()
    sm.register("ord-1", "BTCUSDT", "BUY", "LIMIT", "1.0", price="42000")
    sm.transition("ord-1", "new", ts_ms=1000)
    sm.transition("ord-1", "partially_filled", filled_qty="0.4", avg_price="42000", ts_ms=1100)
    rust_state = sm.transition("ord-1", "filled", filled_qty="1.0", avg_price="42100", ts_ms=1200)

    assert projection.order_id == rust_state.order_id
    assert projection.status is OrderStatus.FILLED
    assert projection.status.value == rust_state.status
    assert projection.qty == Decimal(rust_state.qty)
    assert projection.filled_qty == Decimal(rust_state.filled_qty)
    assert projection.avg_price == Decimal(rust_state.avg_price)
    assert projection.last_ts_ms == rust_state.last_update_ts
