"""Actual Rust pipeline + strategy + OKX boundary; the exchange is offline."""
from decimal import Decimal as D
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest

from engine.coordinator import EngineCoordinator, CoordinatorConfig
from engine.decision_bridge import DecisionBridge
from engine.durable_fills import DurableFillConsumer
from engine.execution_bridge import ExecutionBridge
from execution.adapters.okx.execution_adapter import OkxExecutionAdapter
from runner.execution_recovery import reconcile_execution
from tests.offline.test_execution_safety import Exchange, fill
from tests.unit.decision.test_alpha_module import _make_module, _make_snapshot


class AccountExchange(Exchange):
    def __init__(self):
        super().__init__()
        self.position_qty = D(0)
        self.open_orders = ()
        self.fail_positions = False

    def get_positions(self, symbol=""):
        if self.fail_positions:
            raise TimeoutError("offline")
        return (NS(symbol="BTCUSDT", qty=self.position_qty, is_flat=self.position_qty == 0,
                   entry_price=D(100)),)

    def get_open_orders(self, **kwargs):
        return self.open_orders


def build(exchange, path):
    module, _, _, _ = _make_module(qty=1)
    coord = EngineCoordinator(cfg=CoordinatorConfig(symbol_default="BTCUSDT", symbols=("BTCUSDT",),
                                                    starting_balance=1000))
    coord.attach_decision_bridge(DecisionBridge(coord.emit, [module]))
    execution = OkxExecutionAdapter(exchange, journal_path=path, scope="test-demo", owner="alpha.BTCUSDT")
    execution._status_callback = module.on_order_status
    coord.attach_execution_bridge(ExecutionBridge(execution, coord.emit, max_retries=0,
                                                  on_order_status=module.on_order_status))
    coord._durable_fill_consumer = DurableFillConsumer(coord, module, execution)
    reconcile_execution(coord, module, restore=True)
    return coord, module, execution


@pytest.fixture(autouse=True)
def settings(monkeypatch):
    monkeypatch.setenv("OKX_LIMIT_FILL_TIMEOUT_S", "0")
    monkeypatch.setenv("OKX_WEEKEND_QTY_FACTOR", "1")


def test_crash_after_fill_written_then_restart_delivers_and_deduplicates(tmp_path):
    exchange = AccountExchange()
    coord, module, execution = build(exchange, tmp_path / "orders.sqlite")
    order = module._make_open_order(100, 1, D(1))[0]
    exchange.fills = [fill()]
    exchange.filled_qty = D("0.4")
    exchange.position_qty = D("0.4")
    # Crash window: executor has returned facts; dispatcher has not seen them.
    events = execution.send_order(order)
    assert module._current_qty == 0
    assert len(execution.journal.fills(execution.owner)) == 1
    restarted, recovered, new_execution = build(exchange, tmp_path / "orders.sqlite")
    assert recovered._current_qty == D("0.4")
    assert recovered._entry_price == 100
    assert restarted._store.get_position("BTCUSDT").qty == 40_000_000
    checkpoint = new_execution.journal.checkpoint(new_execution.owner)
    restarted.emit(events[0])
    assert new_execution.journal.checkpoint(new_execution.owner) == checkpoint
    assert recovered._current_qty == D("0.4")
    assert not new_execution.journal.fills(new_execution.owner)
    assert len([c for c in exchange.calls if c[0] == "limit"]) == 1


def test_partial_fill_restart_then_remaining_fill(tmp_path):
    exchange = AccountExchange()
    coord, module, execution = build(exchange, tmp_path / "orders.sqlite")
    order = module._make_open_order(100, 1, D(1))[0]
    exchange.cancel_ack_only = True
    exchange.fills = [fill()]
    exchange.filled_qty = exchange.position_qty = D("0.4")
    coord.emit(order)
    assert module._current_qty == D("0.4")
    assert module._pending_orders == {order.order_id}
    restarted, recovered, execution = build(exchange, tmp_path / "orders.sqlite")
    assert not recovered._execution_ready
    exchange.fills += [fill("trade-2", qty="0.6", price="110")]
    exchange.filled_qty = exchange.position_qty = D(1)
    exchange.state = "filled"
    assert reconcile_execution(restarted, recovered) == []
    assert recovered._current_qty == D(1)
    assert recovered._entry_price == 106
    assert not recovered._pending_orders
    assert recovered._execution_ready
    assert restarted._store.get_position("BTCUSDT").qty == 100_000_000


def test_consumer_failure_rolls_back_rust_and_python_state(tmp_path):
    exchange = AccountExchange()
    coord, module, execution = build(exchange, tmp_path / "orders.sqlite")
    order = module._make_open_order(100, 1, D(1))[0]
    exchange.fills = [fill()]
    exchange.filled_qty = exchange.position_qty = D("0.4")
    event = execution.send_order(order)[0]
    with patch.object(coord._durable_fill_consumer.position, "apply", side_effect=RuntimeError("crash")):
        with pytest.raises(RuntimeError):
            coord.emit(event)
    assert module._current_qty == 0
    assert coord._store.get_position("BTCUSDT").qty == 0
    assert len(execution.journal.fills(execution.owner)) == 1
    coord.emit(event)  # volatile dedup must not discard this retry
    assert module._current_qty == D("0.4")


@pytest.mark.parametrize("problem", ["position", "open_order", "timeout"])
def test_reconciliation_blocks_without_importing_unowned_positions(tmp_path, problem):
    exchange = AccountExchange()
    if problem == "position":
        exchange.position_qty = D(2)
    elif problem == "open_order":
        exchange.open_orders = (NS(order_id="external"),)
    else:
        exchange.fail_positions = True
    coord, module, execution = build(exchange, tmp_path / "orders.sqlite")
    assert not module._execution_ready
    assert module._current_qty == 0
    assert not execution._ready
    assert not [e for e in module.decide(_make_snapshot()) if hasattr(e, "order_id")]


def test_decision_waits_for_fills_and_rejection_does_not_change_position():
    module, _, _, _ = _make_module()
    events = list(module.decide(_make_snapshot()))
    order = next(e for e in events if hasattr(e, "order_id"))
    assert module._current_qty == 0
    assert module._entry_price == 0
    assert not [e for e in module.decide(_make_snapshot()) if hasattr(e, "order_id")]
    module.on_order_status(order.order_id, "rejected")
    assert not module._pending_orders
    assert module._signal == 0


def test_partial_close_preserves_remaining_position():
    module, _, _, _ = _make_module()
    order = module._make_open_order(100, 1, D(1))[0]
    module.on_fill(NS(order_id=order.order_id, fill_id="one", qty=D(1), price=D(100), side="buy"))
    module.on_order_status(order.order_id, "filled")
    close = module._make_close_order(110, 1, "signal_change")[0]
    assert module._current_qty == D(1)
    module.on_fill(NS(order_id=close.order_id, fill_id="two", qty=D("0.3"), price=D(110), side="sell"))
    assert module._current_qty == D("0.7")
    assert module._signal == 1
    assert module._pending_orders == {close.order_id}


def test_fill_event_cannot_change_persisted_exchange_fact(tmp_path):
    from dataclasses import replace
    exchange = AccountExchange()
    coord, module, execution = build(exchange, tmp_path / "orders.sqlite")
    order = module._make_open_order(100, 1, D(1))[0]
    exchange.fills = [fill()]
    exchange.filled_qty = D("0.4")
    event = execution.send_order(order)[0]
    with pytest.raises(ValueError, match="differs"):
        coord.emit(replace(event, qty=D(10), _rust=None))
    assert coord._store.get_position("BTCUSDT").qty == 0
    assert module._current_qty == 0


def test_observers_run_after_commit_and_only_once(tmp_path):
    exchange = AccountExchange()
    coord, module, execution = build(exchange, tmp_path / "orders.sqlite")
    observed = []
    def observe(event):
        assert not execution.journal.fills(execution.owner)
        observed.append(event.fill_id)
    coord._durable_fill_consumer.observers.append(observe)
    order = module._make_open_order(100, 1, D(1))[0]
    exchange.fills = [fill()]
    exchange.filled_qty = D("0.4")
    event = execution.send_order(order)[0]
    coord.emit(event)
    coord.emit(event)
    assert observed == [event.fill_id]


def test_partial_reduction_does_not_count_a_completed_stop():
    module, _, _, _ = _make_module()
    opening = module._make_open_order(100, 1, D(1))[0]
    module.on_fill(NS(order_id=opening.order_id, fill_id="entry", qty=D(1), price=D(100), side="buy"))
    module.on_order_status(opening.order_id, "filled")
    closing = module._make_close_order(90, 1, "quick_loss")[0]
    assert module._consecutive_stops == 0
    module.on_fill(NS(order_id=closing.order_id, fill_id="part", qty=D("0.4"), price=D(90), side="sell"))
    assert module._consecutive_stops == 0
    module.on_fill(NS(order_id=closing.order_id, fill_id="rest", qty=D("0.6"), price=D(90), side="sell"))
    assert module._consecutive_stops == 1
    assert module._current_qty == 0


def test_trailing_peak_and_hold_counters_survive_market_checkpoint(tmp_path):
    exchange = AccountExchange()
    coord, module, execution = build(exchange, tmp_path / "orders.sqlite")
    order = module._make_open_order(100, 1, D(1))[0]
    exchange.fills = [fill()]
    exchange.filled_qty = exchange.position_qty = D("0.4")
    coord.emit(order)
    module._bars_processed = 50
    module._entry_bar = 42
    module._trade_peak = 130
    coord._durable_fill_consumer.checkpoint_market()
    restarted, restored, execution = build(exchange, tmp_path / "orders.sqlite")
    assert restored._trade_peak == 130
    assert restored._bars_processed - restored._entry_bar == 8
    assert restored._current_qty == D("0.4")
