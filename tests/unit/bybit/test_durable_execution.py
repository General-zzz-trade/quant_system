"""Real native pipeline and Bybit durable boundary; all exchange IO is offline."""
from decimal import Decimal as D
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest

from engine.coordinator import CoordinatorConfig, EngineCoordinator
from engine.decision_bridge import DecisionBridge
from engine.durable_fills import DurableFillConsumer
from engine.execution_bridge import ExecutionBridge
from execution.adapters.bybit.execution_adapter import BybitExecutionAdapter
from execution.models.fills import CanonicalFill
from execution.models.orders import CanonicalOrder
from runner.execution_recovery import reconcile_execution
from tests.unit.decision.test_alpha_module import _make_module, _make_snapshot


def fill(trade_id="trade-1", *, order_id="exchange-1", qty="0.4", price="100",
         side="buy", fee="0", fee_asset="USDT", symbol="BTCUSDT"):
    return CanonicalFill(
        venue="bybit", symbol=symbol, order_id=order_id, trade_id=trade_id,
        fill_id=f"bybit:{symbol}:{trade_id}", side=side, qty=D(qty), price=D(price),
        fee=D(fee), fee_asset=fee_asset,
    )


class AccountExchange:
    """Canonical Bybit statuses, strict account queries, and observable placements."""

    venue = "bybit"

    def __init__(self):
        self.calls = []
        self.lookups = []
        self.orders = {}
        self.status = "new"
        self.filled_qty = D(0)
        self.fills = ()
        self.position_qty = D(0)
        self.hedge_positions = None
        self.open_orders = ()
        self.submit_timeout = False
        self.hide_order = False
        self.fail_positions = False
        self.fail_open_orders = False

    def send_market_order(self, symbol, side, qty, **kwargs):
        self.calls.append((symbol, side, D(str(qty)), kwargs))
        order_id = f"exchange-{len(self.calls)}"
        self.orders[order_id] = NS(symbol=symbol, side=side, qty=D(str(qty)),
                                   client_id=kwargs["client_order_id"])
        if self.submit_timeout:
            raise TimeoutError("exchange accepted; placement response was lost")
        return {"status": "submitted", "orderId": order_id}

    def send_limit_order(self, *args, **kwargs):
        raise AssertionError("Bybit market execution must not place limit orders")

    def cancel_order(self, *args, **kwargs):
        raise AssertionError("Recovery must not turn a Bybit market order into a limit lifecycle")

    def get_ticker(self, *args, **kwargs):
        raise AssertionError("Bybit market execution must not depend on a limit quote")

    def get_order(self, symbol, *, order_id="", client_order_id=""):
        self.lookups.append((order_id, client_order_id))
        if self.hide_order:
            return None
        for exchange_id, order in self.orders.items():
            if ((order_id and order_id == exchange_id)
                    or (not order_id and client_order_id == order.client_id)):
                return CanonicalOrder(
                    venue="bybit", symbol=order.symbol, order_id=exchange_id,
                    client_order_id=order.client_id, status=self.status, side=order.side,
                    order_type="market", tif="ioc", qty=order.qty, filled_qty=self.filled_qty,
                )
        return None

    def get_recent_fills(self, **kwargs):
        # Deliberately include unrelated records: the executor must verify identity.
        return tuple(self.fills)

    def get_positions(self, symbol="", *, strict=False):
        assert strict, "Durable reconciliation and close sizing must use strict account queries"
        if self.fail_positions:
            raise TimeoutError("positions unavailable")
        if self.hedge_positions is not None:
            return tuple(NS(symbol="BTCUSDT", qty=qty, is_flat=False, entry_price=D(100))
                         for qty in self.hedge_positions)
        return (NS(symbol="BTCUSDT", qty=self.position_qty, is_flat=self.position_qty == 0,
                   entry_price=D(100)),)

    def get_open_orders(self, symbol="", *, strict=False):
        assert strict, "Durable reconciliation must not treat a failed query as an empty account"
        if self.fail_open_orders:
            raise TimeoutError("open orders unavailable")
        return self.open_orders


def build(exchange, path):
    module, _, _, _ = _make_module(qty=1)
    coord = EngineCoordinator(cfg=CoordinatorConfig(
        symbol_default="BTCUSDT", symbols=("BTCUSDT",), starting_balance=1000,
    ))
    coord.attach_decision_bridge(DecisionBridge(coord.emit, [module]))
    execution = BybitExecutionAdapter(
        exchange, journal_path=path, scope="bybit-offline", owner="alpha.BTCUSDT", wait_seconds=0,
    )
    execution._status_callback = module.on_order_status
    coord.attach_execution_bridge(ExecutionBridge(
        execution, coord.emit, max_retries=0, on_order_status=module.on_order_status,
    ))
    coord._durable_fill_consumer = DurableFillConsumer(coord, module, execution)
    reconcile_execution(coord, module, restore=True)
    return coord, module, execution


def test_market_ack_without_fills_never_updates_position_or_scales_quantity(tmp_path, monkeypatch):
    monkeypatch.setenv("OKX_WEEKEND_QTY_FACTOR", "0.01")
    exchange = AccountExchange()
    coord, module, execution = build(exchange, tmp_path / "orders.sqlite")
    order = module._make_open_order(100, 1, D(1))[0]
    coord.emit(order)
    assert len(exchange.calls) == 1
    assert exchange.calls[0][2] == D(1)
    assert module._current_qty == 0
    assert module._signal == 0
    assert coord._store.get_position("BTCUSDT").qty == 0
    assert module._pending_orders == {order.order_id}
    assert len(execution.journal.orders(unresolved=True)) == 1
    assert not execution.journal.fills(execution.owner)


def test_unknown_timeout_is_submitted_once_and_recovered_by_client_id_after_restart(tmp_path):
    exchange = AccountExchange()
    coord, module, execution = build(exchange, tmp_path / "orders.sqlite")
    exchange.submit_timeout = exchange.hide_order = True
    order = module._make_open_order(100, 1, D(1))[0]
    coord.emit(order)
    original_client_id = exchange.calls[0][3]["client_order_id"]
    assert module._current_qty == 0
    execution.send_order(order)
    restarted, restored, execution = build(exchange, tmp_path / "orders.sqlite")
    assert not restored._execution_ready
    assert restored._pending_orders == {order.order_id}
    assert len(exchange.calls) == 1
    assert exchange.lookups[-1] == ("", original_client_id)

    exchange.hide_order = False
    exchange.status = "filled"
    exchange.filled_qty = exchange.position_qty = D(1)
    exchange.fills = (fill(qty="1"),)
    assert reconcile_execution(restarted, restored) == []
    assert restored._current_qty == D(1)
    assert restarted._store.get_position("BTCUSDT").qty == 100_000_000
    assert not restored._pending_orders
    assert restored._execution_ready
    assert len(exchange.calls) == 1


def test_partial_delayed_and_duplicate_fills_use_real_quantity_and_signed_fee(tmp_path):
    exchange = AccountExchange()
    coord, module, execution = build(exchange, tmp_path / "orders.sqlite")
    exchange.status = "partially_filled"
    exchange.filled_qty = exchange.position_qty = D("0.4")
    first = fill(fee="-0.02")
    exchange.fills = (first, first, fill("unrelated", order_id="another-order", qty="9"),
                      fill("wrong-side", side="sell", qty="9"))
    order = module._make_open_order(100, 1, D(1))[0]
    coord.emit(order)
    assert module._current_qty == D("0.4")
    assert module._entry_price == 100
    assert module._pending_orders == {order.order_id}
    assert len(execution.journal.fills(execution.owner, undelivered=False)) == 1

    restarted, restored, execution = build(exchange, tmp_path / "orders.sqlite")
    assert restored._current_qty == D("0.4")
    assert not restored._execution_ready
    exchange.status = "filled"
    exchange.filled_qty = exchange.position_qty = D(1)
    second = fill("trade-2", qty="0.6", price="110", fee="0.03")
    exchange.fills = (first, first, second, second)
    assert reconcile_execution(restarted, restored) == []
    assert restored._current_qty == D(1)
    assert restored._entry_price == 106
    assert not restored._pending_orders
    checkpoint = execution.journal.checkpoint(execution.owner)
    assert D(checkpoint["position"]["fees"]["USDT"]) == D("0.01")
    facts = execution.journal.fills(execution.owner, undelivered=False)
    assert [D(fact["fee"]) for fact in facts] == [D("-0.02"), D("0.03")]
    assert reconcile_execution(restarted, restored) == []
    assert execution.journal.checkpoint(execution.owner) == checkpoint
    assert len(exchange.calls) == 1


def test_partial_close_projects_existing_realized_pnl_and_signed_fees(tmp_path):
    exchange = AccountExchange()
    coord, module, execution = build(exchange, tmp_path / "orders.sqlite")
    exchange.status = "filled"
    exchange.filled_qty = exchange.position_qty = D(1)
    exchange.fills = (fill(qty="1", fee="1"),)
    coord.emit(module._make_open_order(100, 1, D(1))[0])

    close = module._make_close_order(110, 1, "signal_change")[0]
    exchange.filled_qty = D("0.4")
    exchange.position_qty = D("0.6")
    exchange.fills = (fill("close-1", order_id="exchange-2", side="sell",
                           qty="0.4", price="110", fee="-0.5"),)
    observed = []
    coord._durable_fill_consumer.observers.append(observed.append)
    coord.emit(close)

    checkpoint = execution.journal.checkpoint(execution.owner)
    assert D(checkpoint["position"]["realized_pnl"]) == D(4)
    assert D(checkpoint["position"]["fees"]["USDT"]) == D("0.5")
    assert observed[0].realized_pnl == D(4)
    account = coord._store.get_account()
    assert account.balance == 100_350_000_000
    assert account.realized_pnl == 400_000_000
    assert account.fees_paid == 50_000_000


def test_non_account_currency_fee_blocks_projection_and_keeps_outbox(tmp_path):
    exchange = AccountExchange()
    coord, module, execution = build(exchange, tmp_path / "orders.sqlite")
    exchange.status = "filled"
    exchange.filled_qty = D(1)
    exchange.fills = (fill(qty="1", fee="0.001", fee_asset="BTC"),)
    event = execution.send_order(module._make_open_order(100, 1, D(1))[0])[0]
    with pytest.raises(ValueError, match="fee currency"):
        coord.emit(event)
    assert module._current_qty == 0
    assert coord._store.get_position("BTCUSDT").qty == 0
    assert len(execution.journal.fills(execution.owner)) == 1


def test_pipeline_failure_rolls_back_precomputed_position_and_fee(tmp_path):
    exchange = AccountExchange()
    coord, module, execution = build(exchange, tmp_path / "orders.sqlite")
    exchange.status = "filled"
    exchange.filled_qty = D(1)
    exchange.fills = (fill(qty="1", fee="0.25"),)
    event = execution.send_order(module._make_open_order(100, 1, D(1))[0])[0]
    with patch.object(coord._pipeline, "apply", side_effect=RuntimeError("pipeline crash")):
        with pytest.raises(RuntimeError, match="pipeline crash"):
            coord.emit(event)
    consumer = coord._durable_fill_consumer
    assert consumer.position.qty == 0
    assert consumer.position.fees == {}
    assert coord._store.get_account().balance == 100_000_000_000
    assert len(execution.journal.fills(execution.owner)) == 1
    coord.emit(event)
    assert consumer.position.qty == D(1)


def test_crash_after_fill_journal_commit_before_dispatch_replays_once(tmp_path):
    exchange = AccountExchange()
    coord, module, execution = build(exchange, tmp_path / "orders.sqlite")
    exchange.status = "filled"
    exchange.filled_qty = exchange.position_qty = D(1)
    exchange.fills = (fill(qty="1"),)
    order = module._make_open_order(100, 1, D(1))[0]
    events = execution.send_order(order)
    assert module._current_qty == 0
    assert len(events) == len(execution.journal.fills(execution.owner)) == 1

    restarted, restored, execution = build(exchange, tmp_path / "orders.sqlite")
    assert restored._current_qty == D(1)
    assert restarted._store.get_position("BTCUSDT").qty == 100_000_000
    assert not execution.journal.fills(execution.owner)
    checkpoint = execution.journal.checkpoint(execution.owner)
    restarted.emit(events[0])
    assert execution.journal.checkpoint(execution.owner) == checkpoint
    assert len(exchange.calls) == 1


def test_consumer_failure_rolls_back_and_keeps_fill_available_for_retry(tmp_path):
    exchange = AccountExchange()
    coord, module, execution = build(exchange, tmp_path / "orders.sqlite")
    exchange.status = "filled"
    exchange.filled_qty = D("0.4")
    exchange.fills = (fill(),)
    event = execution.send_order(module._make_open_order(100, 1, D(1))[0])[0]
    with patch.object(coord._durable_fill_consumer.position, "apply", side_effect=RuntimeError("crash")):
        with pytest.raises(RuntimeError, match="crash"):
            coord.emit(event)
    assert module._current_qty == 0
    assert coord._store.get_position("BTCUSDT").qty == 0
    assert len(execution.journal.fills(execution.owner)) == 1
    coord.emit(event)
    assert module._current_qty == D("0.4")
    assert not execution.journal.fills(execution.owner)


def test_zero_quantity_close_sizes_owned_position_and_applies_only_real_partial_fill(tmp_path):
    exchange = AccountExchange()
    coord, module, execution = build(exchange, tmp_path / "orders.sqlite")
    exchange.status = "filled"
    exchange.filled_qty = exchange.position_qty = D(1)
    exchange.fills = (fill(qty="1"),)
    coord.emit(module._make_open_order(100, 1, D(1))[0])
    assert module._current_qty == D(1)
    close = module._make_close_order(110, 1, "signal_change")[0]
    assert close.qty == 0
    exchange.status = "partially_filled"
    exchange.filled_qty = D("0.3")
    exchange.fills = (fill("close-1", order_id="exchange-2", side="sell", qty="0.3", price="110"),)
    coord.emit(close)
    assert len(exchange.calls) == 2
    assert exchange.calls[1][1:3] == ("sell", D(1))
    assert exchange.calls[1][3]["reduce_only"] is True
    assert module._current_qty == D("0.7")
    assert module._signal == 1
    assert coord._store.get_position("BTCUSDT").qty == 70_000_000
    assert module._pending_reductions == {close.order_id}
    assert execution.journal.orders()[-1]["qty"] == "1"


@pytest.mark.parametrize("filled_qty", ["0", "0.4"])
def test_expired_market_order_clears_pending_with_only_confirmed_fills(tmp_path, filled_qty):
    exchange = AccountExchange()
    coord, module, execution = build(exchange, tmp_path / "orders.sqlite")
    exchange.status = "expired"
    exchange.filled_qty = exchange.position_qty = D(filled_qty)
    exchange.fills = (fill(),) if D(filled_qty) else ()
    order = module._make_open_order(100, 1, D(1))[0]
    coord.emit(order)
    assert module._current_qty == D(filled_qty)
    assert not module._pending_orders
    assert not execution.journal.orders(unresolved=True)
    assert execution.journal.orders()[0]["status"] == "expired"
    assert reconcile_execution(coord, module) == []


@pytest.mark.parametrize("problem", ["position", "open_order", "positions_timeout", "orders_timeout", "hedge"])
def test_account_inconsistency_or_failed_queries_block_entries_without_inventing_position(tmp_path, problem):
    exchange = AccountExchange()
    if problem == "position":
        exchange.position_qty = D(2)
    elif problem == "open_order":
        exchange.open_orders = (NS(order_id="external"),)
    elif problem == "positions_timeout":
        exchange.fail_positions = True
    elif problem == "orders_timeout":
        exchange.fail_open_orders = True
    else:
        exchange.hedge_positions = (D(1), D(-1))
    coord, module, execution = build(exchange, tmp_path / "orders.sqlite")
    assert not module._execution_ready
    assert not execution._ready
    assert module._current_qty == 0
    assert coord._store.get_position("BTCUSDT").qty == 0
    assert not [event for event in module.decide(_make_snapshot()) if hasattr(event, "order_id")]
    assert not exchange.calls


def test_zero_quantity_close_refuses_ambiguous_hedge_positions(tmp_path):
    exchange = AccountExchange()
    coord, module, execution = build(exchange, tmp_path / "orders.sqlite")
    exchange.hedge_positions = (D(1), D(-1))
    coord.emit(module._make_close_order(100, 1, "quick_loss")[0])
    assert not exchange.calls
    assert not execution.journal.orders()
    assert module._current_qty == 0


def test_unconsumed_close_cannot_reduce_another_owners_remaining_position(tmp_path):
    from dataclasses import replace
    exchange = AccountExchange()
    coord, module, execution = build(exchange, tmp_path / "orders.sqlite")
    exchange.status = "filled"
    exchange.filled_qty = exchange.position_qty = D(1)
    exchange.fills = (fill(qty="1"),)
    coord.emit(module._make_open_order(100, 1, D(1))[0])
    close = module._make_close_order(110, 1, "signal_change")[0]
    exchange.fills = (fill("close-1", order_id="exchange-2", side="sell", qty="1", price="110"),)
    # This owner's close is committed, but dispatch/consumer has not happened.
    assert len(execution.send_order(close)) == 1
    assert len(execution.journal.fills(execution.owner)) == 1
    assert module._current_qty == D(1)
    # The only venue quantity left could now belong to another strategy.
    exchange.position_qty = D(1)
    assert execution.send_order(replace(close, order_id="another-close-intent")) == ()
    assert len(exchange.calls) == 2  # original entry and first close only


def test_production_builder_attaches_bybit_durable_consumer_before_execution(tmp_path, monkeypatch):
    from runner.builders import alpha_builder
    monkeypatch.setenv("RISK_SCOPE", "bybit-offline")
    monkeypatch.setenv("ORDER_JOURNAL_PATH", str(tmp_path / "orders.sqlite"))
    monkeypatch.setattr(alpha_builder, "_build_data_sources", lambda *args, **kwargs: {})
    monkeypatch.setattr(alpha_builder, "get_initial_balance", lambda adapter: 1000)
    info = {"zscore_window": 10, "zscore_warmup": 2, "horizon_models": [], "config": {},
            "deadzone": 0.5, "min_hold": 1, "max_hold": 10}
    exchange = AccountExchange()
    coord, module = alpha_builder.build_coordinator(
        "BTCUSDT", "BTCUSDT", info, exchange, dry_run=False,
        risk_gate=NS(check=lambda event: NS(allowed=True)),
    )
    assert coord._durable_fill_consumer.executor is coord._execution_bridge.adapter
    assert coord._execution_bridge.adapter.owner == "alpha.BTCUSDT"
    assert not module._execution_ready
    assert reconcile_execution(coord, module, restore=True) == []
    assert module._execution_ready
    assert not exchange.calls


@pytest.mark.parametrize("response_lost", [False, True])
def test_v5_wire_response_runs_through_real_adapter_journal_and_native_consumer(tmp_path, response_lost):
    from unittest.mock import Mock
    from execution.adapters.bybit.adapter import BybitAdapter
    from execution.adapters.bybit.config import BybitConfig
    exchange = BybitAdapter(BybitConfig.demo(api_key="test", api_secret="test"))
    exchange._client = Mock()
    posted = {}

    def post(path, body):
        assert path == "/v5/order/create"
        posted.update(body)
        if response_lost:
            # Same shape as BybitRestClient._send after a network timeout.
            return {"retCode": -1, "retMsg": "accepted; response lost", "retryable": True}
        return {"retCode": 0, "result": {"orderId": "venue-1", "orderLinkId": body["orderLinkId"]}}

    def get(path, params):
        rows = []
        if path == "/v5/position/list":
            if posted:
                rows = [{"symbol": "BTCUSDT", "side": "Buy", "size": "0.4",
                         "avgPrice": "101", "markPrice": "101", "unrealisedPnl": "0"}]
        elif path == "/v5/order/realtime" and (params.get("orderId") or params.get("orderLinkId")):
            assert params["orderLinkId"] == posted["orderLinkId"]
            rows = [{"symbol": "BTCUSDT", "orderId": "venue-1", "orderLinkId": posted["orderLinkId"],
                     "side": "Buy", "orderType": "Market", "orderStatus": "PartiallyFilledCanceled",
                     "qty": "1", "cumExecQty": "0.4", "avgPrice": "101", "price": "0"}]
        elif path == "/v5/execution/list":
            assert params["orderId"] == "venue-1"
            rows = [{"symbol": "BTCUSDT", "orderId": "venue-1", "execId": "trade-1", "execType": "Trade",
                     "side": "Buy", "execQty": "0.4", "execPrice": "101", "execFee": "-0.02",
                     "feeCurrency": "USDT", "execTime": "1000"}]
        else:
            assert path == "/v5/order/realtime", path
        return {"retCode": 0, "result": {"list": rows, "nextPageCursor": ""}}

    exchange._client.post.side_effect = post
    exchange._client.get.side_effect = get
    coord, module, execution = build(exchange, tmp_path / "orders.sqlite")
    order = module._make_open_order(100, 1, D(1))[0]
    coord.emit(order)
    assert module._current_qty == D("0.4")
    assert module._entry_price == 101
    assert coord._store.get_position("BTCUSDT").qty == 40_000_000
    assert execution.journal.orders()[0]["client_id"] == posted["orderLinkId"]
    assert execution.journal.orders()[0]["status"] == "canceled"
    assert not execution.journal.fills(execution.owner)
    assert execution.journal.checkpoint(execution.owner)["position"]["fees"] == {"USDT": "-0.02"}
    from engine.dispatcher import DuplicateEventError
    with pytest.raises(DuplicateEventError):
        coord.emit(order)
    assert execution.send_order(order) == ()
    assert exchange._client.post.call_count == 1
    assert not module._pending_reductions
