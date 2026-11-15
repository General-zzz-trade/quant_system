"""OwnerExecutor: the execution service as a session uses it, with no legacy event type (migration step 10)."""
from __future__ import annotations

import json
import threading
from decimal import Decimal as D

import pytest

from tests.unit.trading.test_execution_service import OWNER, SCOPE, Log, Venue
from trading.adapters.sqlite.order_journal import SqliteOrderJournal
from trading.application.execution.executor import Fill, OwnerExecutor
from trading.application.execution.fills import TargetFillConsumer
from trading.application.execution.lifecycle import OrderLifecycle
from trading.application.execution.service import ExecutionService
from trading.domain.orders import BLOCKED, OrderIntent
from tests.support.state_db import state_db


@pytest.fixture
def env(tmp_path):
    def build(*, entry_mode="market", venue=None, log=None):
        venue = venue or Venue()
        clocks = {"wall": 1000.0, "mono": 10.0}
        orders = OrderLifecycle(venue, SqliteOrderJournal(state_db(tmp_path / "orders.sqlite"), SCOPE), SCOPE,
                                clock=lambda: clocks["wall"], sleeper=lambda s: None, lock=threading.RLock(),
                                wait_seconds=0.5, owner=OWNER,
                                entry_order_type="limit" if entry_mode == "maker" else "market",
                                not_found_grace_seconds=90,
                                entry_ttl_seconds=3 * 3600.0 if entry_mode == "maker" else None)
        service = ExecutionService(venue, orders, owner=OWNER, entry_mode=entry_mode, limit_offset=D("0.0002"),
                                   lock=threading.RLock(), clock=lambda: clocks["wall"],
                                   monotonic=lambda: clocks["mono"], event_log=log)
        executor = OwnerExecutor(service, scope=SCOPE)
        statuses = []
        executor.on_status(lambda intent_id, status: statuses.append((intent_id, status)))
        return executor, venue, statuses
    return build


def _intent(n, side="buy", qty="1", *, reduce_only=False, price="100", reason="rebalance"):
    return OrderIntent(order_id=f"BTCUSDT:{n}:{reason}:0", source=OWNER, symbol="BTCUSDT", side=side, qty=D(qty),
                       price=D(price), reduce_only=reduce_only, reason=reason)


def _ready(executor):
    assert executor.reconcile_account("BTCUSDT") == []


def test_an_intent_is_stored_as_plain_fields_and_its_fills_come_back_typed(env):
    executor, venue, _ = env()
    _ready(executor)
    fills = executor.send_order(_intent(1, price="100.5"))
    assert fills == (Fill(fill_id=f"{SCOPE}:BTCUSDT:1:t1", order_id="BTCUSDT:1:rebalance:0", symbol="BTCUSDT",
                          side="buy", qty=D("1"), price=D("100")),)
    (row,) = executor.journal.orders(OWNER)
    assert json.loads(row["parent"]) == {"order_id": "BTCUSDT:1:rebalance:0", "symbol": "BTCUSDT", "side": "buy",
                                         "price": "100.5", "reason": "rebalance"}
    assert venue.calls == [("market", "buy", 1.0, False)]


def test_every_fill_is_delivered_until_consumed_and_consumed_exactly_once(env):
    executor, _, _ = env()
    _ready(executor)
    executor.send_order(_intent(1))
    consumer = TargetFillConsumer(executor)
    (fill,) = executor.reconcile_pending()
    assert executor.reconcile_pending() == (fill,)                # delivered again until it is consumed
    assert consumer.consume(fill) is True and consumer.position.qty == D("1")
    assert executor.reconcile_pending() == ()
    assert TargetFillConsumer(executor).position.qty == D("1")    # the checkpoint survives a restart


def test_a_fill_that_differs_from_the_stored_fact_is_refused(env):
    executor, _, _ = env()
    _ready(executor)
    (fill,) = executor.send_order(_intent(1))
    consumer = TargetFillConsumer(executor)
    with pytest.raises(ValueError, match="differs"):
        consumer.consume(Fill(fill.fill_id, fill.order_id, fill.symbol, fill.side, fill.qty + 1, fill.price))
    assert consumer.position.qty == 0


def test_an_entry_before_a_clean_reconciliation_is_blocked_and_a_repeat_places_nothing(env):
    executor, venue, statuses = env()
    assert executor.ready is False
    assert executor.send_order(_intent(1)) == () and statuses == [("BTCUSDT:1:rebalance:0", BLOCKED)]
    _ready(executor)
    assert len(executor.send_order(_intent(1))) == 1
    assert executor.send_order(_intent(1)) == () and len(venue.calls) == 1
    executor.notify_status()
    assert statuses[-1] == ("BTCUSDT:1:rebalance:0", "filled")


def test_blocking_entries_stops_the_next_entry_but_never_a_reduction(env):
    executor, venue, statuses = env()
    _ready(executor)
    (fill,) = executor.send_order(_intent(1))
    TargetFillConsumer(executor).consume(fill)
    executor.block_entries()
    assert executor.ready is False and executor.send_order(_intent(2)) == ()
    assert statuses[-1] == ("BTCUSDT:2:rebalance:0", BLOCKED)
    (closing,) = executor.send_order(_intent(3, side="sell", reduce_only=True, reason="kill_switch"))
    assert (closing.side, closing.qty) == ("sell", D("1")) and venue.calls[-1] == ("market", "sell", 1.0, True)


def test_a_triggered_stop_is_adopted_once_and_its_fill_reaches_the_owner(env):
    executor, venue, _ = env()
    _ready(executor)
    consumer = TargetFillConsumer(executor)
    consumer.consume(executor.send_order(_intent(1))[0])
    venue.send_market_order("BTCUSDT", "sell", "1", reduce_only=True, client_order_id="autoclose-1")   # the stop
    adopt = dict(intent_id="stop:1", symbol="BTCUSDT", side="sell", order_id="2", client_order_id="autoclose-1",
                 reason="protective_stop")
    assert executor.adopt_exchange_order(**adopt) is True and executor.adopt_exchange_order(**adopt) is False
    row = next(r for r in executor.journal.orders(OWNER) if r["intent_id"] == "stop:1")
    assert json.loads(row["parent"]) == {"order_id": "stop:1", "symbol": "BTCUSDT", "side": "sell", "price": "0",
                                         "reason": "protective_stop"}
    (fill,) = executor.reconcile_pending()
    assert (fill.order_id, fill.side, fill.qty) == ("stop:1", "sell", D("1"))
    assert consumer.consume(fill) is True and consumer.position.qty == 0


def test_the_executor_names_its_owner_scope_and_journal(env):
    executor, _, _ = env()
    assert (executor.owner, executor.scope) == (OWNER, SCOPE)
    assert executor.journal.scope == SCOPE
    with pytest.raises(ValueError, match="Unknown fill"):
        executor.fill_fact("no-such-fill")


def test_orders_journaled_by_the_legacy_adapter_are_collected_by_the_new_executor(env, tmp_path):
    """The stored record of the legacy adapter carries an event header; the new executor ignores it."""
    from tests.support.legacy_execution_adapter import BinanceExecutionAdapter
    from lab.orders import make_order
    venue, log = Venue(), Log()
    legacy = BinanceExecutionAdapter(venue, owner=OWNER, scope=SCOPE, journal_path=tmp_path / "orders.sqlite",
                                     entry_mode="maker")
    assert legacy.reconcile_account("BTCUSDT") == []
    legacy.send_order(make_order(order_id="BTCUSDT:1:rebalance:0", source=OWNER, symbol="BTCUSDT", side="buy",
                                 qty=D("1"), price=D("250"), reduce_only=False, reason="rebalance"))
    venue.orders["1"].update(state="filled", filled_qty="1", trades=[("t1", "1", "249.95")])
    executor, _, _ = env(entry_mode="maker", venue=venue, log=log)
    (fill,) = executor.reconcile_pending()
    assert (fill.order_id, fill.qty, fill.price) == ("BTCUSDT:1:rebalance:0", D("1"), D("249.95"))
    assert log.rows[0][1]["ref_price"] == 250.0
    assert TargetFillConsumer(executor).consume(fill) is True
