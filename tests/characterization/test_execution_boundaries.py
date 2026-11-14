"""Exit sizing, account reconciliation, fill consumption and stop bookkeeping of the Binance durable executor.

Written before migration step 9b moves this logic (plan 0.6, step 9): each test kills a defect that
survived the mutation run in current-state.md 5.5 or pins a stop state no test drove before.
"""
from __future__ import annotations

from dataclasses import replace
from decimal import Decimal as D
from types import SimpleNamespace

import pytest

from execution.adapters.binance import BinanceConfig
from execution.adapters.binance.adapter import BinanceAdapter
from execution.adapters.binance.execution_adapter import BinanceExecutionAdapter
from execution.adapters.binance.protective_stops import ProtectiveStopManager
from lab.orders import make_order
from runner.binance_target import TargetFillConsumer
from tests.unit.binance.fake_venue import FakeBinance, make_bars

OWNER = "target.BTCUSDT"


class Venue(FakeBinance):
    """Also keeps each order exactly as sent (the fake trims a reduce-only quantity in place)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sent: list[dict] = []

    def request_signed(self, *, method, path, params=None):
        if path == "/fapi/v1/order" and method == "POST":
            self.sent.append(dict(params or {}))
        return super().request_signed(method=method, path=path, params=params)


@pytest.fixture
def env(tmp_path):
    clock = {"now": 1_800_000_000.0}
    fake = Venue(make_bars(10), clock=lambda: clock["now"])
    fake.price["BTCUSDT"] = D("100")
    adapter = BinanceAdapter(BinanceConfig(api_key="k", api_secret="s"), client=fake)
    assert adapter.connect(symbols=("BTCUSDT",))
    executor = BinanceExecutionAdapter(adapter, scope="binance-testnet-ut", owner=OWNER, entry_mode="market",
                                       journal_path=tmp_path / "orders.sqlite", wait_seconds=0,
                                       clock=lambda: clock["now"])
    statuses = []
    executor._status_callback = lambda oid, status: statuses.append((oid, status))
    executor._ready = True
    return fake, adapter, executor, clock, TargetFillConsumer(executor), statuses


def _order(n, side, qty, *, reduce_only=False, reason="rebalance"):
    return make_order(order_id=f"BTCUSDT:{n}:{reason}:0", source=OWNER, symbol="BTCUSDT", side=side,
                      qty=D(qty), price=D("100"), reduce_only=reduce_only, reason=reason)


def _buy(executor, consumer, qty="0.1"):
    for event in executor.send_order(_order(1, "buy", qty)):
        consumer.consume(event)


def _posted(fake):
    return [(p["side"], D(p["quantity"]), p.get("reduceOnly")) for p in fake.sent]


# --- reductions are sized from the venue position and never exceed what this owner confirmed ---------------

@pytest.mark.parametrize("qty", ["0", "0.5"])
def test_a_reduction_closes_only_the_owned_part_of_a_larger_venue_position(env, qty):
    fake, _, executor, _, consumer, _ = env
    _buy(executor, consumer)
    fake.positions["BTCUSDT"] += D("0.2")                       # someone else's 0.2 on the same account
    executor.send_order(_order(2, "sell", qty, reduce_only=True, reason="exit"))
    assert _posted(fake)[-1] == ("SELL", D("0.1"), True)
    assert fake.positions["BTCUSDT"] == D("0.2")


@pytest.mark.parametrize("qty", ["0", "0.5"])
def test_a_reduction_is_capped_by_the_venue_position_when_it_is_smaller(env, qty):
    fake, _, executor, _, consumer, _ = env
    _buy(executor, consumer)
    fake.positions["BTCUSDT"] = D("0.04")                      # the venue already closed part of it
    executor.send_order(_order(2, "sell", qty, reduce_only=True, reason="exit"))
    assert _posted(fake)[-1] == ("SELL", D("0.04"), True)


def test_two_venue_positions_on_one_symbol_refuse_reductions_and_block_entries(env):
    fake, adapter, executor, _, consumer, statuses = env
    _buy(executor, consumer)
    adapter.get_positions = lambda symbol=None, strict=True: (
        SimpleNamespace(symbol="BTCUSDT", qty=D("0.1")), SimpleNamespace(symbol="BTCUSDT", qty=D("-0.05")))
    posts = len(fake.sent)
    assert executor.send_order(_order(2, "sell", "0", reduce_only=True, reason="exit")) == ()
    assert len(fake.sent) == posts and ("BTCUSDT:2:exit:0", "blocked") in statuses
    assert executor.reconcile_account("BTCUSDT") == ["BTCUSDT: multiple positions require explicit reconciliation",
                                                     "BTCUSDT: venue qty=0.05, ledger qty=0.1"]


@pytest.mark.parametrize("venue,owned_fill,side,qty", [
    ("0.1", None, "sell", "0"),        # venue position, but this owner confirmed none
    ("-0.1", "0.1", "buy", "0"),       # owner long, venue short (qty 0 takes its side from the venue)
    ("0.1", "0.1", "buy", "0.05"),     # a "reduction" that would add to the long
    ("0", "0.1", "sell", "0"),         # nothing on the venue
])
def test_a_reduction_that_cannot_be_matched_to_an_owned_position_is_never_sent(env, venue, owned_fill, side, qty):
    fake, _, executor, _, consumer, statuses = env
    if owned_fill:
        _buy(executor, consumer, owned_fill)
    fake.positions["BTCUSDT"] = D(venue)
    posts = len(fake.sent)
    assert executor.send_order(_order(2, side, qty, reduce_only=True, reason="exit")) == ()
    assert len(fake.sent) == posts
    assert ("BTCUSDT:2:exit:0", "blocked") in statuses


# --- reconciliation: any mismatch blocks entries until a clean pass ----------------------------------------

def _entry_is_sent(fake, executor, n):
    posts = len(fake.sent)
    executor.send_order(_order(n, "buy", "0.01"))
    return len(fake.sent) > posts


def test_a_ledger_mismatch_blocks_entries_and_a_clean_pass_unblocks_them(env):
    fake, _, executor, _, consumer, _ = env
    _buy(executor, consumer)
    fake.positions["BTCUSDT"] = D("0.13")
    assert executor.reconcile_account("BTCUSDT") == ["BTCUSDT: venue qty=0.13, ledger qty=0.1"]
    assert not _entry_is_sent(fake, executor, 5)
    fake.positions["BTCUSDT"] = D("0.1")
    assert executor.reconcile_account("BTCUSDT") == []
    assert _entry_is_sent(fake, executor, 6)


def test_an_open_order_this_system_did_not_place_blocks_entries(env):
    fake, _, executor, _, _, _ = env
    fake._new_order({"symbol": "BTCUSDT", "side": "BUY", "type": "LIMIT", "quantity": "0.01", "price": "90",
                     "newClientOrderId": "manual-1"})
    (issue,) = executor.reconcile_account("BTCUSDT")
    assert issue.startswith("BTCUSDT: unowned open order ")
    assert not _entry_is_sent(fake, executor, 5)


def test_a_failed_reconciliation_blocks_entries(env):
    fake, _, executor, _, _, _ = env
    fake.position_error = True
    assert executor.reconcile_account("BTCUSDT") == ["BTCUSDT: reconciliation failed (BinanceRetryableError)"]
    assert not _entry_is_sent(fake, executor, 5)


def test_an_unresolved_order_blocks_entries(env):
    fake, _, executor, _, _, _ = env
    fake.fault = "timeout_before"
    executor.send_order(_order(1, "buy", "0.1"))               # outcome unknown: stays unresolved
    assert executor.reconcile_account("BTCUSDT") == ["BTCUSDT: unresolved orders or legacy journal records"]
    assert not _entry_is_sent(fake, executor, 5)


def test_an_undelivered_fill_blocks_entries(env):
    fake, _, executor, _, consumer, _ = env
    executor.send_order(_order(1, "buy", "0.1"))               # filled, but never consumed
    assert executor.reconcile_account("BTCUSDT") == ["BTCUSDT: venue qty=0.1, ledger qty=0",
                                                     "BTCUSDT: undelivered fills"]
    for event in executor.reconcile_pending():
        consumer.consume(event)
    assert executor.reconcile_account("BTCUSDT") == []


# --- the fill consumer applies only events that match their durable exchange fact --------------------------

@pytest.mark.parametrize("change", [{"order_id": "BTCUSDT:9:rebalance:0"}, {"symbol": "ETHUSDT"}, {"side": "sell"},
                                    {"qty": D("0.2")}, {"price": D("101")}])
def test_the_consumer_refuses_an_event_that_differs_from_its_fact(env, change):
    _, _, executor, _, consumer, _ = env
    (event,) = executor.send_order(_order(1, "buy", "0.1"))
    with pytest.raises(ValueError, match="differs from its durable exchange fact"):
        consumer.consume(replace(event, **change))
    assert consumer.position.qty == 0 and executor.journal.checkpoint(OWNER) is None
    assert consumer.consume(event) is True and consumer.position.qty == D("0.1")


def test_the_consumer_refuses_an_unknown_fill_and_applies_a_known_one_once(env):
    _, _, executor, _, consumer, _ = env
    (event,) = executor.send_order(_order(1, "buy", "0.1"))
    with pytest.raises(ValueError, match="Unknown fill"):
        consumer.consume(replace(event, fill_id="binance-testnet-ut:BTCUSDT:1:t-forged"))
    assert consumer.consume(event) is True
    assert consumer.consume(event) is False
    assert consumer.position.qty == D("0.1")
    assert executor.journal.checkpoint(OWNER)["position"]["qty"] == "0.1"


# --- protective stops: an unknown placement outcome and a stop stuck in canceling ---------------------------

def _stops(env):
    fake, adapter, executor, clock, consumer, _ = env
    _buy(executor, consumer)
    return ProtectiveStopManager(adapter, executor, "BTCUSDT", stop_pct=D("0.04"), clock=lambda: clock["now"])


def _rows(executor):
    return [(r["attempt"], r["status"]) for r in executor.journal.stop_rows(OWNER, "BTCUSDT", live_only=False)]


def test_a_stop_whose_placement_was_lost_is_looked_up_not_placed_again(env):
    fake, _, executor, _, _, _ = env
    stops = _stops(env)
    original = fake.request_signed

    def lose_the_answer(*, method, path, params=None):
        result = original(method=method, path=path, params=params)
        if path == "/fapi/v1/algoOrder" and method == "POST":
            raise TimeoutError("answer lost after the venue accepted")
        return result

    fake.request_signed = lose_the_answer
    assert stops.sync() == ["BTCUSDT: protective stop placement outcome unknown; will query"]
    fake.request_signed = original
    assert _rows(executor) == [(0, "unknown")]
    assert stops.sync() == []
    assert _rows(executor) == [(0, "active")] and len(fake.algo) == 1


def test_a_stop_never_placed_is_retried_only_after_the_lookup_grace(env):
    fake, _, executor, clock, _, _ = env
    stops = _stops(env)
    original = fake.request_signed

    def fail_before_the_venue(*, method, path, params=None):
        if path == "/fapi/v1/algoOrder" and method == "POST":
            raise TimeoutError("never reached the venue")
        return original(method=method, path=path, params=params)

    fake.request_signed = fail_before_the_venue
    stops.sync()
    fake.request_signed = original
    clock["now"] += 89
    assert stops.sync() == [] and fake.algo == {}             # still inside the 90 s lookup grace
    clock["now"] += 1
    assert stops.sync() == []
    assert _rows(executor) == [(0, "inactive"), (1, "active")] and len(fake.algo) == 1


def test_a_stop_stuck_in_canceling_is_canceled_again_and_never_counts_as_the_stop(env):
    fake, _, executor, _, consumer, _ = env
    stops = _stops(env)
    stops.sync()
    (old,) = fake.algo
    original = fake.request_signed

    def ignore_cancels(*, method, path, params=None):
        if path == "/fapi/v1/algoOrder" and method == "DELETE":
            return {"code": 200, "msg": "success"}            # acknowledged, but the stop stays live
        return original(method=method, path=path, params=params)

    fake.request_signed = ignore_cancels
    for event in executor.send_order(_order(2, "buy", "0.05")):  # new position revision
        consumer.consume(event)
    stops.sync()
    statuses = {r["client_algo_id"]: r["status"] for r in executor.journal.stop_rows(OWNER, "BTCUSDT",
                                                                                      live_only=False)}
    assert statuses[old] == "canceling" and sorted(statuses.values()) == ["active", "canceling"]
    deletes = []

    def record_cancels(*, method, path, params=None):
        if path == "/fapi/v1/algoOrder" and method == "DELETE":
            deletes.append(params["clientAlgoId"])
        return ignore_cancels(method=method, path=path, params=params)

    fake.request_signed = record_cancels
    stops.sync()
    assert deletes == [old]                                    # canceled again on every pass
    assert sum(1 for s in fake.algo.values() if s["algoStatus"] == "NEW") == 2
    fake.request_signed = original
    stops.sync()                                               # the cancel finally takes effect
    stops.sync()
    statuses = {r["client_algo_id"]: r["status"] for r in executor.journal.stop_rows(OWNER, "BTCUSDT",
                                                                                      live_only=False)}
    assert statuses[old] == "inactive" and stops.has_active_stop()
