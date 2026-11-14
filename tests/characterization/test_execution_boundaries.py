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
    assert sorted(_rows(executor)) == [(0, "inactive"), (1, "active")] and len(fake.algo) == 1


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


# --- the adapter takes its scope and journal from the caller and refuses what it cannot parse ---------------

def test_the_scope_and_the_journal_path_are_never_taken_from_the_environment(env, tmp_path, monkeypatch):
    _, adapter, _, _, _, _ = env
    monkeypatch.setenv("RISK_SCOPE", "binance-testnet-env")
    monkeypatch.setenv("ORDER_JOURNAL_PATH", str(tmp_path / "from-env.sqlite"))
    with pytest.raises(ValueError, match="RISK_SCOPE"):
        BinanceExecutionAdapter(adapter, journal_path=tmp_path / "orders2.sqlite", owner=OWNER)
    with pytest.raises(ValueError, match="journal path"):
        BinanceExecutionAdapter(adapter, scope="binance-testnet-ut", owner=OWNER)
    assert not (tmp_path / "from-env.sqlite").exists() and not (tmp_path / "orders2.sqlite").exists()


def test_an_order_whose_quantity_cannot_be_parsed_sends_nothing_even_with_a_position_to_close(env):
    fake, _, executor, _, consumer, statuses = env
    _buy(executor, consumer)
    good = _order(2, "sell", "0.05")
    broken = SimpleNamespace(header=good.header, order_id=good.order_id, symbol="BTCUSDT", side="sell",
                             qty="not-a-number", price=D("100"), reduce_only=False, reason="rebalance")
    sent = len(fake.sent)
    assert executor.send_order(broken) == ()
    assert len(fake.sent) == sent and fake.positions["BTCUSDT"] == D("0.1")     # not read as "close everything"
    assert executor._ready is False and (good.order_id, "blocked") in statuses


# --- step-9 audit F1/F2: a refused order is handled exactly once, and fills reach the cost log --------------

class _Breaker:
    def __init__(self):
        self.allow, self.successes, self.failures = True, 0, 0

    def allow_request(self):
        return self.allow

    def record_success(self):
        self.successes += 1

    def record_failure(self):
        self.failures += 1


def _with_breaker(tmp_path):
    fake = Venue(make_bars(10), clock=lambda: 1_800_000_000.0)
    fake.price["BTCUSDT"] = D("100")
    adapter = BinanceAdapter(BinanceConfig(api_key="k", api_secret="s"), client=fake)
    assert adapter.connect(symbols=("BTCUSDT",))
    breaker, statuses = _Breaker(), []
    executor = BinanceExecutionAdapter(adapter, breaker, scope="binance-testnet-ut", owner=OWNER, entry_mode="market",
                                       journal_path=tmp_path / "orders.sqlite", wait_seconds=0)
    executor._status_callback = lambda oid, status: statuses.append((oid, status))
    return fake, executor, breaker, statuses


@pytest.mark.parametrize("ready,allow,source,failures,ready_after,errors", [
    (False, True, OWNER, 0, False, 0),               # not reconciled yet: blocked, not a failure
    (True, False, OWNER, 0, True, 0),                # breaker open: blocked, and entries stay enabled
    (True, True, "target.ETHUSDT", 1, False, 1),     # another strategy's order: one failure, one error log
])
def test_a_refused_entry_is_reported_and_counted_exactly_once(tmp_path, caplog, ready, allow, source, failures,
                                                              ready_after, errors):
    fake, executor, breaker, statuses = _with_breaker(tmp_path)
    executor._ready, breaker.allow = ready, allow
    order = make_order(order_id="BTCUSDT:1:rebalance:0", source=source, symbol="BTCUSDT", side="buy", qty=D("0.1"),
                       price=D("100"), reduce_only=False, reason="rebalance")
    with caplog.at_level("ERROR"):
        assert executor.send_order(order) == ()
    assert statuses == [("BTCUSDT:1:rebalance:0", "blocked")]
    assert (breaker.failures, breaker.successes, executor._ready) == (failures, 0, ready_after)
    assert len([r for r in caplog.records if r.levelname == "ERROR"]) == errors and fake.sent == []


def test_a_repeated_intent_changes_nothing_even_while_entries_are_blocked(tmp_path, caplog):
    fake, executor, breaker, statuses = _with_breaker(tmp_path)
    executor._ready = True
    order = _order(1, "buy", "0.1")
    assert len(executor.send_order(order)) == 1
    executor._ready, before = False, (list(statuses), breaker.failures, len(fake.sent))
    with caplog.at_level("ERROR"):
        assert executor.send_order(order) == ()
    assert (statuses, breaker.failures, len(fake.sent)) == before and not caplog.records


def test_fills_are_written_to_the_cost_log_next_to_the_journal(tmp_path):
    import json
    _, executor, breaker, _ = _with_breaker(tmp_path)
    executor._ready = True
    (event,) = executor.send_order(_order(1, "buy", "0.1"))
    (row,) = [json.loads(line) for line in (tmp_path / "tca_binance.jsonl").read_text().splitlines()]
    assert (row["fill_id"], row["side"], row["qty"], row["ref_price"], row["fill_price"]) == (
        event.fill_id, "buy", 0.1, 100.0, 100.0)
    assert breaker.successes == 1


def test_an_active_stop_that_vanishes_from_the_venue_is_replaced_at_once(env):
    fake, _, executor, _, _, _ = env
    stops = _stops(env)
    assert stops.sync() == [] and _rows(executor) == [(0, "active")]
    fake.algo.clear()                                            # the venue no longer knows the stop
    assert stops.sync() == []                                    # no 90 s lookup grace for a stop that was active
    assert sorted(_rows(executor)) == [(0, "inactive"), (1, "active")] and len(fake.algo) == 1
