"""The exchange boundary speaks typed domain records; Binance's own words stay in its adapter (step 10b)."""
from __future__ import annotations

import dataclasses
from decimal import Decimal as D

import pytest

from tests.unit.binance.fake_venue import FakeBinance, make_bars
from tests.unit.trading.test_execution_service import OWNER, SCOPE
from trading.adapters.binance.adapter import BinanceAdapter
from trading.adapters.binance.config import BinanceConfig
from trading.adapters.sqlite.order_journal import SqliteOrderJournal
from trading.application.execution.stops import ProtectiveStopManager
from trading.domain.venue import (STOP_ACTIVE, STOP_GONE, STOP_TRIGGERED, OpenOrder, StopOrder, VenueOrder,
                                  VenuePosition)


@pytest.fixture
def venue():
    clock = {"now": 1_800_000_000.0}
    fake = FakeBinance(make_bars(10), clock=lambda: clock["now"])
    fake.price["BTCUSDT"] = D("100")
    adapter = BinanceAdapter(BinanceConfig(api_key="k", api_secret="s"), client=fake)
    assert adapter.connect(symbols=("BTCUSDT",))
    return fake, adapter


def test_orders_positions_and_open_orders_are_frozen_records(venue):
    fake, adapter = venue
    placed = adapter.send_market_order("BTCUSDT", "buy", 0.1, client_order_id="c1")
    order = adapter.get_order("BTCUSDT", order_id=placed["orderId"])
    assert order == VenueOrder(order_id=placed["orderId"], client_order_id="c1", symbol="BTCUSDT", side="buy",
                               state="filled", filled_qty=D("0.1"), avg_price=D("0"))
    with pytest.raises(dataclasses.FrozenInstanceError):
        order.state = "new"
    (position,) = adapter.get_positions("BTCUSDT")
    assert isinstance(position, VenuePosition) and position.qty == D("0.1")
    resting = adapter.send_limit_order("BTCUSDT", "buy", 1.0, 99.0, client_order_id="c2")
    assert adapter.get_open_orders("BTCUSDT") == (
        OpenOrder(order_id=resting["orderId"], client_order_id="c2", symbol="BTCUSDT", side="buy", type="LIMIT"),)
    assert adapter.get_order("BTCUSDT", client_order_id="c2").state == "new"
    assert adapter.get_order("BTCUSDT", client_order_id="never-sent") is None


@pytest.mark.parametrize("word,state", [("NEW", STOP_ACTIVE), ("TRIGGERING", STOP_ACTIVE),
                                        ("TRIGGERED", STOP_TRIGGERED), ("FINISHED", STOP_TRIGGERED),
                                        ("CANCELED", STOP_GONE), ("EXPIRED", STOP_GONE), ("REJECTED", STOP_GONE),
                                        ("", STOP_GONE), ("SOMETHING_NEW", STOP_GONE)])
def test_the_adapter_translates_every_stop_status_into_one_of_three_states(venue, word, state):
    fake, adapter = venue
    adapter.send_market_order("BTCUSDT", "buy", 0.1, client_order_id="c1")
    assert adapter.place_stop_market("BTCUSDT", "sell", D("96"), client_algo_id="s1")["status"] == "submitted"
    fake.algo["s1"].update(algoStatus=word, actualOrderId="77")
    stop = adapter.get_algo_order(client_algo_id="s1")
    assert isinstance(stop, StopOrder)
    assert (stop.state, stop.status, stop.actual_order_id, stop.side) == (state, word, "77", "sell")
    assert adapter.get_algo_order(client_algo_id="missing") is None


def test_only_the_adapter_knows_which_rejection_means_the_price_is_already_beyond_the_stop(venue):
    fake, adapter = venue
    adapter.send_market_order("BTCUSDT", "buy", 0.1, client_order_id="c1")
    assert adapter.place_stop_market("BTCUSDT", "sell", D("96"), client_algo_id="s1").get("breached", False) is False
    beyond = adapter.place_stop_market("BTCUSDT", "sell", D("1000"), client_algo_id="s2")
    assert (beyond["status"], beyond["breached"]) == ("rejected", True)
    fake.fault = None
    original = fake.request_signed

    def refuse(*, method, path, params=None):
        if path == "/fapi/v1/algoOrder" and method == "POST":
            fake._error(-4120, "not supported")
        return original(method=method, path=path, params=params)

    fake.request_signed = refuse
    other = adapter.place_stop_market("BTCUSDT", "sell", D("96"), client_algo_id="s3")
    assert (other["status"], other["breached"]) == ("rejected", False)


class _Owner:
    """The stop manager's view of an executor, over a real journal."""

    def __init__(self, tmp_path):
        self.journal, self.scope, self.owner, self.adopted = SqliteOrderJournal(tmp_path / "o.sqlite", SCOPE), \
            SCOPE, OWNER, []

    def adopt_exchange_order(self, **kw):
        self.adopted.append(kw)
        return True


class _StopVenue:
    """A venue that speaks only the domain's words: no exchange status, no error code."""

    def __init__(self):
        self.stops, self.placement, self.cancelled = {}, {"status": "submitted"}, []

    def place_stop_market(self, symbol, side, trigger_price, *, client_algo_id):
        if self.placement["status"] == "submitted":
            self.stops[client_algo_id] = StopOrder(algo_id="1", client_algo_id=client_algo_id, symbol=symbol,
                                                   side=side, status="?", state=STOP_ACTIVE,
                                                   trigger_price=trigger_price, actual_order_id="")
        return dict(self.placement)

    def get_algo_order(self, *, client_algo_id):
        return self.stops.get(client_algo_id)

    def cancel_algo_order(self, *, client_algo_id):
        self.cancelled.append(client_algo_id)

    def get_order(self, symbol, *, order_id="", client_order_id=""):
        return VenueOrder(order_id=order_id, client_order_id="autoclose", symbol=symbol, side="sell",
                          state="filled", filled_qty=D("1"), avg_price=D("95"))


def _manager(tmp_path, venue):
    tmp_path.mkdir(parents=True, exist_ok=True)
    owner = _Owner(tmp_path)
    owner.journal.insert_fill(fill_id="f1", intent_id="i1", owner=OWNER, payload={"fill_id": "f1"})
    owner.journal.consume(OWNER, "f1", lambda: {
        "position": {"qty": "1", "entry_price": "100", "realized_pnl": "0", "fees": {}},
        "revision": "f1", "strategy": {}}, lambda: None, expected_revision=None)
    return owner, ProtectiveStopManager(venue, owner, "BTCUSDT", stop_pct=D("0.04"), clock=lambda: 1000.0)


def test_the_stop_manager_acts_on_the_three_states_only(tmp_path):
    venue = _StopVenue()
    owner, stops = _manager(tmp_path, venue)
    assert stops.sync() == [] and stops.has_active_stop()
    (client_id,) = venue.stops
    assert stops.sync() == [] and len(venue.stops) == 1                      # active: nothing to do

    venue.stops[client_id] = dataclasses.replace(venue.stops[client_id], state=STOP_GONE)
    assert stops.sync() == [] and len(venue.stops) == 2                      # gone: a new stop is placed
    assert stops.has_active_stop()

    newest = [k for k in venue.stops if k != client_id][0]
    venue.stops[newest] = dataclasses.replace(venue.stops[newest], state=STOP_TRIGGERED, actual_order_id="55")
    assert "BTCUSDT: protective stop triggered" in stops.sync()
    assert [(a["order_id"], a["side"], a["reason"]) for a in owner.adopted] == [("55", "sell", "protective_stop")]


def test_a_breached_stop_is_reported_from_the_flag_and_never_from_an_error_code(tmp_path):
    venue = _StopVenue()
    venue.placement = {"status": "rejected", "breached": True, "msg": "would trigger"}
    _, stops = _manager(tmp_path, venue)
    assert any("stop_breached" in issue for issue in stops.sync())

    venue = _StopVenue()
    venue.placement = {"status": "rejected", "code": -2021, "msg": "raw exchange refusal"}
    _, stops = _manager(tmp_path / "b", venue)
    issues = stops.sync()
    assert issues == ["BTCUSDT: protective stop rejected: raw exchange refusal"]


def test_a_confirmed_fill_is_a_frozen_record_with_decimal_amounts(tmp_path):
    import threading

    from tests.unit.trading.test_execution_service import Venue
    from trading.application.execution.lifecycle import OrderLifecycle
    from trading.domain.orders import ConfirmedFill
    orders = OrderLifecycle(Venue(), SqliteOrderJournal(tmp_path / "orders.sqlite", SCOPE), SCOPE,
                            clock=lambda: 1000.0, sleeper=lambda s: None, lock=threading.RLock(), wait_seconds=0,
                            owner=OWNER, entry_order_type="market", not_found_grace_seconds=90)
    (fact,) = orders.submit(intent_id="i1", symbol="BTCUSDT", side="buy", qty=D("1"))
    assert fact == ConfirmedFill(fill_id=f"{SCOPE}:BTCUSDT:1:t1", intent_id="i1", symbol="BTCUSDT", order_id="1",
                                 trade_id="t1", side="buy", qty=D("1"), price=D("100"), fee="0", fee_asset="USDT")
    with pytest.raises(dataclasses.FrozenInstanceError):
        fact.qty = D("2")
    assert orders.journal.fill(fact.fill_id)["qty"] == "1"                  # the journal keeps strings
