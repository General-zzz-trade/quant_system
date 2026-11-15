"""Protective stops on the simulated exchange (decision L, ADR-0013, S10 part 2).

A stop is placed, looked up and canceled as on Binance (the contract suite holds both to the same rules). What is
simulated: a closed bar that reaches the trigger fills the whole position at the stop model's price with the
broker's taker fee, and the fill is recorded as an exchange order that the stop's ``actual_order_id`` names, so
the existing stop manager adopts it like a Binance stop's order.
"""
from __future__ import annotations

from decimal import Decimal as D

import pytest

from tests.unit.trading.test_simulated_exchange import Feed, exchange
from trading.domain.venue import STOP_ACTIVE, STOP_GONE, STOP_TRIGGERED, StopOrder, VenueOrder


@pytest.fixture
def world():
    venue, brokers = exchange("market")
    feed = Feed(venue)
    feed()                                           # the last close is 100
    feed(symbol="ETHUSDT")
    return venue, brokers, feed


def _hold(world, qty, symbol="BTCUSDT"):
    venue, brokers, feed = world
    side = "buy" if qty > 0 else "sell"
    venue.send_market_order(symbol, side, abs(qty), client_order_id=f"open-{symbol}-{qty}")
    feed(symbol=symbol)                              # fills at the open, 100, plus 3 bp slippage
    assert brokers[symbol].position.qty == D(str(qty))


def test_a_stop_is_placed_only_beyond_the_last_close_and_with_a_new_id(world):
    venue, _, _ = world
    assert venue.place_stop_market("BTCUSDT", "SELL", D("96"), client_algo_id="s1") == {"status": "submitted",
                                                                                        "algoId": "s1"}
    assert venue.get_algo_order(client_algo_id="s1") == StopOrder(
        algo_id="s1", client_algo_id="s1", symbol="BTCUSDT", side="sell", status=STOP_ACTIVE, state=STOP_ACTIVE,
        trigger_price=D("96"), actual_order_id="")
    assert venue.place_stop_market("BTCUSDT", "buy", D("104"), client_algo_id="s2")["status"] == "submitted"
    for side, trigger in (("sell", "100"), ("sell", "101"), ("buy", "100"), ("buy", "99")):
        refused = venue.place_stop_market("BTCUSDT", side, D(trigger), client_algo_id=f"b-{side}-{trigger}")
        assert (refused["status"], refused["breached"]) == ("rejected", True)
    for side, trigger, cid in (("hold", "96", "x1"), ("sell", "0", "x2"), ("sell", "-1", "x3"),
                               ("sell", "NaN", "x4"), ("sell", "96", ""), ("sell", "95", "s1")):
        refused = venue.place_stop_market("BTCUSDT", side, D(trigger), client_algo_id=cid)
        assert refused["status"] == "rejected" and refused.get("breached") is not True
    assert [s.client_algo_id for s in venue.get_open_algo_orders()] == ["s1", "s2"]
    assert venue.get_algo_order(client_algo_id="s1").trigger_price == D("96")       # the duplicate changed nothing
    assert venue.get_algo_order(client_algo_id="missing") is None
    with pytest.raises(RuntimeError):
        venue.place_stop_market("XRPUSDT", "sell", D("96"), client_algo_id="x5")


def test_a_stop_needs_a_price_to_be_placed():
    venue, _ = exchange("market")
    refused = venue.place_stop_market("BTCUSDT", "sell", D("96"), client_algo_id="s1")
    assert refused["status"] == "rejected" and refused.get("breached") is not True
    assert venue.get_open_algo_orders() == ()


def test_a_long_stop_reached_by_a_bars_low_closes_the_position_as_an_exchange_order(world):
    venue, brokers, feed = world
    _hold(world, 1)
    broker = brokers["BTCUSDT"]
    fees, fills = broker.fees, len(broker.fills)
    venue.place_stop_market("BTCUSDT", "sell", D("96"), client_algo_id="s1")
    bar = feed(o=99, h=99, lo=95, c=97)
    price = D("96") * D("0.999")
    fee = D("1.000") * price * 5 / 10000
    stop = venue.get_algo_order(client_algo_id="s1")
    assert stop.state == STOP_TRIGGERED and stop.status == STOP_TRIGGERED and stop.actual_order_id
    order = venue.get_order("BTCUSDT", order_id=stop.actual_order_id)
    assert order == VenueOrder(order_id=stop.actual_order_id, client_order_id="autoclose-s1", symbol="BTCUSDT",
                               side="sell", state="filled", filled_qty=D("1.000"), avg_price=price)
    assert venue.get_order("BTCUSDT", client_order_id="autoclose-s1") == order
    (fill,) = venue.get_recent_fills(symbol="BTCUSDT", order_id=stop.actual_order_id)
    assert (fill.side, fill.qty, fill.price, fill.fee, fill.liquidity, fill.ts_ms) == \
        ("sell", D("1.000"), price, fee, "taker", bar.available_at)
    assert fill in venue.get_recent_fills(symbol="BTCUSDT")
    assert len(broker.fills) == fills + 1 and broker.fills[-1]["order"] == "stop:s1"
    assert broker.position.qty == 0 and broker.fees == fees + fee
    assert broker.equity == D("10000") + broker.position.realized_pnl - broker.fees
    assert venue.get_positions() == () and venue.get_open_algo_orders() == () and venue.get_open_orders() == ()
    venue.cancel_order("BTCUSDT", stop.actual_order_id)                # nothing rests: it stays filled
    assert venue.get_order("BTCUSDT", order_id=stop.actual_order_id).state == "filled"


def test_a_short_stop_reached_by_a_bars_high_fills_at_an_open_that_gapped_through_it(world):
    venue, brokers, feed = world
    _hold(world, -1)
    venue.place_stop_market("BTCUSDT", "buy", D("104"), client_algo_id="s1")
    feed(o=103, h=D("103.99"), lo=102, c=103)                          # not reached
    assert venue.get_algo_order(client_algo_id="s1").state == STOP_ACTIVE
    feed(o=106, h=107, lo=105, c=106)
    order = venue.get_order("BTCUSDT", client_order_id="autoclose-s1")
    assert (order.side, order.filled_qty, order.avg_price) == ("buy", D("1.000"), D("106") * D("1.001"))
    assert brokers["BTCUSDT"].position.qty == 0


def test_a_stop_without_its_position_does_not_trigger_and_stays_active(world):
    venue, brokers, feed = world
    venue.place_stop_market("BTCUSDT", "sell", D("96"), client_algo_id="flat")
    feed(o=99, h=99, lo=90, c=92)                                      # no position yet
    _hold(world, -1)
    venue.place_stop_market("BTCUSDT", "sell", D("91"), client_algo_id="other-side")
    _hold(world, 1, "ETHUSDT")
    venue.place_stop_market("ETHUSDT", "sell", D("96"), client_algo_id="eth")
    feed(o=92, h=92, lo=80, c=85)                                      # a short is not closed by a sell stop
    assert {s.client_algo_id for s in venue.get_open_algo_orders()} == {"flat", "other-side", "eth"}
    assert brokers["BTCUSDT"].position.qty == D("-1.000") and brokers["ETHUSDT"].position.qty == D("1.000")
    assert not [f for b in brokers.values() for f in b.fills if str(f["order"]).startswith("stop:")]


def test_a_stop_is_triggered_only_by_its_own_symbols_bars(world):
    venue, brokers, feed = world
    _hold(world, 1)
    _hold(world, 1, "ETHUSDT")
    venue.place_stop_market("ETHUSDT", "sell", D("96"), client_algo_id="eth")
    feed(o=99, h=99, lo=90, c=92)                                      # BTCUSDT crashes; the ETHUSDT stop waits
    assert brokers["BTCUSDT"].position.qty == D("1.000") and brokers["ETHUSDT"].position.qty == D("1.000")
    assert venue.get_algo_order(client_algo_id="eth").state == STOP_ACTIVE
    feed(o=99, h=99, lo=90, c=92, symbol="ETHUSDT")
    assert brokers["ETHUSDT"].position.qty == 0 and venue.get_algo_order(client_algo_id="eth").state == STOP_TRIGGERED


@pytest.mark.parametrize("qty,side,triggers,bar", [(1, "sell", ("96", "97"), (99, 99, 90, 92)),
                                                   (-1, "buy", ("104", "103"), (101, 110, 101, 108))])
def test_two_stops_reached_by_one_bar_close_the_position_once(world, qty, side, triggers, bar):
    venue, brokers, feed = world
    _hold(world, qty)
    venue.place_stop_market("BTCUSDT", side, D(triggers[0]), client_algo_id="s1")
    venue.place_stop_market("BTCUSDT", side, D(triggers[1]), client_algo_id="s2")
    feed(*bar)
    states = sorted(venue.get_algo_order(client_algo_id=c).state for c in ("s1", "s2"))
    assert states == [STOP_ACTIVE, STOP_TRIGGERED] and brokers["BTCUSDT"].position.qty == 0
    assert len([f for f in brokers["BTCUSDT"].fills if str(f["order"]).startswith("stop:")]) == 1


def test_a_canceled_stop_is_gone_and_a_triggered_one_stays_triggered(world):
    venue, _, feed = world
    _hold(world, 1)
    venue.place_stop_market("BTCUSDT", "sell", D("96"), client_algo_id="s1")
    assert venue.cancel_algo_order(client_algo_id="s1") == {"status": "cancel_requested"}
    assert venue.get_algo_order(client_algo_id="s1").state == STOP_GONE and venue.get_open_algo_orders() == ()
    venue.cancel_algo_order(client_algo_id="s1")
    assert venue.get_algo_order(client_algo_id="s1").state == STOP_GONE
    feed(o=99, h=99, lo=90, c=92)                                      # a canceled stop does not fill
    assert venue.get_positions()[0].qty == D("1.000")
    venue.place_stop_market("BTCUSDT", "sell", D("88"), client_algo_id="s2")
    feed(o=92, h=92, lo=80, c=85)
    venue.cancel_algo_order(client_algo_id="s2")
    assert venue.get_algo_order(client_algo_id="s2").state == STOP_TRIGGERED
    with pytest.raises(RuntimeError):
        venue.cancel_algo_order(client_algo_id="missing")
