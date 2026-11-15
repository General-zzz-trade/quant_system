"""The simulated exchange (ADR-0013 decision 1, design 2026-11 M5 section 4.2, S10 part 2).

What only the simulated exchange does: the bar stop model of decision L, matching on closed bars exactly as the
research sleeve does, the lot step and the maker price research uses, and its saved state. What it shares with
the Binance adapter is fixed by the port-contract suite (tests/contract/test_exchange_ports.py).
"""
from __future__ import annotations

from decimal import Decimal as D

import pytest

from trading.adapters.simulated.bar_broker import BarBroker, ReplayConfig
from trading.adapters.simulated.exchange import BarStopModel, SimFilters, SimulatedExchange
from trading.application import ports
from trading.domain.bars import HOUR, Bar
from trading.domain.orders import LIMIT_OFFSET_BPS
from trading.domain.venue import OpenOrder

START = 1_800_000_000_000 // HOUR * HOUR


def candle(o, h, lo, c, *, at=START, volume="1000"):
    return Bar(at, D(str(o)), D(str(h)), D(str(lo)), D(str(c)), D(volume))


# -- the bar stop model (decision L) --------------------------------------------------------------------------
def test_a_sell_stop_fills_at_the_trigger_when_the_low_reaches_it_minus_the_stop_slippage():
    model = BarStopModel(slippage_bps=D("10"))
    assert model.triggered("sell", D("90"), candle(100, 101, D("90.01"), 95)) is None
    assert model.triggered("sell", D("90"), candle(100, 101, 90, 95)) == D("90") * D("0.999")      # touched
    assert model.triggered("sell", D("90"), candle(100, 101, 89, 95)) == D("90") * D("0.999")


def test_a_buy_stop_fills_at_the_trigger_when_the_high_reaches_it_plus_the_stop_slippage():
    model = BarStopModel(slippage_bps=D("10"))
    assert model.triggered("buy", D("110"), candle(100, D("109.99"), 99, 105)) is None
    assert model.triggered("buy", D("110"), candle(100, 110, 99, 105)) == D("110") * D("1.001")
    assert model.triggered("buy", D("110"), candle(100, 111, 99, 105)) == D("110") * D("1.001")


def test_an_open_that_gapped_through_the_stop_fills_at_the_open():
    model = BarStopModel(slippage_bps=D("10"))
    assert model.triggered("sell", D("90"), candle(85, 86, 80, 84)) == D("85") * D("0.999")
    assert model.triggered("buy", D("110"), candle(115, 116, 112, 113)) == D("115") * D("1.001")


def test_a_stop_is_not_triggered_by_the_other_side_of_the_bar_or_an_unknown_side():
    model = BarStopModel(slippage_bps=D("0"))
    crash, spike = candle(100, 100, 80, 85), candle(100, 120, 100, 115)
    assert model.triggered("buy", D("110"), crash) is None and model.triggered("sell", D("90"), spike) is None
    assert model.triggered("BUY", D("110"), spike) is None and model.triggered("", D("90"), crash) is None
    assert model.triggered("sell", D("90"), crash) == D("90")                 # no slippage: the trigger itself


def test_the_default_stop_slippage_is_the_research_brokers_stop_slippage():
    assert BarStopModel().slippage_bps == D(ReplayConfig().stop_slippage_bps) == D("10")
    for bad in (D("-1"), D("10000"), D("NaN")):
        with pytest.raises(ValueError):
            BarStopModel(slippage_bps=bad)


def test_the_bar_listener_port_takes_a_closed_bar_and_its_funding_rate():
    class Listener:
        def on_closed_bar(self, symbol, bar, funding_rate):
            return None

    assert isinstance(Listener(), ports.BarListener)
    assert "BarListener" in ports.__all__


# -- orders and account over the research brokers -------------------------------------------------------------
SYMBOLS = ("BTCUSDT", "ETHUSDT")


def exchange(entry_mode="market", *, symbols=SYMBOLS, **settings):
    values = {"starting_balance": "10000", "qty_step": "0.001", "min_notional": "5", "entry_mode": entry_mode,
              **settings}
    brokers = {s: BarBroker(ReplayConfig(**values)) for s in symbols}
    filters = {s: SimFilters(D(values["qty_step"]), D("0.001"), D(values["min_notional"])) for s in symbols}
    return SimulatedExchange(brokers, filters=filters, stop_model=BarStopModel()), brokers


class Feed:
    """Closed bars for one exchange, one hour apart per symbol."""

    def __init__(self, venue):
        self.venue, self.at = venue, dict.fromkeys(SYMBOLS, START)

    def __call__(self, o=100, h=101, lo=99, c=100, *, symbol="BTCUSDT", volume="1000", funding=None):
        bar = candle(o, h, lo, c, at=self.at[symbol], volume=volume)
        self.venue.on_closed_bar(symbol, bar, funding)
        self.at[symbol] += HOUR
        return bar


def _opened(entry_mode):
    venue, brokers = exchange(entry_mode)
    feed = Feed(venue)
    feed()                                   # the last close, 100, is the reference of the next order
    feed(symbol="ETHUSDT")
    return venue, brokers, feed


@pytest.fixture
def market():
    return _opened("market")


@pytest.fixture
def maker():
    return _opened("maker")


def maker_price(side, close="100"):
    sign = 1 if side == "buy" else -1
    return D(close) * (1 - sign * LIMIT_OFFSET_BPS / 10000)


def test_a_float_quantity_goes_back_on_the_lot_step_with_the_steps_exponent(market):
    venue, brokers, feed = market
    venue.send_market_order("BTCUSDT", "buy", 0.15, client_order_id="a")
    venue.send_market_order("BTCUSDT", "buy", 0.1239, client_order_id="b")
    assert [row["qty"] for row in brokers["BTCUSDT"].orders] == ["0.150", "0.123"]
    feed()
    assert [str(f.qty) for f in venue.get_recent_fills(symbol="BTCUSDT")] == ["0.150", "0.123"]


def test_a_quantity_below_the_lot_step_is_refused_and_never_closes_the_whole_position(market):
    """The broker reads a reduction of quantity zero as "close all"; a quantity that floors to zero must not
    reach it."""
    venue, brokers, feed = market
    venue.send_market_order("BTCUSDT", "buy", 1.0, client_order_id="entry")
    feed()
    for n, qty in enumerate((0.0004, 0.0, -1.0, float("nan"), float("inf"))):
        placed = venue.send_market_order("BTCUSDT", "sell", qty, reduce_only=True, client_order_id=f"r{n}")
        assert placed["status"] == "rejected" and "orderId" not in placed
    feed()
    assert brokers["BTCUSDT"].position.qty == D("1.000") and len(brokers["BTCUSDT"].orders) == 1


def test_a_maker_entry_is_accepted_only_at_the_research_maker_price(maker):
    venue, brokers, feed = maker
    for n, side in enumerate(("buy", "sell")):
        exact = maker_price(side)
        for factor in (D("1.0000000009"), D("0.9999999991")):        # within 1e-9 relative
            placed = venue.send_limit_order("BTCUSDT", side, 0.1, float(exact * factor),
                                            client_order_id=f"in{n}{factor}")
            assert placed["status"] == "submitted", placed
        for factor in (D("1.0000000011"), D("0.9999999989"), D("1.0002")):
            placed = venue.send_limit_order("BTCUSDT", side, 0.1, float(exact * factor),
                                            client_order_id=f"out{n}{factor}")
            assert placed["status"] == "rejected" and "research maker price" in placed["msg"]
    for price in (float("nan"), float("inf"), 0.0):
        assert venue.send_limit_order("BTCUSDT", "buy", 0.1, price, client_order_id=f"p{price}")["status"] == \
            "rejected"
    rows = brokers["BTCUSDT"].pending
    assert {row["limit"] for row in rows} == {str(maker_price("buy")), str(maker_price("sell"))}
    assert {row["reference"] for row in rows} == {"100"}


def test_the_exchange_takes_only_the_entries_its_brokers_simulate(market, maker):
    venue, _, _ = market
    assert venue.send_limit_order("BTCUSDT", "buy", 0.1, float(maker_price("buy")),
                                  client_order_id="x")["status"] == "rejected"
    venue, _, _ = maker
    assert venue.send_market_order("BTCUSDT", "buy", 0.1, client_order_id="m")["status"] == "rejected"
    refused = [venue.send_limit_order("BTCUSDT", "buy", 0.1, float(maker_price("buy")), post_only=False,
                                      client_order_id="gtc"),
               venue.send_limit_order("BTCUSDT", "sell", 0.1, float(maker_price("sell")), reduce_only=True,
                                      client_order_id="ro")]
    assert [p["status"] for p in refused] == ["rejected", "rejected"]
    assert venue.get_open_orders() == ()
    placed = venue.send_market_order("BTCUSDT", "sell", 0.1, reduce_only=True, client_order_id="exit")
    assert placed["status"] == "rejected" and placed["orderId"]          # the broker refused it: nothing to reduce
    assert venue.get_order("BTCUSDT", order_id=placed["orderId"]).state == "rejected"


def test_a_market_order_rests_until_the_next_closed_bar_and_fills_at_its_open(market):
    venue, brokers, feed = market
    placed = venue.send_market_order("BTCUSDT", "buy", 0.1, client_order_id="c1")
    order = venue.get_order("BTCUSDT", client_order_id="c1")
    assert (order.state, order.filled_qty, order.avg_price) == ("new", 0, 0)
    assert venue.get_open_orders("BTCUSDT") == (OpenOrder(order_id=placed["orderId"], client_order_id="c1",
                                                          symbol="BTCUSDT", side="buy", type="MARKET"),)
    assert venue.get_recent_fills(symbol="BTCUSDT", order_id=placed["orderId"]) == ()
    feed(o=102, h=103, lo=101, c=102)
    order = venue.get_order("BTCUSDT", order_id=placed["orderId"])
    price = D("102") * D("1.0003")
    assert (order.state, order.filled_qty, order.avg_price) == ("filled", D("0.100"), price)
    (fill,) = venue.get_recent_fills(symbol="BTCUSDT", order_id=placed["orderId"])
    assert (fill.venue, fill.order_id, fill.side, fill.qty, fill.price, fill.liquidity, fill.ts_ms, fill.fee_asset) \
        == ("simulated", placed["orderId"], "buy", D("0.100"), price, "taker", START + HOUR, "USDT")
    assert fill.fee == D("0.100") * price * 5 / 10000 and fill.trade_id == fill.fill_id != ""
    assert venue.get_open_orders() == ()
    (position,) = venue.get_positions()
    assert (position.venue, position.symbol, position.qty, position.entry_price, position.mark_price) == \
        ("simulated", "BTCUSDT", D("0.100"), price, D("102"))
    assert venue.get_positions("ETHUSDT") == ()


def test_a_canceled_order_never_fills_and_a_cancel_needs_a_known_order(maker):
    venue, brokers, feed = maker
    placed = venue.send_limit_order("BTCUSDT", "buy", 0.1, float(maker_price("buy")), client_order_id="c1")
    assert venue.cancel_order("BTCUSDT", placed["orderId"]) == {"status": "cancel_requested"}
    feed(lo=90)
    assert venue.get_order("BTCUSDT", order_id=placed["orderId"]).state == "canceled"
    assert brokers["BTCUSDT"].fills == [] and venue.get_open_orders() == ()
    filled = venue.send_limit_order("BTCUSDT", "buy", 0.1, float(maker_price("buy")), client_order_id="c2")
    feed(lo=90)
    venue.cancel_order("BTCUSDT", filled["orderId"])                   # too late: it stays filled
    assert venue.get_order("BTCUSDT", order_id=filled["orderId"]).state == "filled"
    for symbol, order_id in (("BTCUSDT", "BTCUSDT-999"), ("ETHUSDT", placed["orderId"])):
        with pytest.raises(RuntimeError):
            venue.cancel_order(symbol, order_id)


