"""The simulated exchange (ADR-0013 decision 1, design 2026-11 M5 section 4.2, S10 part 2).

What only the simulated exchange does: the bar stop model of decision L, matching on closed bars exactly as the
research sleeve does, the lot step and the maker price research uses, and its saved state. What it shares with
the Binance adapter is fixed by the port-contract suite (tests/contract/test_exchange_ports.py).
"""
from __future__ import annotations

from decimal import Decimal as D
from types import SimpleNamespace as NS

import pytest

from lab.simulate import Sleeve
from trading.adapters.simulated.bar_broker import SYMBOL_RULES, BarBroker, ReplayConfig, broker_config
from trading.adapters.simulated.exchange import BarStopModel, SimFilters, SimulatedExchange
from trading.application import ports
from trading.domain.bars import HOUR, Bar
from trading.domain.orders import LIMIT_OFFSET_BPS, OrderIntent
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


def test_the_exchange_refuses_brokers_and_filters_that_do_not_belong_together():
    good = SimFilters(D("0.001"), D("0.001"), D("5"))
    broker = BarBroker(ReplayConfig(qty_step="0.001", min_notional="5"))
    SimulatedExchange({"BTCUSDT": broker}, filters={"BTCUSDT": good}, stop_model=BarStopModel())
    cases = [({}, {}), ({"BTCUSDT": broker}, {"ETHUSDT": good}),
             ({"BTCUSDT": broker}, {"BTCUSDT": SimFilters(D("0.01"), D("0.001"), D("5"))}),
             ({"BTCUSDT": broker}, {"BTCUSDT": SimFilters(D("0.001"), D("0.001"), D("20"))}),
             ({"BTCUSDT": BarBroker(ReplayConfig(qty_step="0.001", min_notional="5", isolated_leverage="3",
                                                 max_exposure="1"))}, {"BTCUSDT": good})]
    for brokers, filters in cases:
        with pytest.raises(ValueError):
            SimulatedExchange(brokers, filters=filters, stop_model=BarStopModel())


# (bar, orders sent after it as (side, qty, reduce_only), funding rate): an entry, a partial reduction, a flip, an
# entry capped by the bar's volume that the broker expires, and funding on a long, a zero rate and on a short.
SCRIPT = [((2000, 2010, 1990, 2000, "1000"), [("buy", "1.250", False)], None),
          ((2001, 2012, 1995, 2005, "1000"), [], None),
          ((2006, 2015, 2000, 2010, "1000"), [("sell", "0.500", True)], D("0.0001")),
          ((2009, 2011, 1990, 1995, "1000"), [], D("0")),
          ((1996, 2000, 1980, 1985, "1000"), [("sell", "0.750", True), ("sell", "1.000", False)], None),
          ((1984, 1990, 1975, 1980, "1000"), [], None),
          ((1981, 1983, 1970, 1975, "1000"), [("buy", "1.000", True), ("buy", "2.000", False)], D("-0.0002")),
          ((1976, 1980, 1960, 1977, "50"), [], None),
          ((1976, 1979, 1961, 1970, "50"), [], None),
          ((1969, 1975, 1955, 1972, "50"), [], None),
          ((1971, 1980, 1965, 1978, "50"), [], None)]


@pytest.mark.parametrize("entry_mode", ["market", "maker"])
def test_the_exchange_matches_exactly_as_the_research_sleeve(entry_mode):
    """The same bars, funding and orders give the research sleeve's fills to the last digit of every string, and
    its equity. The lifecycle sends quantities and prices as floats; research sends Decimals on the lot step."""
    rules = SYMBOL_RULES["ETHUSDT"]
    config = broker_config(NS(ttl_bars=3, execution=entry_mode), rules)
    sleeve, broker = Sleeve("ETHUSDT", None, rules, config), BarBroker(config)
    venue = SimulatedExchange({"ETHUSDT": broker}, stop_model=BarStopModel(),
                              filters={"ETHUSDT": SimFilters(rules.step, rules.min_qty, rules.min_notional)})
    for k, ((o, h, lo, c, volume), orders, funding) in enumerate(SCRIPT):
        bar = candle(o, h, lo, c, at=START + k * HOUR, volume=volume)
        sleeve.on_bar(bar, funding)
        venue.on_closed_bar("ETHUSDT", bar, funding)
        for j, (side, qty, reduce_only) in enumerate(orders):
            oid = f"ETHUSDT:{k}:{j}"
            sleeve.broker.send_order(OrderIntent(order_id=oid, source="research", symbol="ETHUSDT", side=side,
                                                 qty=D(qty), price=bar.close, reduce_only=reduce_only,
                                                 reason="rebalance"))
            if reduce_only or entry_mode == "market":
                placed = venue.send_market_order("ETHUSDT", side, float(qty), reduce_only=reduce_only,
                                                 client_order_id=oid)
            else:
                placed = venue.send_limit_order("ETHUSDT", side, float(qty), float(maker_price(side, bar.close)),
                                                client_order_id=oid)
            assert placed["status"] == "submitted", placed
    key = lambda f: (f["at"], f["side"], f["qty"], f["price"], f["fee"])  # noqa: E731
    simulated = venue.get_recent_fills(symbol="ETHUSDT", limit=1000)
    assert [(f.ts_ms, f.side, str(f.qty), str(f.price), str(f.fee)) for f in simulated] == \
        [key(f) for f in sleeve.broker.fills]
    assert len(simulated) >= 7 and any(f["qty"] == "0.500" for f in sleeve.broker.fills)
    assert venue.get_symbol_equity("ETHUSDT") == sleeve.broker.equity
    assert broker.funding_log == sleeve.broker.funding_log and len(broker.funding_log) == 3   # a zero rate too
    assert sleeve.broker.funding != 0


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
    assert sorted((o.side, o.type) for o in venue.get_open_orders("BTCUSDT")) == \
        [("buy", "LIMIT")] * 2 + [("sell", "LIMIT")] * 2


def test_nothing_is_taken_before_the_first_price():
    venue, _ = exchange("maker")
    placed = venue.send_limit_order("BTCUSDT", "buy", 0.1, 0.0, client_order_id="early")
    assert placed["status"] == "rejected" and "orderId" not in placed and venue.get_open_orders() == ()


def test_the_lot_minimum_is_the_filters_minimum():
    """A quantity under the filters' minimum is refused before the broker sees it; with a minimum of zero, a
    quantity that floors to zero is still refused (the broker would read it as "close all")."""
    for min_qty, qty, status in (("0.01", 0.009, "rejected"), ("0.01", 0.01, "submitted"), ("0", 0.0004, "rejected")):
        broker = BarBroker(ReplayConfig(qty_step="0.001", min_notional="5", entry_mode="market"))
        venue = SimulatedExchange({"BTCUSDT": broker}, filters={"BTCUSDT": SimFilters(D("0.001"), D(min_qty), D("5"))},
                                  stop_model=BarStopModel())
        venue.on_closed_bar("BTCUSDT", candle(1000, 1010, 990, 1000), None)
        placed = venue.send_market_order("BTCUSDT", "buy", qty, client_order_id="c")
        assert placed["status"] == status and len(broker.orders) == (status == "submitted")


def test_the_exchange_takes_only_the_entries_its_brokers_simulate(market, maker):
    venue, _, _ = market
    assert venue.send_limit_order("BTCUSDT", "buy", 0.1, float(maker_price("buy")),
                                  client_order_id="x")["status"] == "rejected"
    venue, _, _ = maker
    assert venue.send_market_order("BTCUSDT", "buy", 0.1, client_order_id="m")["status"] == "rejected"
    refused = [venue.send_limit_order("BTCUSDT", "buy", 0.1, float(maker_price("buy")), post_only=False,
                                      client_order_id="gtc"),
               venue.send_limit_order("BTCUSDT", "sell", 0.1, float(maker_price("sell")), reduce_only=True,
                                      client_order_id="ro"),
               venue.send_limit_order("BTCUSDT", "hold", 0.1, float(maker_price("sell")), client_order_id="side"),
               venue.send_limit_order("BTCUSDT", "buy", 0.1, float(maker_price("buy")), client_order_id="")]
    assert [p["status"] for p in refused] == ["rejected"] * 4
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


def test_a_maker_entry_capped_by_the_bars_volume_fills_in_parts_and_the_broker_expires_the_rest(maker):
    venue, _, feed = maker
    placed = venue.send_limit_order("BTCUSDT", "buy", 2.0, float(maker_price("buy")), client_order_id="big")
    feed(volume="50")                                                  # 1 % of 50: 0.5 per bar
    order = venue.get_order("BTCUSDT", order_id=placed["orderId"])
    assert (order.state, order.filled_qty, order.avg_price) == ("partially_filled", D("0.500"), maker_price("buy"))
    assert [o.order_id for o in venue.get_open_orders("BTCUSDT")] == [placed["orderId"]]
    assert venue.get_recent_fills(symbol="BTCUSDT")[0].liquidity == "maker"
    feed(volume="50")
    feed(volume="50")                                                  # ttl_bars = 3
    order = venue.get_order("BTCUSDT", order_id=placed["orderId"])
    assert (order.state, order.filled_qty) == ("canceled", D("1.500")) and venue.get_open_orders() == ()


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


def test_orders_are_found_by_either_id_and_only_under_their_symbol(market):
    venue, _, feed = market
    placed = venue.send_market_order("BTCUSDT", "sell", 0.1, client_order_id="c1")
    feed()
    assert venue.get_recent_fills(symbol="BTCUSDT", order_id=placed["orderId"])        # filled, under BTCUSDT
    by_id = venue.get_order("BTCUSDT", order_id=placed["orderId"])
    assert by_id == venue.get_order("BTCUSDT", client_order_id="c1") == \
        venue.get_order("BTCUSDT", order_id=placed["orderId"], client_order_id="c1")
    assert (by_id.order_id, by_id.client_order_id, by_id.symbol, by_id.side) == (placed["orderId"], "c1",
                                                                                  "BTCUSDT", "sell")
    assert venue.get_order("ETHUSDT", order_id=placed["orderId"]) is None
    assert venue.get_order("BTCUSDT", order_id="BTCUSDT-999", client_order_id="c1") is None
    assert venue.get_recent_fills(symbol="ETHUSDT", order_id=placed["orderId"]) == ()
    assert venue.get_recent_fills(symbol="BTCUSDT", order_id="BTCUSDT-999") == ()
    assert venue.send_market_order("BTCUSDT", "sell", 0.1, client_order_id="c1")["status"] == "rejected"
    assert [venue.send_market_order("BTCUSDT", side, 0.1, client_order_id=cid)["status"]
            for side, cid in (("hold", "c3"), ("buy", ""))] == ["rejected", "rejected"]
    assert venue.send_market_order("BTCUSDT", "BUY", 0.1, client_order_id="c4")["status"] == "submitted"
    assert venue.get_order("BTCUSDT", client_order_id="c4").side == "buy"


def test_trade_history_without_an_order_is_every_fill_of_the_symbol_in_order(market):
    venue, _, feed = market
    for n in range(3):
        venue.send_market_order("BTCUSDT", "buy", 0.1, client_order_id=f"b{n}")
        venue.send_market_order("ETHUSDT", "buy", 0.1, client_order_id=f"e{n}")
        feed()
        feed(symbol="ETHUSDT")
    fills = venue.get_recent_fills(symbol="BTCUSDT")
    assert [f.order_id for f in fills] == [venue.get_order("BTCUSDT", client_order_id=f"b{n}").order_id
                                           for n in range(3)]
    assert [f.ts_ms for f in fills] == [START + HOUR, START + 2 * HOUR, START + 3 * HOUR]
    assert venue.get_recent_fills(symbol="BTCUSDT", since_ms=START + 2 * HOUR) == fills[1:]
    assert venue.get_recent_fills(symbol="BTCUSDT", limit=2) == fills[:2]
    with pytest.raises(ValueError):
        venue.get_recent_fills(order_id=fills[0].order_id)


def test_each_symbol_is_its_own_simulated_account(market):
    venue, brokers, feed = market
    venue.send_market_order("BTCUSDT", "buy", 1.0, client_order_id="b")
    feed(o=100, h=110, lo=100, c=110)
    btc, eth = venue.get_symbol_equity("BTCUSDT"), venue.get_symbol_equity("ETHUSDT")
    assert btc == brokers["BTCUSDT"].equity and eth == D("10000") and btc > eth
    assert venue.get_account_equity() == float(btc + eth)
    assert venue.get_ticker("BTCUSDT") == {"symbol": "BTCUSDT", "lastPrice": 110.0, "bid1Price": 110.0,
                                           "ask1Price": 110.0}
    assert venue.execution_blockers() == [] and venue.get_symbol_settings("ETHUSDT") == (1, "cross")


def test_a_bar_is_matched_once_and_only_for_a_simulated_symbol(market):
    venue, brokers, feed = market
    venue.send_market_order("BTCUSDT", "buy", 0.1, client_order_id="b")
    bar = feed()
    for again in (bar, candle(100, 101, 99, 100, at=bar.open_time - HOUR)):
        with pytest.raises(ValueError):
            venue.on_closed_bar("BTCUSDT", again, None)
    assert len(brokers["BTCUSDT"].fills) == 1
    feed(funding=D("0.001"))
    assert brokers["BTCUSDT"].funding_log[-1]["cashflow"] == str(-D("0.100") * D("100") * D("0.001"))
    for call in (lambda: venue.on_closed_bar("XRPUSDT", bar, None), lambda: venue.filters("XRPUSDT"),
                 lambda: venue.get_positions("XRPUSDT"), lambda: venue.get_ticker("XRPUSDT"),
                 lambda: venue.send_market_order("XRPUSDT", "buy", 1.0, client_order_id="x"),
                 lambda: venue.get_recent_fills(symbol="XRPUSDT"), lambda: venue.get_symbol_settings("XRPUSDT")):
        with pytest.raises(RuntimeError):
            call()
    assert venue.filters("ETHUSDT") == SimFilters(D("0.001"), D("0.001"), D("5"))
