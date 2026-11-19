"""One exchange-port contract, run on both exchanges execution mode can talk to (ADR-0013, S10 part 2).

Every test runs on the simulated exchange (trading/adapters/simulated/exchange.py over research bar brokers)
and on the Binance adapter over the FakeBinance test double, built as the other Binance tests build it. A venue
harness gives both the same two verbs: ``match()`` lets the venue work its resting orders once (the simulated
exchange matches one ordinary closed bar; on FakeBinance each resting limit order fills, a market order there
has filled already), and ``crash(symbol)`` takes the price through every protective stop of the symbol.

Where the two legitimately differ the contract is written so that both pass. The differences:

1. Fill timing. Binance fills a market order when it is placed. The simulated exchange fills orders only in
   ``on_closed_bar``: a market order at the next bar's open plus slippage, a maker entry when the bar trades
   through its limit. Until then a market order is in state ``new`` and is listed among the open orders (type
   ``MARKET``). The contract reads outcomes after ``match()``.
2. Maker price. The simulated exchange takes a maker entry only post-only and only at the research maker price
   (the last close -/+ LIMIT_OFFSET_BPS, within 1e-9 relative); Binance takes any post-only price that does not
   cross, floored to the tick. The contract places maker entries at that price.
3. Entry type. The simulated exchange takes only the entries its brokers simulate (market entries in
   entry_mode market, maker entries in entry_mode maker); Binance takes both. The harness builds the simulated
   brokers in the entry mode a test uses.
4. A refusal after acceptance. The research broker's own refusal (below its minimum, e.g. a reduction without a
   position) is a recorded order in state ``rejected``, with an order id; a Binance refusal leaves no order.
   The contract reads only the placement status.
5. Stops. Binance triggers a stop on the mark price and fills it at the market; the simulated exchange triggers
   it on a closed bar's low or high and fills it at the trigger, or the open that gapped through it, worse by
   10 bp, with the taker fee (decision L). Binance floors the trigger to the tick; the simulated exchange keeps
   it. The contract reads what the stop manager reads: the stop's state, the order it became and its trades.
6. Ids. The simulated algoId is the client algo id and order ids are "<symbol>-<n>"; Binance's are numbers.
   Nothing reads them beyond equality.
7. Only one exchange covered (not in the contract): the simulated exchange refuses a second stop with the same
   client algo id (FakeBinance overwrites it; Binance refuses it); a stop whose position is flat or on the other
   side does not trigger there and stays active; it reports (1, "cross") and no blockers whatever happens;
   each symbol is its own account (``get_symbol_equity``; the account equity is their sum); the research broker
   expires a maker entry after ``ttl_bars`` bars (state ``canceled``), Binance only when the lifecycle cancels
   it; bid, ask and last are the last close; trade history without an order id is the symbol's trades from
   ``since_ms`` on, oldest first, at most ``limit`` (the lifecycle always asks by order id); and every read of a
   symbol it does not simulate raises (FakeBinance answers position reads for any symbol).
"""
from __future__ import annotations

import dataclasses
import threading
from decimal import Decimal as D

import pytest

from tests.contract.test_ports import missing_members
from tests.support.state_db import state_db
from tests.unit.binance.fake_venue import FakeBinance, make_bars
from trading.adapters.binance.adapter import BinanceAdapter
from trading.adapters.binance.config import BinanceConfig
from trading.adapters.simulated.bar_broker import BarBroker, ReplayConfig
from trading.adapters.simulated.exchange import BarStopModel, SimFilters, SimulatedExchange
from trading.adapters.sqlite.order_journal import SqliteOrderJournal
from trading.application import ports
from trading.application.execution.executor import OwnerExecutor
from trading.application.execution.fills import TargetFillConsumer
from trading.application.execution.lifecycle import OrderLifecycle
from trading.application.execution.service import ExecutionService
from trading.application.execution.stops import ProtectiveStopManager
from trading.domain.bars import HOUR, Bar
from trading.domain.orders import LIMIT_OFFSET_BPS, OrderIntent
from trading.domain.venue import (STOP_ACTIVE, STOP_GONE, STOP_TRIGGERED, CanonicalFill, OpenOrder, StopOrder,
                                  VenueOrder, VenuePosition)

SYMBOLS = ("BTCUSDT", "ETHUSDT")
SCOPE, OWNER = "contract-ut", "target.BTCUSDT"
# The price both venues start at. At 1000 the maker offset (0.2) is larger than FakeBinance's tick (0.1), so a
# maker sell floored to the tick still rests above the price, as with real BTC and ETH prices.
PRICE = D(1000)


class Simulated:
    """The simulated exchange over research brokers; every symbol's last close is PRICE."""

    def __init__(self, entry_mode: str) -> None:
        config = ReplayConfig(starting_balance="10000", qty_step="0.001", min_notional="5", entry_mode=entry_mode)
        self.venue = SimulatedExchange({s: BarBroker(config) for s in SYMBOLS}, stop_model=BarStopModel(),
                                       filters=dict.fromkeys(SYMBOLS, SimFilters(D("0.001"), D("0.001"), D("5"))))
        self.at = dict.fromkeys(SYMBOLS, 1_800_000_000_000 // HOUR * HOUR)
        self.match()

    def _bar(self, symbol, o, h, lo, c):
        self.venue.on_closed_bar(symbol, Bar(self.at[symbol], o, h, lo, c, D(1000)), None)
        self.at[symbol] += HOUR

    def match(self):
        for symbol in SYMBOLS:
            self._bar(symbol, PRICE, PRICE * D("1.01"), PRICE * D("0.99"), PRICE)

    def crash(self, symbol):
        self._bar(symbol, PRICE * D("0.99"), PRICE * D("0.99"), PRICE / 2, PRICE * D("0.6"))


class Binance:
    """The Binance adapter over FakeBinance, connected; every symbol's mark is PRICE."""

    def __init__(self, entry_mode: str) -> None:
        self.fake = FakeBinance(make_bars(10), clock=lambda: 1_800_000_000.0)
        for symbol in SYMBOLS:
            self.fake.price[symbol] = PRICE
        self.venue = BinanceAdapter(BinanceConfig(api_key="k", api_secret="s"), client=self.fake)
        assert self.venue.connect(symbols=SYMBOLS)

    def match(self):
        for order_id, order in list(self.fake.orders.items()):
            if order["type"] == "LIMIT" and order["status"] in ("NEW", "PARTIALLY_FILLED"):
                self.fake.fill_limit(order_id)

    def crash(self, symbol):
        for client_algo_id, algo in list(self.fake.algo.items()):
            if algo["symbol"] == symbol and algo["algoStatus"] == "NEW":
                self.fake.trigger_stop(client_algo_id, D(algo["triggerPrice"]) * D("0.999"))
        self.fake.price[symbol] = PRICE * D("0.6")


@pytest.fixture(params=["simulated", "binance"])
def make(request):
    return lambda entry_mode="market": (Simulated if request.param == "simulated" else Binance)(entry_mode)


def maker_price(side):
    return PRICE * (1 - (1 if side == "buy" else -1) * LIMIT_OFFSET_BPS / 10000)


def _long(h, qty=0.1, symbol="BTCUSDT"):
    placed = h.venue.send_market_order(symbol, "buy", qty, client_order_id=f"open-{symbol}")
    h.match()
    return placed


# -- ports ----------------------------------------------------------------------------------------------------
def test_both_exchanges_implement_every_port_execution_mode_uses(make):
    venue = make().venue
    for protocol in (ports.ExchangeAccount, ports.OrderGateway, ports.StopOrderGateway):
        assert missing_members(venue, protocol) == [] and isinstance(venue, protocol)
    assert missing_members(venue, ports.AccountSnapshot) == []


def test_only_the_simulated_exchange_listens_to_closed_bars(make):
    venue = make().venue
    assert isinstance(venue, ports.BarListener) is isinstance(venue, SimulatedExchange)


# -- orders -----------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("side,sign", [("buy", 1), ("sell", -1)])
def test_a_market_entry_fills_and_its_order_trades_and_position_can_be_read(make, side, sign):
    h = make("market")
    placed = h.venue.send_market_order("BTCUSDT", side, 0.1, client_order_id="c1")
    assert placed["status"] == "submitted" and placed["orderId"] and placed["clientOrderId"] == "c1"
    h.match()
    order = h.venue.get_order("BTCUSDT", order_id=placed["orderId"])
    assert isinstance(order, VenueOrder) and order == h.venue.get_order("BTCUSDT", client_order_id="c1")
    assert (order.order_id, order.client_order_id, order.symbol, order.side, order.state, order.filled_qty) == \
        (placed["orderId"], "c1", "BTCUSDT", side, "filled", D("0.1"))
    trades = h.venue.get_recent_fills(symbol="BTCUSDT", order_id=placed["orderId"], limit=100)
    assert trades and all(isinstance(t, CanonicalFill) for t in trades)
    assert sum(t.qty for t in trades) == D("0.1")
    for t in trades:
        assert (t.symbol, t.order_id, t.side, t.liquidity, t.fee_asset) == ("BTCUSDT", placed["orderId"], side,
                                                                            "taker", "USDT")
        assert t.trade_id and t.price > 0 and t.fee >= 0
    assert set(trades) <= set(h.venue.get_recent_fills(symbol="BTCUSDT"))
    (position,) = h.venue.get_positions("BTCUSDT", strict=True)
    assert isinstance(position, VenuePosition) and (position.symbol, position.qty) == ("BTCUSDT", sign * D("0.1"))
    assert position.entry_price > 0 and position.mark_price > 0
    (again,) = h.venue.get_positions(strict=True)                    # ts_ms is stamped per read: not compared
    assert dataclasses.replace(again, ts_ms=0) == dataclasses.replace(position, ts_ms=0)
    assert h.venue.get_positions("ETHUSDT") == ()
    assert h.venue.get_open_orders("BTCUSDT", strict=True) == ()
    with pytest.raises(dataclasses.FrozenInstanceError):
        order.state = "new"


def test_a_reduce_only_order_closes_the_position_and_is_refused_without_one(make):
    h = make("market")
    _long(h)
    placed = h.venue.send_market_order("BTCUSDT", "sell", 0.1, reduce_only=True, client_order_id="exit")
    assert placed["status"] == "submitted"
    h.match()
    order = h.venue.get_order("BTCUSDT", client_order_id="exit")
    assert (order.state, order.side, order.filled_qty) == ("filled", "sell", D("0.1"))
    assert h.venue.get_positions("BTCUSDT") == ()
    refused = h.venue.send_market_order("BTCUSDT", "sell", 0.1, reduce_only=True, client_order_id="again")
    assert refused["status"] == "rejected"


def test_a_post_only_maker_entry_rests_until_canceled(make):
    h = make("maker")
    placed = h.venue.send_limit_order("BTCUSDT", "buy", 0.1, float(maker_price("buy")), post_only=True,
                                      client_order_id="m1")
    assert placed["status"] == "submitted"
    assert h.venue.get_open_orders("BTCUSDT") == (OpenOrder(order_id=placed["orderId"], client_order_id="m1",
                                                            symbol="BTCUSDT", side="buy", type="LIMIT"),)
    assert {o.order_id for o in h.venue.get_open_orders(strict=True)} == {placed["orderId"]}
    order = h.venue.get_order("BTCUSDT", order_id=placed["orderId"])
    assert (order.state, order.filled_qty) == ("new", 0)
    h.venue.cancel_order("BTCUSDT", placed["orderId"])
    assert h.venue.get_order("BTCUSDT", client_order_id="m1").state == "canceled"
    h.match()
    assert h.venue.get_order("BTCUSDT", order_id=placed["orderId"]).state == "canceled"
    assert h.venue.get_open_orders() == () and h.venue.get_recent_fills(symbol="BTCUSDT") == ()


@pytest.mark.parametrize("side", ["buy", "sell"])
def test_a_resting_maker_entry_fills_as_maker_never_beyond_its_limit(make, side):
    h = make("maker")
    placed = h.venue.send_limit_order("BTCUSDT", side, 0.1, float(maker_price(side)), client_order_id="m1")
    h.match()
    order = h.venue.get_order("BTCUSDT", order_id=placed["orderId"])
    assert (order.state, order.filled_qty) == ("filled", D("0.1"))
    trades = h.venue.get_recent_fills(symbol="BTCUSDT", order_id=placed["orderId"])
    assert sum(t.qty for t in trades) == D("0.1") and {t.liquidity for t in trades} == {"maker"}
    for t in trades:
        assert t.price <= maker_price("buy") if side == "buy" else t.price >= maker_price("sell")
    assert h.venue.cancel_order("BTCUSDT", placed["orderId"]) is not None
    assert h.venue.get_order("BTCUSDT", order_id=placed["orderId"]).state == "filled"


def test_invalid_duplicate_and_too_small_orders_are_refused(make):
    h = make("market")
    assert h.venue.send_market_order("BTCUSDT", "buy", 0.1, client_order_id="c1")["status"] == "submitted"
    for side, qty, cid in (("buy", 0.1, "c1"), ("hold", 0.1, "c2"), ("buy", 0.1, ""), ("buy", 0.0004, "c3")):
        assert h.venue.send_market_order("BTCUSDT", side, qty, client_order_id=cid)["status"] == "rejected"
    assert h.venue.get_order("BTCUSDT", client_order_id="c2") is None


def test_lookups_answer_none_for_an_unknown_order_and_raise_without_an_identity(make):
    h = make()
    assert h.venue.get_order("BTCUSDT", client_order_id="never-sent") is None
    assert h.venue.get_order("BTCUSDT", order_id="999999") is None
    with pytest.raises(ValueError):
        h.venue.get_order("BTCUSDT")
    with pytest.raises(ValueError):
        h.venue.get_recent_fills(order_id="1")
    with pytest.raises(Exception):         # Binance answers "unknown order"; nothing is canceled either way
        h.venue.cancel_order("BTCUSDT", "999999")


def test_a_symbol_the_exchange_does_not_trade_raises(make):
    h = make()
    for call in (lambda v: v.filters("XRPUSDT"),
                 lambda v: v.send_market_order("XRPUSDT", "buy", 1.0, client_order_id="x"),
                 lambda v: v.place_stop_market("XRPUSDT", "sell", D("1"), client_algo_id="x")):
        with pytest.raises(RuntimeError):
            call(h.venue)


def test_the_account_reads(make):
    h = make()
    filters = h.venue.filters("BTCUSDT")
    assert filters.step_size > 0 and filters.min_qty > 0 and filters.min_notional > 0
    assert isinstance(h.venue.get_account_equity(), float) and h.venue.get_account_equity() > 0
    ticker = h.venue.get_ticker("BTCUSDT")
    assert all(isinstance(ticker[k], float) and ticker[k] > 0 for k in ("lastPrice", "bid1Price", "ask1Price"))
    assert h.venue.execution_blockers() == []
    leverage, margin = h.venue.get_symbol_settings("BTCUSDT")
    assert isinstance(leverage, int) and leverage >= 1 and margin in ("cross", "isolated")


def test_every_symbol_s_settings_in_one_read(make):
    """Plan D2: the whole-account check reads every symbol's leverage and margin mode at once."""
    h = make()
    settings = h.venue.get_settings_snapshot()
    assert settings["BTCUSDT"] == h.venue.get_symbol_settings("BTCUSDT")
    assert settings["ETHUSDT"] == h.venue.get_symbol_settings("ETHUSDT")


# -- protective stops -------------------------------------------------------------------------------------------
def test_a_protective_stop_is_active_until_canceled(make):
    h = make()
    _long(h)
    placed = h.venue.place_stop_market("BTCUSDT", "sell", D("960"), client_algo_id="s1")
    assert placed["status"] == "submitted" and placed["algoId"]
    stop = h.venue.get_algo_order(client_algo_id="s1")
    assert isinstance(stop, StopOrder)
    assert (stop.client_algo_id, stop.symbol, stop.side, stop.state, stop.trigger_price, stop.actual_order_id) == \
        ("s1", "BTCUSDT", "sell", STOP_ACTIVE, D("960"), "")
    assert [s.client_algo_id for s in h.venue.get_open_algo_orders()] == ["s1"]
    h.venue.cancel_algo_order(client_algo_id="s1")
    assert h.venue.get_algo_order(client_algo_id="s1").state == STOP_GONE
    assert h.venue.get_open_algo_orders() == () and h.venue.get_algo_order(client_algo_id="missing") is None


@pytest.mark.parametrize("side,trigger", [("sell", "1000"), ("sell", "1010"), ("buy", "1000"), ("buy", "990")])
def test_a_stop_the_price_has_already_reached_is_refused_as_breached(make, side, trigger):
    placed = make().venue.place_stop_market("BTCUSDT", side, D(trigger), client_algo_id="s1")
    assert (placed["status"], placed["breached"]) == ("rejected", True)


def test_a_triggered_stop_becomes_a_filled_order_that_closes_the_position(make):
    h = make()
    _long(h)
    h.venue.place_stop_market("BTCUSDT", "sell", D("960"), client_algo_id="s1")
    h.crash("BTCUSDT")
    stop = h.venue.get_algo_order(client_algo_id="s1")
    assert stop.state == STOP_TRIGGERED and stop.actual_order_id
    order = h.venue.get_order("BTCUSDT", order_id=stop.actual_order_id)
    assert (order.order_id, order.side, order.state, order.filled_qty) == (stop.actual_order_id, "sell", "filled",
                                                                           D("0.1"))
    assert order.client_order_id
    trades = h.venue.get_recent_fills(symbol="BTCUSDT", order_id=stop.actual_order_id)
    assert sum(t.qty for t in trades) == D("0.1") and {t.side for t in trades} == {"sell"}
    assert h.venue.get_positions("BTCUSDT") == () and h.venue.get_open_algo_orders() == ()


def test_the_stop_manager_and_the_order_lifecycle_work_the_same_on_both(make, tmp_path):
    """The existing execution parts, unchanged: a market entry reaches the ledger, the stop manager places the
    stop, adopts the order the triggered stop became, and the ledger follows the venue back to flat."""
    h = make("market")
    clock = lambda: 1_800_000_000.0  # noqa: E731
    journal = SqliteOrderJournal(state_db(tmp_path / "state.sqlite"), SCOPE)
    lifecycle = OrderLifecycle(h.venue, journal, SCOPE, clock=clock, sleeper=lambda s: None, lock=threading.RLock(),
                               wait_seconds=0, owner=OWNER, entry_order_type="market", not_found_grace_seconds=90)
    service = ExecutionService(h.venue, lifecycle, owner=OWNER, entry_mode="market",
                               limit_offset=LIMIT_OFFSET_BPS / 10000, lock=threading.RLock(), clock=clock,
                               monotonic=clock)
    service.ready = True
    executor = OwnerExecutor(service, scope=SCOPE)
    consumer = TargetFillConsumer(executor)
    stops = ProtectiveStopManager(h.venue, executor, "BTCUSDT", stop_pct=D("0.04"), clock=clock)

    def drain():
        for fill in executor.reconcile_pending():
            consumer.consume(fill)

    executor.send_order(OrderIntent(order_id="BTCUSDT:1:rebalance:0", source=OWNER, symbol="BTCUSDT", side="buy",
                                    qty=D("0.1"), price=PRICE, reduce_only=False, reason="rebalance"))
    h.match()
    drain()
    assert consumer.position.qty == D("0.1")
    assert stops.sync() == [] and stops.has_active_stop()
    (stop,) = h.venue.get_open_algo_orders()
    h.crash("BTCUSDT")
    assert stops.sync() == ["BTCUSDT: protective stop triggered"]
    drain()
    assert consumer.position.qty == 0 and h.venue.get_positions("BTCUSDT") == ()
    assert f"stop-{stop.client_algo_id}" in [row["intent_id"] for row in journal.orders(OWNER)]
    assert journal.orders(unresolved=True) == [] and executor.reconcile_account("BTCUSDT") == []
    assert stops.sync() == [] and not stops.unprotected()
