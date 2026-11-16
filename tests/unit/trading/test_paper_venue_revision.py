"""S10 decision 4: the simulated exchange's revision moves with every change of its state, so the paper state is
written with the next journal transaction after any change; reads leave it alone."""
from __future__ import annotations

from decimal import Decimal as D

from lab.strategy import StrategySpec
from trading.domain.bars import HOUR, Bar
from trading.infrastructure.paper import paper_exchange

T0 = 1_700_006_400_000 // HOUR * HOUR
SPEC = StrategySpec(lookbacks=(5, 10, 20), vol_lookback=10, execution="market")


def _bar(k, price="100"):
    p = D(price)
    return Bar(T0 + k * HOUR, p, p * D("1.01"), p * D("0.99"), p, D(1000))


def test_every_change_moves_the_revision_and_reads_do_not():
    venue = paper_exchange(SPEC, ["BTCUSDT", "ETHUSDT"])
    seen = [venue.revision]

    def moved():
        seen.append(venue.revision)
        return seen[-1] != seen[-2]
    venue.on_closed_bar("BTCUSDT", _bar(0), None)
    assert moved()
    venue.send_market_order("BTCUSDT", "buy", 2, client_order_id="c1")
    assert moved()
    venue.cancel_order("BTCUSDT", venue.get_order("BTCUSDT", client_order_id="c1").order_id)
    assert moved()
    venue.send_market_order("BTCUSDT", "buy", 2, client_order_id="c2")   # above the minimum notional
    venue.on_closed_bar("BTCUSDT", _bar(1), D("0.0001"))
    assert moved() and venue.get_position("BTCUSDT").qty == D("2")
    venue.place_stop_market("BTCUSDT", "sell", D("90"), client_algo_id="s1")
    assert moved()
    venue.cancel_algo_order(client_algo_id="s1")
    assert moved()
    venue.restore(venue.snapshot())
    assert moved()
    for read in (lambda: venue.get_positions(), lambda: venue.get_open_orders(), lambda: venue.snapshot(),
                 lambda: venue.get_open_algo_orders(), lambda: venue.get_account_equity(),
                 lambda: venue.get_recent_fills(symbol="BTCUSDT"), lambda: venue.get_ticker("BTCUSDT"),
                 lambda: venue.has_matched("BTCUSDT", _bar(1))):
        read()
        assert not moved()


def test_a_bar_counts_as_matched_once_the_venue_has_processed_it():
    venue = paper_exchange(SPEC, ["BTCUSDT", "ETHUSDT"])
    assert not venue.has_matched("BTCUSDT", _bar(0))
    venue.on_closed_bar("BTCUSDT", _bar(0), None)
    assert venue.has_matched("BTCUSDT", _bar(0)) and not venue.has_matched("BTCUSDT", _bar(1))
    assert not venue.has_matched("ETHUSDT", _bar(0))           # per symbol
