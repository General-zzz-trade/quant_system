"""The simulated exchange's saved state (ADR-0013 consequences: paper state = the exchange's snapshot + the order
journal; S10 part 2). Saving through the paper stores' codec and restoring gives an exchange that trades on
exactly as the original; a snapshot that does not fit is refused and changes nothing.
"""
from __future__ import annotations

import json
from decimal import Decimal as D

import pytest

from tests.unit.trading.test_simulated_exchange import SYMBOLS, Feed, exchange, maker_price
from trading.adapters.codec import decode, encode
from trading.adapters.simulated.exchange import BarStopModel, SimFilters, SimulatedExchange


def _stored(document):
    """Through JSON as the paper stores write it (trading/adapters/codec.py)."""
    return json.loads(json.dumps(document, default=encode), object_hook=decode)


def _traded():
    """Maker entries, a filled and a triggered stop, an active stop, a partly filled entry still resting, a
    canceled order and funding: every kind of state the exchange keeps."""
    venue, brokers = exchange("maker")
    feed = Feed(venue)
    feed()
    feed(symbol="ETHUSDT")
    venue.send_limit_order("BTCUSDT", "buy", 1.0, float(maker_price("buy")), client_order_id="b1")
    feed()
    venue.place_stop_market("BTCUSDT", "sell", D("96"), client_algo_id="s1")
    feed(o=99, h=99, lo=95, c=97)                                      # s1 closes the long
    venue.send_limit_order("BTCUSDT", "buy", 0.5, float(maker_price("buy", "97")), client_order_id="b2")
    feed(o=97, h=98, lo=96, c=97)
    venue.place_stop_market("BTCUSDT", "sell", D("85"), client_algo_id="s2")
    gone = venue.send_limit_order("ETHUSDT", "sell", 0.2, float(maker_price("sell")), client_order_id="e0")
    venue.cancel_order("ETHUSDT", gone["orderId"])
    venue.send_limit_order("ETHUSDT", "buy", 2.0, float(maker_price("buy")), client_order_id="e1")
    feed(symbol="ETHUSDT", volume="50", funding=D("0.0001"))           # 0.5 of 2.0 filled, the rest rests
    return venue, brokers, feed


def _next_bars(venue, feed):
    feed(o=97, h=98, lo=84, c=86)                                      # s2 closes the second long
    feed(symbol="ETHUSDT", volume="50", funding=D("-0.0003"))
    return [venue.send_market_order(s, "sell", 0.1, reduce_only=True, client_order_id=f"x-{s}")
            for s in ("BTCUSDT", "ETHUSDT")]


def test_a_stored_exchange_restores_exactly_and_trades_on_identically():
    venue, brokers, feed = _traded()
    assert venue.get_open_orders("ETHUSDT") and venue.get_open_algo_orders() and venue.get_positions()
    document = _stored(venue.snapshot())
    restored, restored_brokers = exchange("maker")
    restored.restore(document)
    assert restored.snapshot() == venue.snapshot()
    assert restored_brokers["ETHUSDT"].pending[0] is restored_brokers["ETHUSDT"].orders[-1]   # one row, as before
    assert restored.get_open_orders() == venue.get_open_orders()
    for symbol, client_id in (("ETHUSDT", "e1"), ("BTCUSDT", "autoclose-s1")):
        assert restored.get_order(symbol, client_order_id=client_id) == venue.get_order(symbol,
                                                                                        client_order_id=client_id)
    assert restored.send_limit_order("BTCUSDT", "buy", 0.1, float(maker_price("buy", "97")),
                                     client_order_id="b2")["msg"] == "duplicate client order id"
    again = Feed(restored)
    again.at = dict(feed.at)
    assert _next_bars(restored, again) == _next_bars(venue, feed)
    assert restored.snapshot() == venue.snapshot()
    stop = restored.get_algo_order(client_algo_id="s2")
    assert stop.state == "triggered" and restored.get_order("BTCUSDT", order_id=stop.actual_order_id).state == \
        "filled"
    assert restored.get_recent_fills(symbol="ETHUSDT") == venue.get_recent_fills(symbol="ETHUSDT")
    assert restored.get_symbol_equity("ETHUSDT") == venue.get_symbol_equity("ETHUSDT") != D("10000")
    assert restored_brokers["BTCUSDT"].fills == brokers["BTCUSDT"].fills


def test_a_snapshot_is_a_copy_and_so_is_what_is_restored():
    venue, _, feed = _traded()
    snapshot = venue.snapshot()
    kept = _stored(snapshot)
    _next_bars(venue, feed)
    assert snapshot == kept
    restored, _ = exchange("maker")
    restored.restore(kept)
    again = Feed(restored)
    again.at = dict(feed.at)
    _next_bars(restored, again)
    assert kept == _stored(snapshot)


def test_a_snapshot_that_does_not_fit_is_refused_and_changes_nothing():
    venue, _, _ = _traded()
    document = _stored(venue.snapshot())
    other, _ = exchange("maker", ttl_bars=5)                           # another broker configuration
    _, maker_brokers = exchange("maker")
    _, market_brokers = exchange("market")
    mixed = SimulatedExchange({"BTCUSDT": maker_brokers["BTCUSDT"], "ETHUSDT": market_brokers["ETHUSDT"]},
                              filters=dict.fromkeys(SYMBOLS, SimFilters(D("0.001"), D("0.001"), D("5"))),
                              stop_model=BarStopModel())          # BTCUSDT fits, ETHUSDT does not
    fewer, _ = exchange("maker", symbols=("BTCUSDT",))
    lost_row = _stored(document)
    lost_row["brokers"]["ETHUSDT"]["pending"] = [999]
    for target, state in ((other, document), (mixed, document), (fewer, document), (exchange("maker")[0], lost_row)):
        before = target.snapshot()
        with pytest.raises((RuntimeError, KeyError)):
            target.restore(state)
        assert target.snapshot() == before
