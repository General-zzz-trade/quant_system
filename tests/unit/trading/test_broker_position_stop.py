"""The research broker's per-position stop distance (research only; tech lead 2026-11-17: the lab tests a stop
that tightens with low volatility, research/lab/runs/2026-11-17/vol_stop.md).

The driver gives ``set_next_stop_pct`` with the orders of a decision. A position opened by those orders (from flat,
or by a flip) keeps that distance until it is flat; adding to it keeps the distance at the new average entry. The
distance can only tighten: it must lie in (0, ``stop_pct``], the configured stop being the cap a person set. None
(the default) is the configured stop, so a broker nobody gives a distance - paper - is unchanged.
"""
from __future__ import annotations

from decimal import Decimal as D
from types import SimpleNamespace as NS

import pytest

from data.quality.causal_bars import HOUR, Bar
from trading.adapters.simulated.bar_broker import BarBroker, ReplayConfig


def config(**kwargs):
    base = dict(starting_balance="1000", qty_step="0.1", min_notional="1", entry_mode="market", slippage_bps="0",
                taker_bps="5", participation="1", stop_pct="0.08", stop_slippage_bps="10")
    base.update(kwargs)
    return ReplayConfig(**base)


def bar(i, o, h, low, c):
    return Bar(i * HOUR, D(str(o)), D(str(h)), D(str(low)), D(str(c)), D(1000))


def send(broker, side, qty, at, oid):
    broker.now = at * HOUR
    broker.send_order(NS(order_id=oid, symbol="ALTUSDT", qty=D(qty), side=side, price=D(100)))


def stop_fills(broker):
    return [f for f in broker.fills if f["order"] == "protective_stop"]


def entered(pct, side="buy", qty="2", **kwargs):
    broker = BarBroker(config(**kwargs))
    broker.set_next_stop_pct(pct)
    send(broker, side, qty, 1, "entry")
    broker.process_bar(bar(1, 100, 101, 99, 100))
    return broker


def test_a_position_opened_with_a_distance_is_stopped_at_that_distance():
    broker = entered("0.03")
    broker.process_bar(bar(2, 99, 99.5, 96.9, 97.5))               # 100 x 0.97 = 97 reached; 8 % never
    [fill] = stop_fills(broker)
    assert D(fill["price"]) == D("97") * (1 - D("0.001")) and broker.position.qty == 0


def test_without_a_distance_the_configured_stop_applies():
    broker = entered(None)
    broker.process_bar(bar(2, 99, 99.5, 96.9, 97.5))
    assert stop_fills(broker) == [] and broker.position.qty == 2
    broker.process_bar(bar(3, 97, 97, 91.9, 93))                    # 92 reached
    assert D(stop_fills(broker)[0]["price"]) == D("92") * (1 - D("0.001"))


def test_a_short_s_distance_is_above_the_entry():
    broker = entered("0.03", side="sell")
    broker.process_bar(bar(2, 101, 102.9, 100, 102))                # below 103: kept
    assert broker.position.qty == D("-2")
    broker.process_bar(bar(3, 102, 103.2, 101, 103))
    assert D(stop_fills(broker)[0]["price"]) == D("103") * (1 + D("0.001")) and broker.position.qty == 0


def test_a_later_distance_does_not_move_the_open_position_s_stop():
    broker = entered("0.03")
    broker.set_next_stop_pct("0.08")                                 # the next decision's distance
    broker.process_bar(bar(2, 99, 99.5, 96.9, 97.5))
    assert len(stop_fills(broker)) == 1                              # still 3 %


def test_adding_to_a_position_keeps_its_distance_at_the_new_average_entry():
    broker = entered("0.03")
    broker.set_next_stop_pct("0.08")
    send(broker, "buy", "2", 2, "add")
    broker.process_bar(bar(2, 102, 102, 101, 102))                   # adds 2 at 102: average 101
    assert broker.position.entry_price == D(101)
    broker.process_bar(bar(3, 100, 100, 98.0, 98.5))                 # 101 x 0.97 = 97.97: kept
    assert stop_fills(broker) == []
    broker.process_bar(bar(4, 98.5, 98.5, 97.9, 98))
    assert D(stop_fills(broker)[0]["price"]) == D(101) * D("0.97") * (1 - D("0.001"))


def test_a_flip_takes_the_distance_of_the_decision_that_flipped():
    broker = entered("0.03")
    broker.set_next_stop_pct("0.05")
    send(broker, "sell", "4", 2, "flip")
    broker.process_bar(bar(2, 100, 100.5, 99.8, 100))                # short 2 at 100
    assert broker.position.qty == D("-2")
    broker.process_bar(bar(3, 100, 104.5, 100, 104))                 # 103 passed, 105 not: kept
    assert stop_fills(broker) == []
    broker.process_bar(bar(4, 104, 105.2, 103, 105))
    assert D(stop_fills(broker)[0]["price"]) == D(105) * (1 + D("0.001"))


def test_a_new_position_after_a_stop_takes_the_newest_distance():
    broker = entered("0.03")
    broker.process_bar(bar(2, 99, 99.5, 96.9, 97.5))                 # stopped
    broker.set_next_stop_pct("0.05")
    send(broker, "buy", "2", 3, "again")
    broker.process_bar(bar(3, 100, 100, 99, 100))
    broker.process_bar(bar(4, 99, 99, 95.5, 96))                     # 3 % passed, 5 % not
    assert len(stop_fills(broker)) == 1 and broker.position.qty == 2


def test_the_distance_works_in_isolated_mode():
    broker = entered("0.03", isolated_leverage="10", max_exposure="1", stop_pct="0.045")
    broker.process_bar(bar(2, 99, 99.5, 96.9, 97.5))
    assert D(stop_fills(broker)[0]["price"]) == D("97") * (1 - D("0.001")) and broker.liquidations == []


@pytest.mark.parametrize("pct", ["0", "-0.01", "0.0801", "NaN", "Infinity"])
def test_a_distance_outside_zero_to_the_configured_stop_is_refused(pct):
    broker = BarBroker(config())
    with pytest.raises(ValueError, match="stop distance"):
        broker.set_next_stop_pct(pct)


def test_the_configured_stop_itself_is_allowed_and_none_resets():
    broker = BarBroker(config())
    broker.set_next_stop_pct("0.08")
    assert broker.next_stop_pct == D("0.08")
    broker.set_next_stop_pct(None)
    assert broker.next_stop_pct is None


def test_a_broker_without_a_stop_takes_no_distance():
    broker = BarBroker(config(stop_pct="0"))
    with pytest.raises(ValueError, match="stop distance"):
        broker.set_next_stop_pct("0.03")
