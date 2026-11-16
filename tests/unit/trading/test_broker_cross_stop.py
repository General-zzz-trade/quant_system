"""The research broker's cross-margin protective stop (``stop_pct``, tech lead 2026-11-16: the backtest models the
live runtime's exchange stop).

Live (trading/application/execution/stops.py) keeps one STOP_MARKET closePosition per position, triggered at the
venue entry price x (1 - stop_pct) for a long and x (1 + stop_pct) for a short, placed seconds after the entry
fills. The broker's market entries fill at the bar's open, so the stop guards the rest of that bar and every later
bar, at the position's current average entry. A stop reached within a bar fills at the trigger, or at the open when
the open has already gapped through it, worse by ``stop_slippage_bps`` and charged the taker fee: the rule of the
isolated-mode stop and of paper's BarStopModel. Off by default; with it off the broker and its record are unchanged.
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


def opened(side="buy", qty="2", **kwargs):
    """A broker whose position was entered at 100 at bar 1's open; bar 1 stays within 1 %."""
    broker = BarBroker(config(**kwargs))
    send(broker, side, qty, 1, "entry")
    broker.process_bar(bar(1, 100, 101, 99, 100))
    assert broker.position.qty == (D(qty) if side == "buy" else -D(qty))
    return broker


def stop_fills(broker):
    return [f for f in broker.fills if f["order"] == "protective_stop"]


def test_a_long_s_stop_fills_at_the_trigger_less_the_stop_slippage_and_pays_the_taker_fee():
    broker = opened()
    broker.process_bar(bar(2, 99, 99.5, 91.0, 93))              # the low reaches 100 x 0.92 = 92
    price = D("92") * (1 - D("0.001"))
    assert broker.position.qty == 0
    [fill] = stop_fills(broker)
    assert {k: fill[k] for k in ("id", "at", "side", "qty")} == {"id": "offline-fill-2", "at": 2 * HOUR + HOUR,
                                                                 "side": "sell", "qty": "2"}
    assert D(fill["price"]) == price and D(fill["fee"]) == D(2) * price * 5 / 10000
    assert broker.cash == D(1000) + broker.position.realized_pnl - broker.fees


def test_an_open_gapped_through_the_trigger_fills_at_the_open():
    broker = opened()
    broker.process_bar(bar(2, 90, 90.5, 88, 89))
    assert D(stop_fills(broker)[0]["price"]) == D("90") * (1 - D("0.001"))


def test_a_short_s_stop_is_above_the_entry():
    broker = opened(side="sell")
    broker.process_bar(bar(2, 101, 107.9, 100, 107))           # below 108: kept
    assert broker.position.qty == D("-2")
    broker.process_bar(bar(3, 107, 108.5, 106, 108))
    assert broker.position.qty == 0
    assert D(stop_fills(broker)[0]["price"]) == D("108") * (1 + D("0.001"))
    assert stop_fills(broker)[0]["side"] == "buy"


def test_a_bar_that_stays_above_the_trigger_keeps_the_position():
    broker = opened()
    broker.process_bar(bar(2, 99, 99.5, 92.01, 93))
    assert broker.position.qty == D("2") and stop_fills(broker) == []


def test_the_stop_guards_the_bar_whose_open_filled_the_entry():
    broker = BarBroker(config())
    send(broker, "buy", "2", 1, "entry")
    broker.process_bar(bar(1, 100, 100.5, 91.5, 95))            # entered at the open, stopped in the same bar
    assert broker.position.qty == 0 and len(stop_fills(broker)) == 1


def test_the_trigger_follows_the_average_entry_after_an_addition():
    broker = opened(qty="1")
    send(broker, "buy", "1", 2, "add")
    broker.process_bar(bar(2, 96, 96.5, 95.5, 96))              # average entry 98, trigger 90.16
    assert broker.position.entry_price == D(98)
    broker.process_bar(bar(3, 95, 95, 90.2, 91))
    assert broker.position.qty == D(2)                          # 90.2 > 90.16: kept
    broker.process_bar(bar(4, 91, 91, 90.1, 90.5))
    assert broker.position.qty == 0
    assert D(stop_fills(broker)[0]["price"]) == D("98") * D("0.92") * (1 - D("0.001"))


def test_after_the_stop_nothing_more_is_stopped():
    broker = opened()
    broker.process_bar(bar(2, 99, 99.5, 91.0, 93))
    broker.process_bar(bar(3, 93, 93, 50, 60))
    assert len(stop_fills(broker)) == 1 and broker.position.qty == 0


def test_off_by_default_the_broker_and_its_record_are_unchanged():
    legacy = ReplayConfig()
    assert legacy.stop_pct == "0"
    assert "stop_pct" not in legacy.record() and "stop_slippage_bps" not in legacy.record()
    broker = opened(stop_pct="0")
    broker.process_bar(bar(2, 99, 99.5, 50, 60))
    assert broker.position.qty == D(2) and stop_fills(broker) == []
    recorded = config().record()
    assert recorded["stop_pct"] == "0.08" and recorded["stop_slippage_bps"] == "10"
    assert "isolated_leverage" not in recorded


@pytest.mark.parametrize("kwargs,match", [
    ({"stop_pct": "-0.01"}, "stop_pct"),
    ({"stop_pct": "1"}, "stop_pct"),
    ({"stop_pct": "nan"}, "stop_pct"),
    ({"stop_pct": "0.08", "entry_mode": "maker"}, "market"),
    ({"stop_pct": "0.08", "isolated_leverage": "5", "max_exposure": "3"}, "isolated"),
])
def test_invalid_stop_settings_are_refused(kwargs, match):
    with pytest.raises(ValueError, match=match):
        config(**kwargs)


def test_paper_s_simulated_exchange_refuses_a_broker_that_stops_itself():
    from trading.adapters.simulated.exchange import BarStopModel, SimFilters, SimulatedExchange
    broker = BarBroker(config())
    filters = {"ALTUSDT": SimFilters(step_size=D("0.1"), min_qty=D("0.1"), min_notional=D("1"))}
    with pytest.raises(ValueError, match="models its own stops"):
        SimulatedExchange({"ALTUSDT": broker}, filters=filters, stop_model=BarStopModel())
