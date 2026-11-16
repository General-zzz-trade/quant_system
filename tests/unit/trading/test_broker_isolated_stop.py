"""The research broker's ``stop_pct`` in isolated-margin mode (tech lead 2026-11-16: the pilot runs 10x isolated with
a 4.5 % stop and positions of 2 to 3 times the capital; the backtest must model that).

In isolated mode the stop is placed by ``stop_pct`` at the position's current average entry, like the cross-margin
stop and the live runtime's exchange stop; liquidation is still modelled, and a stop that would fill beyond the
liquidation price liquidates instead. Without ``stop_pct`` isolated mode is unchanged (``set_stop`` by hand).
"""
from __future__ import annotations

from decimal import Decimal as D
from types import SimpleNamespace as NS

import pytest

from data.quality.causal_bars import HOUR, Bar
from trading.adapters.simulated.bar_broker import BarBroker, ReplayConfig


def config(**kwargs):
    base = dict(starting_balance="100", qty_step="0.1", min_notional="1", entry_mode="market", slippage_bps="0",
                taker_bps="5", participation="1", isolated_leverage="10", max_exposure="3",
                maintenance_margin_rate="0.005", stop_slippage_bps="10", stop_pct="0.045")
    base.update(kwargs)
    return ReplayConfig(**base)


def bar(i, o, h, low, c):
    return Bar(i * HOUR, D(str(o)), D(str(h)), D(str(low)), D(str(c)), D(1000))


def opened(side="buy", qty="2.5", **kwargs):
    broker = BarBroker(config(**kwargs))
    broker.now = HOUR
    broker.send_order(NS(order_id="entry", symbol="ETHUSDT", qty=D(qty), side=side, price=D(100)))
    broker.process_bar(bar(1, 100, 100.5, 99.5, 100))
    assert broker.position.qty == (D(qty) if side == "buy" else -D(qty))
    return broker


def stop_fills(broker):
    return [f for f in broker.fills if f["order"] == "protective_stop"]


def test_a_leveraged_long_is_stopped_at_its_entry_less_the_stop_distance():
    broker = opened()                                              # 250 notional on 100: 2.5 times the capital
    broker.process_bar(bar(2, 99, 99.5, 95.0, 96))                 # the trigger is 95.5
    assert broker.position.qty == 0 and broker.liquidations == []
    assert D(stop_fills(broker)[0]["price"]) == D("95.5") * (1 - D("0.001"))


def test_a_leveraged_short_is_stopped_above_its_entry():
    broker = opened(side="sell")
    broker.process_bar(bar(2, 101, 104.6, 100, 104))
    assert broker.position.qty == 0
    assert D(stop_fills(broker)[0]["price"]) == D("104.5") * (1 + D("0.001"))


def test_the_stop_guards_the_bar_whose_open_filled_the_entry():
    broker = BarBroker(config())
    broker.now = HOUR
    broker.send_order(NS(order_id="entry", symbol="ETHUSDT", qty=D("2.5"), side="buy", price=D(100)))
    broker.process_bar(bar(1, 100, 100.5, 95.0, 96))
    assert broker.position.qty == 0 and len(stop_fills(broker)) == 1


def test_a_gap_beyond_the_liquidation_price_still_liquidates():
    broker = opened()
    broker.process_bar(bar(2, 89, 89, 88, 88))                     # opens 11 % down: past the 9.5 % liquidation
    assert broker.position.qty == 0 and len(broker.liquidations) == 1 and stop_fills(broker) == []


def test_without_stop_pct_isolated_mode_is_unchanged():
    broker = opened(stop_pct="0")
    broker.process_bar(bar(2, 99, 99.5, 95.0, 96))
    assert broker.position.qty == D("2.5") and stop_fills(broker) == []
    broker.set_stop(96)
    broker.process_bar(bar(3, 96, 96.5, 95.5, 96))
    assert broker.position.qty == 0


def test_a_stop_beyond_the_liquidation_distance_is_refused():
    with pytest.raises(ValueError, match="liquidation"):
        config(stop_pct="0.10")                                    # 10x isolated liquidates about 9.5 % away
