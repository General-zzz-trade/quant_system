"""The simulated exchange (ADR-0013 decision 1, design 2026-11 M5 section 4.2, S10 part 2).

What only the simulated exchange does: the bar stop model of decision L, matching on closed bars exactly as the
research sleeve does, the lot step and the maker price research uses, and its saved state. What it shares with
the Binance adapter is fixed by the port-contract suite (tests/contract/test_exchange_ports.py).
"""
from __future__ import annotations

from decimal import Decimal as D

import pytest

from trading.adapters.simulated.bar_broker import ReplayConfig
from trading.adapters.simulated.exchange import BarStopModel
from trading.application import ports
from trading.domain.bars import HOUR, Bar

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
