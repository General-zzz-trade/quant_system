"""Isolated-margin mode of the replay broker: exposure up to 3x, protective stops, liquidation."""
from __future__ import annotations

from decimal import Decimal as D
from types import SimpleNamespace as NS

import pytest

from data.quality.causal_bars import HOUR, Bar
from runner.offline.broker import BarBroker, ReplayConfig


def config(**kwargs):
    base = dict(starting_balance="100", qty_step="0.1", min_notional="1", entry_mode="market", slippage_bps="0",
                taker_bps="5", participation="1", isolated_leverage="10", max_exposure="3",
                maintenance_margin_rate="0.005", stop_slippage_bps="10")
    base.update(kwargs)
    return ReplayConfig(**base)


def bar(i, o, h, low, c):
    return Bar(i * HOUR, D(str(o)), D(str(h)), D(str(low)), D(str(c)), D(100))


def opened(side="buy", qty="2", **kwargs):
    """Broker holding a position entered at 100 on bar 1."""
    broker = BarBroker(config(**kwargs))
    broker.now = HOUR
    broker.send_order(NS(order_id="entry", symbol="ETHUSDT", qty=D(qty), side=side, price=D(100)))
    broker.process_bar(bar(1, 100, 100.5, 99.5, 100))
    assert broker.position.qty == (D(qty) if side == "buy" else -D(qty))
    return broker


def reconciles(broker):
    expected = D(broker.config.starting_balance) + broker.position.realized_pnl - broker.fees + broker.funding
    return broker.cash == expected


def test_legacy_configuration_is_unchanged():
    assert set(ReplayConfig().record()) == {
        "starting_balance", "maker_bps", "taker_bps", "slippage_bps", "limit_offset_bps", "participation",
        "max_exposure", "qty_step", "min_notional", "ttl_bars", "daily_loss_pct", "weekly_loss_pct", "entry_mode",
        "deadzone", "zscore_window", "zscore_warmup", "min_hold", "max_hold"}
    with pytest.raises(ValueError, match="exposure"):
        ReplayConfig(max_exposure="2")
    with pytest.raises(ValueError, match="isolated-margin mode"):
        BarBroker(ReplayConfig()).set_stop(95)
    assert {"isolated_leverage", "maintenance_margin_rate", "stop_slippage_bps"} <= set(config().record())


@pytest.mark.parametrize("kwargs", [{"max_exposure": "3.1"}, {"isolated_leverage": "2", "max_exposure": "3"},
                                    {"isolated_leverage": "21"}, {"maintenance_margin_rate": "0.1"},
                                    {"stop_slippage_bps": "-1"}])
def test_invalid_isolated_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        config(**kwargs)


def test_margin_and_liquidation_price():
    broker = opened()
    assert broker.margin == D("20")                                   # 2 x 100 / 10
    assert broker.liquidation_price() == D("90.5")                    # 100 x (1 - 0.1 + 0.005)
    short = opened(side="sell")
    assert short.liquidation_price() == D("109.5")


def test_stop_crossed_intrabar_fills_at_stop_minus_slippage():
    broker = opened()
    broker.set_stop(97)
    broker.process_bar(bar(2, 99, 99.5, 96, 96.5))
    price = D(97) * D("0.999")
    assert broker.position.qty == 0 and broker.stop_price is None and broker.margin == 0
    assert broker.fills[-1]["order"] == "protective_stop" and D(broker.fills[-1]["price"]) == price
    assert broker.position.realized_pnl == 2 * (price - 100)
    assert broker.fees == D(2) * 100 * D("0.0005") + D(2) * price * D("0.0005")
    assert reconciles(broker)


def test_gap_through_stop_fills_at_open():
    broker = opened()
    broker.set_stop(97)
    broker.process_bar(bar(2, 95, 95.5, 94, 95))
    assert D(broker.fills[-1]["price"]) == D(95) * D("0.999") and broker.position.qty == 0


def test_gap_through_liquidation_loses_exactly_the_posted_margin():
    broker = opened()
    broker.set_stop(97)
    broker.process_bar(bar(2, 90, 92, 89, 91))
    assert broker.position.qty == 0 and broker.fills[-1]["order"] == "liquidation"
    assert broker.position.realized_pnl == -D(20) and broker.liquidations[0]["margin_lost"] == "20"
    assert reconciles(broker)


def test_without_a_stop_touching_liquidation_liquidates():
    broker = opened()
    broker.process_bar(bar(2, 99, 99.5, 90.4, 95))
    assert broker.position.qty == 0 and broker.position.realized_pnl == -D(20)
    survivor = opened()
    survivor.process_bar(bar(2, 99, 99.5, 90.6, 95))                 # stays above 90.5
    assert survivor.position.qty == 2 and not survivor.liquidations


def test_stop_fill_beyond_liquidation_becomes_a_liquidation():
    broker = opened()
    broker.set_stop("90.55")                                          # 90.55 x 0.999 < 90.5
    broker.process_bar(bar(2, 95, 95, 90, 91))
    assert broker.fills[-1]["order"] == "liquidation" and broker.position.realized_pnl == -D(20)


def test_short_stop_is_symmetric():
    broker = opened(side="sell")
    broker.set_stop(103)
    broker.process_bar(bar(2, 101, 104, 100.5, 103.5))
    price = D(103) * D("1.001")
    assert broker.position.qty == 0 and D(broker.fills[-1]["price"]) == price
    assert broker.position.realized_pnl == 2 * (100 - price)


def test_partial_reduction_releases_margin_proportionally():
    broker = opened()
    broker.now = 2 * HOUR
    broker.send_order(NS(order_id="cut", symbol="ETHUSDT", qty=D(1), side="sell", price=D(100), reduce_only=True))
    broker.process_bar(bar(2, 100, 100.5, 99.5, 100))
    assert broker.position.qty == 1 and broker.margin == D(10)


def test_exposure_cap_allows_up_to_three_times_equity():
    allowed = opened(qty="2.5")
    assert allowed.position.qty == D("2.5")
    broker = BarBroker(config())
    broker.now = HOUR
    broker.send_order(NS(order_id="big", symbol="ETHUSDT", qty=D(4), side="buy", price=D(100)))
    broker.process_bar(bar(1, 100, 101, 99, 100))
    assert broker.position.qty == 0 and broker.orders[0]["rejection"] == "fill exceeds exposure cap"
