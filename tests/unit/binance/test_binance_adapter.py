"""BinanceAdapter contract: never fail open, deterministic ids, closed bars only."""
from __future__ import annotations

from decimal import Decimal as D

import pytest

from execution.adapters.binance import BinanceConfig
from execution.adapters.binance.adapter import MAX_LEVERAGE, BinanceAdapter
from tests.unit.binance.fake_venue import HOUR, FakeBinance, make_bars


@pytest.fixture
def venue():
    bars = make_bars(50)
    clock = {"now": (bars[-1]["start"] + HOUR // 2) / 1000}  # the last bar is still open
    fake = FakeBinance(bars, clock=lambda: clock["now"])
    adapter = BinanceAdapter(BinanceConfig(api_key="k", api_secret="s", testnet=True), client=fake)
    assert adapter.connect(symbols=("BTCUSDT",))
    return fake, adapter, bars


def test_position_snapshot_errors_propagate(venue):
    fake, adapter, _ = venue
    fake.position_error = True
    with pytest.raises(Exception):
        adapter.get_positions("BTCUSDT")


def test_hedge_mode_blocks_execution():
    fake = FakeBinance(make_bars(5))
    fake.hedge = True
    adapter = BinanceAdapter(BinanceConfig(api_key="k", api_secret="s"), client=fake)
    assert adapter.connect(symbols=("BTCUSDT",))
    assert any("hedge" in issue for issue in adapter.execution_blockers())


def test_leverage_is_hard_capped(venue):
    _, adapter, _ = venue
    adapter.set_leverage("BTCUSDT", MAX_LEVERAGE)
    with pytest.raises(ValueError):
        adapter.set_leverage("BTCUSDT", MAX_LEVERAGE + 1)


def test_market_order_contract(venue):
    fake, adapter, _ = venue
    ok = adapter.send_market_order("BTCUSDT", "buy", 0.0029, client_order_id="qs1")
    assert ok["status"] == "submitted" and fake.posts[-1]["quantity"] == "0.002"  # floored to step
    assert fake.posts[-1]["newClientOrderId"] == "qs1"
    assert adapter.send_market_order("BTCUSDT", "buy", 0.0004, client_order_id="qs2")["status"] == "rejected"
    fake.fault = "reject"
    assert adapter.send_market_order("BTCUSDT", "buy", 0.01, client_order_id="qs3")["status"] == "rejected"
    fake.fault = "timeout_before"
    assert adapter.send_market_order("BTCUSDT", "buy", 0.01, client_order_id="qs4")["status"] == "error"


def test_get_order_none_only_for_missing(venue):
    _, adapter, _ = venue
    assert adapter.get_order("BTCUSDT", client_order_id="never-sent") is None
    adapter.send_market_order("BTCUSDT", "sell", 0.01, client_order_id="qs5")
    order = adapter.get_order("BTCUSDT", client_order_id="qs5")
    assert order.state == "filled" and order.filled_qty == D("0.01") and order.side == "sell"


def test_klines_closed_only_and_ascending(venue):
    fake, adapter, bars = venue
    rows = adapter.get_klines("BTCUSDT", "60", 30, now_ms=fake.now_ms())
    starts = [r["start"] for r in rows]
    assert starts == sorted(starts) and len(rows) == 30
    assert starts[-1] == bars[-2]["start"]  # the open bar is excluded
    assert all(r["time"] * 1000 == r["start"] for r in rows)


def test_protective_stop_uses_algo_service(venue):
    fake, adapter, _ = venue
    result = adapter.place_stop_market("BTCUSDT", "sell", D("50.05"), client_algo_id="qsstop1")
    assert result["status"] == "submitted"
    algo = fake.algo["qsstop1"]
    assert algo["closePosition"] == "true" and algo["triggerPrice"] == "50"
    assert adapter.get_algo_order(client_algo_id="qsstop1").status == "NEW"
    assert adapter.get_algo_order(client_algo_id="missing") is None
    bad = adapter.place_stop_market("BTCUSDT", "sell", D("1000"), client_algo_id="qsstop2")
    assert bad["status"] == "rejected" and bad["code"] == -2021
