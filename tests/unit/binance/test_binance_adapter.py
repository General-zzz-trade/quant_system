"""BinanceAdapter contract: never fail open, deterministic ids, closed bars only."""
from __future__ import annotations

from decimal import Decimal as D

import pytest

from trading.adapters.binance.config import BinanceConfig
from trading.adapters.binance.adapter import BinanceAdapter
from trading.domain.venue import STOP_ACTIVE
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


def test_connecting_is_read_only():
    """ADR-0010: leverage and margin mode are set by a person; connecting only reads."""
    fake = FakeBinance(make_bars(5))
    calls = []
    signed = fake.request_signed
    fake.request_signed = lambda **kw: calls.append(kw["method"]) or signed(**kw)
    adapter = BinanceAdapter(BinanceConfig(api_key="k", api_secret="s"), client=fake)
    assert adapter.connect(symbols=("BTCUSDT", "ETHUSDT"))
    assert set(calls) == {"GET"} and fake.leverage == {} and fake.margin_type == {}
    assert adapter.execution_blockers() == []


def test_symbol_settings_are_read_as_the_exchange_reports_them():
    fake = FakeBinance(make_bars(5))
    fake.leverage["BTCUSDT"], fake.margin_type["BTCUSDT"] = 10, "isolated"
    adapter = BinanceAdapter(BinanceConfig(api_key="k", api_secret="s"), client=fake)
    assert adapter.get_symbol_settings("BTCUSDT") == (10, "isolated")
    assert adapter.get_symbol_settings("ETHUSDT") == (3, "cross")
    fake.margin_type["ETHUSDT"] = "unknown"
    with pytest.raises(RuntimeError, match="could not be read"):
        adapter.get_symbol_settings("ETHUSDT")


def test_open_conditional_orders_of_every_symbol_are_listed(venue):
    fake, adapter, _ = venue
    adapter.place_stop_market("BTCUSDT", "sell", D("50.05"), client_algo_id="qsstop1")
    fake.algo["web"] = {"algoId": 7, "clientAlgoId": "web", "symbol": "SOXLUSDT", "side": "BUY",
                        "algoStatus": "NEW", "triggerPrice": "40"}
    fake.algo["done"] = {"algoId": 8, "clientAlgoId": "done", "symbol": "ETHUSDT", "side": "BUY",
                         "algoStatus": "CANCELED", "triggerPrice": "40"}
    listed = {(o.symbol, o.client_algo_id, o.state) for o in adapter.get_open_algo_orders()}
    assert listed == {("BTCUSDT", "qsstop1", STOP_ACTIVE), ("SOXLUSDT", "web", STOP_ACTIVE)}


def _answer(fake, rows):
    original = fake.request_signed

    def answer(*, method, path, params=None):
        if path == "/fapi/v2/positionRisk":
            return rows
        return original(method=method, path=path, params=params)
    fake.request_signed = answer


def test_one_symbol_s_position_flat_missing_and_hedge_mode(venue):
    # S4-S5 audit, mutant A5: get_position (ADR-0012) had no direct test.
    fake, adapter, _ = venue
    flat = adapter.get_position("BTCUSDT")
    assert flat is not None and flat.qty == 0                          # flat is a row with qty 0
    fake.positions["BTCUSDT"], fake.entry["BTCUSDT"] = D("-0.2"), D("101")
    held = adapter.get_position("BTCUSDT")
    assert held is not None and held.qty == D("-0.2") and held.entry_price == D("101")
    _answer(fake, [])
    assert adapter.get_position("BTCUSDT") is None                     # no row: unknown, never flat
    row = {"symbol": "BTCUSDT", "positionAmt": "0.1", "entryPrice": "100", "markPrice": "100", "leverage": "3",
           "marginType": "cross"}
    _answer(fake, [dict(row, positionSide="LONG"), dict(row, positionSide="SHORT", positionAmt="0")])
    with pytest.raises(RuntimeError, match="Hedge-mode"):
        adapter.get_position("BTCUSDT")
    _answer(fake, [dict(row, positionSide="LONG")])
    with pytest.raises(RuntimeError, match="Hedge-mode"):
        adapter.get_position("BTCUSDT")
