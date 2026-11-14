"""Protective stops and fill collection of the Binance durable executor, against the fake venue."""
from __future__ import annotations

from decimal import Decimal as D

import pytest

from execution.adapters.binance import BinanceConfig
from execution.adapters.binance.adapter import BinanceAdapter
from execution.adapters.binance.execution_adapter import BinanceExecutionAdapter
from execution.adapters.binance.protective_stops import ProtectiveStopManager
from execution.adapters.binance.rest import BinanceNonRetryableError
from lab.orders import make_order
from runner.binance_target import TargetFillConsumer
from tests.unit.binance.fake_venue import FakeBinance, make_bars

OWNER = "target.BTCUSDT"


class SlicedVenue(FakeBinance):
    """Matches every market order in 0.001 slices and honours the userTrades ``limit`` like Binance."""

    def _apply_fill(self, order, qty, price):
        if order["type"] == "MARKET" and qty > D("0.001"):
            for _ in range(int(qty / D("0.001"))):
                super()._apply_fill(order, D("0.001"), price)
            return
        super()._apply_fill(order, qty, price)

    def request_signed(self, *, method, path, params=None):
        rows = super().request_signed(method=method, path=path, params=params)
        return rows[:int(params["limit"])] if path == "/fapi/v1/userTrades" else rows


def _executor(tmp_path, venue_cls=FakeBinance):
    clock = {"now": 1_800_000_000.0}
    fake = venue_cls(make_bars(10), clock=lambda: clock["now"])
    fake.price["BTCUSDT"] = D("100")
    adapter = BinanceAdapter(BinanceConfig(api_key="k", api_secret="s"), client=fake)
    assert adapter.connect(symbols=("BTCUSDT",))
    executor = BinanceExecutionAdapter(adapter, scope="binance-testnet-ut", owner=OWNER, entry_mode="market",
                                       journal_path=tmp_path / "orders.sqlite", wait_seconds=0,
                                       clock=lambda: clock["now"])
    executor._ready = True
    return fake, adapter, executor, clock, TargetFillConsumer(executor)


def _order(n: int, side: str, qty: str, *, reduce_only: bool = False, reason: str = "rebalance"):
    return make_order(order_id=f"BTCUSDT:{n}:{reason}:0", source=OWNER, symbol="BTCUSDT", side=side,
                      qty=D(qty), price=D("100"), reduce_only=reduce_only, reason=reason)


def _drain(executor, consumer) -> None:
    for event in executor.reconcile_pending():
        consumer.consume(event)


def _with_position(tmp_path):
    fake, adapter, executor, clock, consumer = _executor(tmp_path)
    for event in executor.send_order(_order(1, "buy", "0.1")):
        consumer.consume(event)
    stops = ProtectiveStopManager(adapter, executor, "BTCUSDT", stop_pct=D("0.04"), clock=lambda: clock["now"])
    return fake, executor, consumer, stops


@pytest.mark.known_defect("R3")
def test_current_rejected_stop_leaves_entries_enabled(tmp_path):
    fake, executor, consumer, stops = _with_position(tmp_path)
    original = fake.request_signed

    def refuse_algo_orders(*, method, path, params=None):
        if path == "/fapi/v1/algoOrder" and method == "POST":
            raise BinanceNonRetryableError('HTTP 400: {"code":-4120,"msg":"not supported"}',
                                           http_status=400, code=-4120)
        return original(method=method, path=path, params=params)

    fake.request_signed = refuse_algo_orders
    for _ in range(3):
        issues = executor.reconcile_account("BTCUSDT") + stops.sync()   # as runner/binance_target.py:304
        assert any("protective stop rejected" in issue for issue in issues)
        assert executor._ready is True                  # entries stay enabled without a stop
        assert stops.has_active_stop() is False
    assert fake.positions["BTCUSDT"] == D("0.1")


@pytest.mark.known_defect("R8")
def test_current_stop_triggered_without_order_id_is_never_adopted(tmp_path):
    fake, executor, consumer, stops = _with_position(tmp_path)
    stops.sync()
    (client_id,) = [k for k, v in fake.algo.items() if v["algoStatus"] == "NEW"]
    fake.algo[client_id]["algoStatus"] = "TRIGGERED"      # status first, actualOrderId still empty
    stops.sync()
    fake.trigger_stop(client_id, D("95.9"))              # then the venue closes the position
    stops.sync()
    assert not [r for r in executor.journal.orders(OWNER) if r["intent_id"].startswith("stop-")]
    assert executor.reconcile_pending() == ()
    assert fake.positions["BTCUSDT"] == 0
    assert executor.journal.checkpoint(OWNER)["position"]["qty"] == "0.1"
    assert executor.reconcile_account("BTCUSDT") == ["BTCUSDT: venue qty=0, ledger qty=0.1"]


def test_triggered_stop_with_order_id_reaches_the_ledger(tmp_path):
    fake, executor, consumer, stops = _with_position(tmp_path)
    stops.sync()
    (client_id,) = [k for k, v in fake.algo.items() if v["algoStatus"] == "NEW"]
    fake.trigger_stop(client_id, D("95.9"))
    stops.sync()
    _drain(executor, consumer)
    assert consumer.position.qty == 0


@pytest.mark.known_defect("R9")
def test_current_order_with_more_than_100_trades_never_settles_and_blocks_reductions(tmp_path):
    fake, adapter, executor, clock, consumer = _executor(tmp_path, SlicedVenue)
    executor.send_order(_order(1, "buy", "0.05"))         # 50 trades: settles
    _drain(executor, consumer)
    assert executor.reconcile_account("BTCUSDT") == []
    executor.send_order(_order(2, "buy", "0.15"))         # 150 trades: only 100 are ever read
    for _ in range(3):
        clock["now"] += 3600
        _drain(executor, consumer)
    assert consumer.position.qty == D("0.15") and fake.positions["BTCUSDT"] == D("0.2")
    executor.send_order(_order(3, "sell", "0.2", reduce_only=True, reason="kill_switch"))
    _drain(executor, consumer)
    assert [(r["status"], r["reducing"]) for r in executor.journal.orders(OWNER, unresolved=True)] == [
        ("submitted", 0), ("submitted", 1)]
    posts = len(fake.posts)
    executor.send_order(_order(4, "sell", "0.05", reduce_only=True, reason="kill_switch"))
    assert len(fake.posts) == posts                     # the next reduction is refused before placement
    assert fake.positions["BTCUSDT"] == D("0.05")
