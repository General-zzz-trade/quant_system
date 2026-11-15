"""Protective stops and fill collection of the Binance durable executor, against the fake venue."""
from __future__ import annotations

from decimal import Decimal as D

from trading.adapters.binance.config import BinanceConfig
from trading.adapters.binance.adapter import BinanceAdapter
from execution.adapters.binance.execution_adapter import BinanceExecutionAdapter
from trading.application.execution.stops import ProtectiveStopManager
from trading.adapters.binance.rest import BinanceNonRetryableError
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


def test_a_position_without_a_working_stop_blocks_entries_until_one_is_in_place(tmp_path):
    """R3, fixed in migration step 12 (ADR-0005 4b, invariant K5). Until then this test pinned the defect: the
    stop was refused at every cycle and entries stayed enabled. The position itself is never touched."""
    from tests.characterization.test_runtime_decisions import _events, _start, _until_position
    from tests.characterization.test_exit_paths import _tick
    world, runtime = _start(tmp_path)
    _until_position(world, runtime)
    btc, session = world["fakes"]["BTCUSDT"], runtime.sessions["BTCUSDT"]
    position = btc.positions["BTCUSDT"]
    assert session.stops.has_active_stop() and session.executor.ready
    original = btc.request_signed

    def refuse_algo_orders(*, method, path, params=None):
        if path == "/fapi/v1/algoOrder" and method == "POST":
            raise BinanceNonRetryableError('HTTP 400: {"code":-4120,"msg":"not supported"}',
                                           http_status=400, code=-4120)
        return original(method=method, path=path, params=params)

    btc.request_signed = refuse_algo_orders
    for stop in btc.algo.values():
        stop["algoStatus"] = "CANCELED"                  # the venue dropped the stop; a new one is refused
    for _ in range(3):
        _tick(world, runtime)
        assert session.executor.ready is False           # entries are blocked without a stop
        assert session.stops.has_active_stop() is False
    issues = [r["issue"] for r in _events(tmp_path, "issue") if r.get("symbol") == "BTCUSDT"]
    assert any("protective stop rejected" in issue for issue in issues)
    assert "BTCUSDT: position without an active protective stop; entries blocked" in issues
    assert btc.positions["BTCUSDT"] == position          # the position itself is left alone
    btc.request_signed = original
    _tick(world, runtime)
    assert session.stops.has_active_stop() and session.executor.ready is True


def test_a_triggered_stop_is_followed_until_the_order_it_became_is_known(tmp_path):
    """R8, fixed in migration step 12. Until then this test pinned the defect: a stop reported triggered before
    its order id was known was marked final at once, its fill never reached the ledger, and the ledger kept a
    position the venue no longer had."""
    fake, executor, consumer, stops = _with_position(tmp_path)
    stops.sync()
    (client_id,) = [k for k, v in fake.algo.items() if v["algoStatus"] == "NEW"]
    fake.algo[client_id]["algoStatus"] = "TRIGGERED"      # status first, actualOrderId still empty
    issues = stops.sync()
    assert issues == ["BTCUSDT: protective stop triggered; the order it became is not known yet"]
    assert stops.unprotected() is True and stops.has_active_stop() is False      # entries get blocked
    fake.algo.pop(client_id)                              # a lookup that finds nothing does not end the wait
    assert stops.sync() == ["BTCUSDT: protective stop triggered; the order it became is not known yet"]
    fake.algo[client_id] = {"algoId": 1, "clientAlgoId": client_id, "symbol": "BTCUSDT", "side": "SELL",
                            "algoStatus": "NEW", "triggerPrice": "96", "actualOrderId": "", "closePosition": "true"}
    fake.trigger_stop(client_id, D("95.9"))              # then the venue closes the position
    assert stops.sync() == ["BTCUSDT: protective stop triggered"]
    assert len(fake.algo) == 1                            # no second stop was placed while waiting
    assert [r["intent_id"] for r in executor.journal.orders(OWNER) if r["intent_id"].startswith("stop-")] == [
        f"stop-{client_id}"]
    _drain(executor, consumer)
    assert fake.positions["BTCUSDT"] == 0 and consumer.position.qty == 0
    assert executor.reconcile_account("BTCUSDT") == []
    assert stops.unprotected() is False


def test_triggered_stop_with_order_id_reaches_the_ledger(tmp_path):
    fake, executor, consumer, stops = _with_position(tmp_path)
    stops.sync()
    (client_id,) = [k for k, v in fake.algo.items() if v["algoStatus"] == "NEW"]
    fake.trigger_stop(client_id, D("95.9"))
    stops.sync()
    _drain(executor, consumer)
    assert consumer.position.qty == 0


def test_no_new_stop_is_placed_in_the_pass_that_adopts_a_triggered_one(tmp_path):
    """Found while fixing R8 (step 12): the ledger still shows the position until the adopted fill is consumed,
    and a stop placed for it then would sit at the venue and could close a later position at a stale price."""
    fake, executor, consumer, stops = _with_position(tmp_path)
    stops.sync()
    (client_id,) = [k for k, v in fake.algo.items() if v["algoStatus"] == "NEW"]
    fake.trigger_stop(client_id, D("95.9"))
    assert stops.sync() == ["BTCUSDT: protective stop triggered"]
    assert list(fake.algo) == [client_id]
    _drain(executor, consumer)
    assert stops.sync() == [] and list(fake.algo) == [client_id] and consumer.position.qty == 0


def test_an_order_with_more_than_100_trades_settles_and_reductions_go_through(tmp_path):
    """R9, fixed in migration step 14b: the adapter reads an order's whole trade history page by page. Until
    then this test pinned the defect: only 100 trades were ever read, the 150-trade order never settled, and
    it blocked the next reduction before placement."""
    fake, adapter, executor, clock, consumer = _executor(tmp_path, SlicedVenue)
    executor.send_order(_order(1, "buy", "0.05"))         # 50 trades
    _drain(executor, consumer)
    assert executor.reconcile_account("BTCUSDT") == []
    executor.send_order(_order(2, "buy", "0.15"))         # 150 trades: all of them are read now
    _drain(executor, consumer)
    assert consumer.position.qty == D("0.2") and fake.positions["BTCUSDT"] == D("0.2")
    assert executor.journal.orders(OWNER, unresolved=True) == []
    executor.send_order(_order(3, "sell", "0.2", reduce_only=True, reason="kill_switch"))   # 200 trades
    _drain(executor, consumer)
    assert fake.positions["BTCUSDT"] == 0 and consumer.position.qty == 0
    assert executor.reconcile_account("BTCUSDT") == []
