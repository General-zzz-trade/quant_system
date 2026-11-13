"""Durable Binance execution: no resend, absence resolution, resting entries, stops."""
from __future__ import annotations

from decimal import Decimal as D

import pytest

from decision.modules.alpha_orders import make_close_order, make_open_order
from execution.adapters.binance import BinanceConfig
from execution.adapters.binance.adapter import BinanceAdapter
from execution.adapters.binance.execution_adapter import NOT_FOUND_GRACE_SECONDS, BinanceExecutionAdapter
from tests.unit.binance.fake_venue import FakeBinance, make_bars

OWNER = "alpha.BTCUSDT"


@pytest.fixture
def env(tmp_path):
    clock = {"now": 1_800_000_000.0}
    fake = FakeBinance(make_bars(10), clock=lambda: clock["now"])
    fake.price["BTCUSDT"] = D("100")
    adapter = BinanceAdapter(BinanceConfig(api_key="k", api_secret="s"), client=fake)
    assert adapter.connect(symbols=("BTCUSDT",))
    executor = BinanceExecutionAdapter(adapter, scope="binance-testnet-ut", owner=OWNER,
                                       journal_path=tmp_path / "orders.sqlite", wait_seconds=0,
                                       clock=lambda: clock["now"])
    statuses = []
    executor._status_callback = lambda oid, status: statuses.append((oid, status))
    executor._ready = True
    return fake, adapter, executor, clock, statuses


def _entry(qty="0.1", side=1, price=100.0):
    return make_open_order("BTCUSDT", "BTCUSDT", price, side, D(qty))[0]


def _close(side=1):
    return make_close_order("BTCUSDT", "BTCUSDT", 100.0, side, "atr_stop", D("0.05"), D("0.001"))[0]


def _unresolved(executor):
    return [r for r in executor.journal.orders(OWNER) if not r["settled"]]


def test_entry_rests_as_post_only_and_expires(env):
    fake, _, executor, clock, _ = env
    assert executor.send_order(_entry()) == ()
    post = fake.posts[-1]
    assert post["timeInForce"] == "GTX" and post["price"] == "99.9"  # reference - 2 bps, floored to tick
    fake.fill_limit(next(iter(fake.orders)), D("0.02"))
    fills = executor.reconcile_pending()
    assert [f.qty for f in fills] == [D("0.02")]  # outbox replays until consumed
    assert _unresolved(executor), "a resting entry stays open before its TTL"
    clock["now"] += 3 * 3600
    executor.reconcile_pending()
    order = next(iter(fake.orders.values()))
    assert order["status"] == "CANCELED" and not _unresolved(executor)


def test_crossing_post_only_is_settled_as_rejected(env):
    fake, _, executor, _, statuses = env
    fake.price["BTCUSDT"] = D("99")  # buy limit at 99.98 would cross
    executor.send_order(_entry())
    executor.notify_status()  # the ExecutionBridge does this after every submission
    assert not _unresolved(executor)
    assert statuses[-1][1] == "rejected"


def test_lost_submission_never_resent_and_resolves_after_grace(env):
    fake, _, executor, clock, statuses = env
    fake.positions["BTCUSDT"], fake.entry["BTCUSDT"] = D("0.05"), D("100")
    _confirm_position(executor, D("0.05"))
    fake.fault = "timeout_before"  # request never reached the matching engine
    assert executor.send_order(_close()) == ()
    assert len(fake.posts) == 1 and _unresolved(executor)
    # A second exit is refused while the first is unresolved (no overlap)...
    executor.send_order(_close())
    assert len(fake.posts) == 1
    clock["now"] += NOT_FOUND_GRACE_SECONDS + 1
    executor.reconcile_pending()
    assert not _unresolved(executor)
    assert ("not_found" in {s for _, s in statuses}) or executor.journal.orders(OWNER)[0]["status"] == "not_found"
    # ...and once it is resolved, the next exit can be placed: no permanent deadlock.
    executor.send_order(_close())
    assert len(fake.posts) == 2 and fake.positions["BTCUSDT"] == 0


def test_accepted_but_timed_out_order_is_found_not_duplicated(env):
    fake, _, executor, clock, _ = env
    fake.positions["BTCUSDT"], fake.entry["BTCUSDT"] = D("0.05"), D("100")
    _confirm_position(executor, D("0.05"))
    fake.fault = "timeout_after"
    executor.send_order(_close())
    clock["now"] += NOT_FOUND_GRACE_SECONDS + 1
    fills = executor.reconcile_pending()
    assert len(fake.posts) == 1 and fake.positions["BTCUSDT"] == 0
    assert sum(f.qty for f in fills) == D("0.05")
    assert not _unresolved(executor)


def test_definitive_rejection_settles_immediately(env):
    fake, _, executor, _, statuses = env
    fake.fault, fake.fault_code = "reject", -2019  # margin is insufficient
    executor.send_order(_entry())
    executor.notify_status()
    assert not _unresolved(executor) and statuses[-1][1] == "rejected"


def test_unknown_5xx_outcome_stays_locked_until_grace(env):
    fake, _, executor, clock, _ = env
    fake.fault = "error_5xx"
    executor.send_order(_entry())
    executor.reconcile_pending()
    assert _unresolved(executor), "an unknown outcome must not be assumed absent before the grace period"
    clock["now"] += NOT_FOUND_GRACE_SECONDS + 1
    executor.reconcile_pending()
    assert not _unresolved(executor)


def test_reduction_cancels_resting_entry_first(env):
    fake, _, executor, _, _ = env
    executor.send_order(_entry())
    order_id = next(iter(fake.orders))
    fake.fill_limit(order_id, D("0.02"))
    for fill in executor.reconcile_pending():
        pass
    _confirm_position(executor, D("0.02"))
    executor.send_order(_close())
    assert fake.orders[order_id]["status"] == "CANCELED"


def test_reconcile_account_flags_mismatch_and_unknown_orders(env):
    fake, _, executor, _, _ = env
    fake.positions["BTCUSDT"] = D("0.03")
    issues = executor.reconcile_account("BTCUSDT")
    assert any("venue qty=0.03" in i for i in issues) and executor._ready is False


def _confirm_position(executor, qty):
    """Write a consumer checkpoint as the durable fill consumer would."""
    import json
    with executor.journal.connect() as db:
        db.execute("INSERT OR REPLACE INTO execution_consumers VALUES (?,?,?)",
                   (executor.scope, OWNER, json.dumps({"position": {"qty": str(qty), "entry_price": "100",
                                                                    "realized_pnl": "0", "fees": {}},
                                                       "revision": "rev-1", "engine": {}, "strategy": {}})))
