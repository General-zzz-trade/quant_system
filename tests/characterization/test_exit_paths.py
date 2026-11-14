"""Exit paths of runner.binance_target in execution mode, against the in-memory fake venue."""
from __future__ import annotations

from decimal import Decimal as D

import pytest

from lab.rebalance import Intent, plan
from lab.strategy import StrategySpec
from runner.binance_target import TargetRuntime
from tests.unit.lab.test_target_runtime import LIVE, _advance, _world


def _execution(tmp_path, **spec_kw):
    spec = StrategySpec(lookbacks=(5, 10, 20), vol_lookback=10, target_vol=0.6, band=0.05, **spec_kw)
    world = _world(tmp_path, spec)
    runtime = TargetRuntime(spec_path=world["spec_path"], state_dir=tmp_path / "exec", mode="execution",
                            market=world["adapter"], account=world["adapter"],
                            clock=lambda: world["clock"]["now"], testnet=True)
    runtime.warmup()
    return world, runtime


def _until(world, runtime, condition) -> int:
    """Advance one closed bar at a time until ``condition()``; returns the next bar index."""
    i = world["warm"]
    while not condition():
        assert i < world["warm"] + LIVE - 10, "condition never reached"
        _advance(world, runtime, i)
        i += 1
    return i


def _tick(world, runtime, seconds: float = 6) -> None:
    """One runtime cycle inside the same hourly bar."""
    world["clock"]["now"] += seconds
    runtime.cycle()


def test_kill_flatten_closes_the_position_when_the_venue_accepts(tmp_path):
    world, runtime = _execution(tmp_path, execution="market")
    btc = world["fakes"]["BTCUSDT"]
    i = _until(world, runtime, lambda: btc.positions.get("BTCUSDT", D(0)) != 0)
    _advance(world, runtime, i)
    runtime.kill.write(active=True, reason="test", flatten=True)
    _tick(world, runtime)
    assert btc.positions["BTCUSDT"] == 0
    assert runtime.sessions["BTCUSDT"].position_qty() == 0


@pytest.mark.known_defect("R1")
def test_current_rejected_flatten_is_not_retried_within_the_same_bar(tmp_path):
    world, runtime = _execution(tmp_path, execution="market")
    btc = world["fakes"]["BTCUSDT"]
    i = _until(world, runtime, lambda: btc.positions.get("BTCUSDT", D(0)) != 0)
    _advance(world, runtime, i)
    i += 1
    position = btc.positions["BTCUSDT"]
    runtime.kill.write(active=True, reason="test", flatten=True)
    posts = len(btc.posts)
    btc.fault, btc.fault_code = "reject", -1008          # the venue rejects the first reduce-only order
    for _ in range(6):
        _tick(world, runtime)
    assert len(btc.posts) - posts == 1                  # one attempt in six cycles of the same bar
    assert btc.positions["BTCUSDT"] == position
    _advance(world, runtime, i)                          # next bar: new order id, flatten succeeds
    assert len(btc.posts) - posts == 2
    assert btc.positions["BTCUSDT"] == 0


@pytest.mark.known_defect("R2")
def test_current_kill_flatten_waits_for_a_resting_entry_which_keeps_filling(tmp_path):
    world, runtime = _execution(tmp_path, execution="maker", ttl_bars=3)
    btc = world["fakes"]["BTCUSDT"]
    _until(world, runtime, lambda: bool(btc.orders))
    order_id = next(iter(btc.orders))
    btc.fill_limit(order_id, D("0.001"))
    _tick(world, runtime)
    runtime.kill.write(active=True, reason="test", flatten=True)
    posts = len(btc.posts)
    for _ in range(3):
        _tick(world, runtime)
    assert len(btc.posts) == posts                      # no flatten order while the entry is pending
    assert btc.orders[order_id]["status"] == "PARTIALLY_FILLED"   # and the entry is not canceled
    before = abs(btc.positions["BTCUSDT"])
    btc.fill_limit(order_id, D("0.002"))
    _tick(world, runtime)
    assert abs(btc.positions["BTCUSDT"]) > before       # exposure grows under an active kill switch
    assert len(btc.posts) == posts


@pytest.mark.known_defect("R13")
def test_current_plan_emits_no_reduction_when_equity_is_unknown():
    common = dict(price=D(100), current_qty=D("0.5"), entry_pending=False, band=0.1, step=D("0.001"),
                  min_qty=D("0.001"), min_notional=D(20))
    assert plan(0.0, equity=D(0), **common) == []
    assert plan(0.0, equity=D(1000), **common) == [Intent("sell", D("0.5"), True)]
