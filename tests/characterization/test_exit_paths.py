"""Exit paths of runner.binance_target in execution mode, against the in-memory fake venue."""
from __future__ import annotations

import json
from decimal import Decimal as D

from lab.rebalance import Intent, plan
from lab.strategy import StrategySpec
from runner.binance_target import TargetRuntime
from tests.unit.lab.test_target_runtime import LIVE, _advance, _world
from tests.support.state_db import state_dir_db


def _execution(tmp_path, **spec_kw):
    spec = StrategySpec(lookbacks=(5, 10, 20), vol_lookback=10, target_vol=0.6, band=0.05, **spec_kw)
    world = _world(tmp_path, spec)
    state_dir_db(tmp_path / "exec")   # a state database as init creates it (ADR-0009)
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


def _kill_intents(tmp_path, symbol="BTCUSDT"):
    path = tmp_path / "exec/target_events.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    return [r["order_id"] for r in rows if r["kind"] == "intent" and r["symbol"] == symbol
            and r["reason"] == "kill_switch"]


def test_a_rejected_flatten_is_retried_thirty_seconds_later_within_the_same_bar(tmp_path):
    """R1, fixed in migration step 11 (ADR-0005 1b). Until then this test pinned the defect: one attempt in six
    cycles of the same bar, and the retry only on the next bar, because the order id was fixed per bar."""
    world, runtime = _execution(tmp_path, execution="market")
    btc = world["fakes"]["BTCUSDT"]
    i = _until(world, runtime, lambda: btc.positions.get("BTCUSDT", D(0)) != 0)
    _advance(world, runtime, i)
    position = btc.positions["BTCUSDT"]
    runtime.kill.write(active=True, reason="test", flatten=True)
    posts = len(btc.posts)
    btc.fault, btc.fault_code = "reject", -2022          # the venue rejects the first reduce-only order
    for _ in range(5):
        _tick(world, runtime)                            # 6 s to 30 s: the first attempt, rejected and settled
    assert len(btc.posts) - posts == 1
    assert btc.positions["BTCUSDT"] == position
    _tick(world, runtime)                                # 30 s after the first attempt: a new attempt, a new id
    assert len(btc.posts) - posts == 2
    assert btc.positions["BTCUSDT"] == 0
    first, second = _kill_intents(tmp_path)
    assert (first.rsplit(":", 1)[1], second.rsplit(":", 1)) == ("0", [first.rsplit(":", 1)[0], "1"])


def test_kill_flatten_cancels_a_resting_entry_first_then_closes_the_position(tmp_path):
    """R2, fixed in migration step 11 (ADR-0005 2b, invariant K2). Until then this test pinned the defect: no
    flatten while the entry was pending, and the entry stayed open and kept filling under the kill switch."""
    world, runtime = _execution(tmp_path, execution="maker", ttl_bars=3)
    btc = world["fakes"]["BTCUSDT"]
    _until(world, runtime, lambda: bool(btc.orders))
    order_id = next(iter(btc.orders))
    btc.fill_limit(order_id, D("0.001"))
    _tick(world, runtime)
    runtime.kill.write(active=True, reason="test", flatten=True)
    posts = len(btc.posts)
    _tick(world, runtime)
    assert btc.orders[order_id]["status"] == "CANCELED"                       # the entry is canceled first
    assert btc.positions["BTCUSDT"] == 0                                      # then the filled part is closed
    assert [p.get("reduceOnly") in (True, "true") for p in btc.posts[posts:]] == [True]
    for _ in range(3):
        _tick(world, runtime)
    assert len(btc.posts) == posts + 1 and btc.positions["BTCUSDT"] == 0     # and nothing else goes out


def test_with_equity_unknown_the_execution_session_still_closes_a_position_the_target_no_longer_wants():
    """R13, fixed in migration step 13d (decided 2026-11-15): with equity unknown the execution session plans
    only a whole-position close (target 0 or the other side) and never an entry; see
    tests/unit/trading/test_unknown_equity.py for the runtime. Until then this test pinned the defect: nothing
    at all was planned. The research plan itself is unchanged and still plans nothing without equity."""
    from trading.domain.rebalance import close_without_equity
    common = dict(price=D(100), current_qty=D("0.5"), entry_pending=False, band=0.1, step=D("0.001"),
                  min_qty=D("0.001"), min_notional=D(20))
    assert plan(0.0, equity=D(0), **common) == []
    assert plan(0.0, equity=D(1000), **common) == [Intent("sell", D("0.5"), True)]
    assert close_without_equity(0.0, D("0.5")) == [Intent("sell", D("0.5"), True)]
