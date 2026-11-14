"""Decisions of the target runtime in execution mode, pinned before migration step 10 moves it.

Plan 0.6 (step 10): a latched drawdown flattens, a breached stop flattens, entries do not stack, a late
bar blocks entries, stale equity sizes to zero; and the entry gate as the runtime really assembles it
(step-6 audit L1).
"""
from __future__ import annotations

import json
from decimal import Decimal as D

from lab.orders import make_order
from lab.rebalance import plan
from lab.strategy import StrategySpec
from runner.binance_main import EquitySource
from runner.binance_target import TargetRuntime
from tests.unit.binance.fake_venue import HOUR
from tests.unit.lab.test_target_runtime import LIVE, _advance, _world

MARKET = dict(lookbacks=(5, 10, 20), vol_lookback=10, target_vol=0.6, band=0.05)


def _runtime(tmp_path, world):
    return TargetRuntime(spec_path=world["spec_path"], state_dir=tmp_path / "exec", mode="execution",
                         market=world["adapter"], account=world["adapter"], clock=lambda: world["clock"]["now"],
                         testnet=True)


def _start(tmp_path, execution="market", **spec):
    tmp_path.mkdir(parents=True, exist_ok=True)
    world = _world(tmp_path, StrategySpec(**MARKET, execution=execution, **spec))
    runtime = _runtime(tmp_path, world)
    runtime.warmup()
    return world, runtime


def _until_position(world, runtime, symbol="BTCUSDT"):
    fake, i = world["fakes"][symbol], world["warm"]
    while fake.positions.get(symbol, D(0)) == 0 and i < world["warm"] + LIVE - 20:
        _advance(world, runtime, i)
        i += 1
    assert fake.positions.get(symbol, D(0)) != 0
    return i


def _events(tmp_path, kind):
    rows = [json.loads(line) for line in (tmp_path / "exec/target_events.jsonl").read_text().splitlines()]
    return [r for r in rows if r["kind"] == kind]


def test_a_latched_drawdown_flattens_every_position_and_blocks_new_entries(tmp_path):
    world, runtime = _start(tmp_path)
    i = _until_position(world, runtime)
    for fake in world["fakes"].values():
        fake.equity = D("7000")                                  # -30%: past the daily and the weekly limit
    for k in range(3):
        _advance(world, runtime, i + k)
    assert runtime.drawdown.latched()
    assert all(f.positions.get(s, D(0)) == 0 for s, f in world["fakes"].items())
    assert {r["reason"] for r in _events(tmp_path, "intent")} >= {"drawdown_latched"}
    posts = sum(len(f.posts) for f in world["fakes"].values())
    for k in range(3, 9):
        _advance(world, runtime, i + k)
    assert sum(len(f.posts) for f in world["fakes"].values()) == posts          # no entry while latched
    assert any(r["reason"].startswith("drawdown") for r in _events(tmp_path, "blocked"))


def test_a_stop_that_would_trigger_at_once_flattens_the_position(tmp_path):
    world, runtime = _start(tmp_path)
    i = _until_position(world, runtime)
    _advance(world, runtime, i)                                   # the exchange-side stop is in place
    btc, session = world["fakes"]["BTCUSDT"], runtime.sessions["BTCUSDT"]
    qty, entry = session.consumer.position.qty, session.consumer.position.entry_price
    assert session.stops.has_active_stop()
    for stop in btc.algo.values():
        stop["algoStatus"] = "CANCELED"                           # the venue dropped the stop ...
    btc.price["BTCUSDT"] = entry * (D("0.85") if qty > 0 else D("1.15"))   # ... and price is already beyond it
    _advance(world, runtime, i + 1)
    assert btc.positions["BTCUSDT"] == 0 and session.position_qty() == 0
    assert any(r["reason"] == "stop_breached" for r in _events(tmp_path, "intent"))
    assert any("stop_breached" in r["issue"] for r in _events(tmp_path, "issue"))
    # Current behaviour, found while writing this test and not in the R1-R22 list: nothing remembers the
    # breach, so the next bar's rebalance opens the same position again. Whether that is wanted is an open
    # question for the stop-handling step (12); this assertion changes only with that decision.
    _advance(world, runtime, i + 2)
    assert btc.positions["BTCUSDT"] == qty and _events(tmp_path, "intent")[-1]["reason"] == "rebalance"


def test_resting_entries_never_stack(tmp_path):
    world, runtime = _start(tmp_path, execution="maker", ttl_bars=4)
    resting_cycles = 0
    for i in range(world["warm"], world["warm"] + 120):
        _advance(world, runtime, i)
        for fake in world["fakes"].values():
            open_entries = [o for o in fake.orders.values() if o["status"] in ("NEW", "PARTIALLY_FILLED")
                            and o["type"] == "LIMIT"]
            assert len(open_entries) <= 1
            resting_cycles += len(open_entries)
    assert resting_cycles >= 4                                    # entries did rest across several bars


def test_a_bar_processed_more_than_ten_minutes_late_cannot_open_a_position(tmp_path):
    reference, runtime = _start(tmp_path / "a")
    first_entry = _until_position(reference, runtime) - 1        # the bar whose close opened the position
    world, runtime = _start(tmp_path / "b")
    for i in range(world["warm"], first_entry):
        _advance(world, runtime, i)
    btc = world["fakes"]["BTCUSDT"]
    assert btc.posts == []
    world["clock"]["now"] = (world["rows"]["BTCUSDT"][first_entry]["start"] + HOUR + 11 * 60_000) / 1000
    runtime.cycle()
    assert btc.posts == [] and runtime.sessions["BTCUSDT"].entry_ready is False
    blocked = [r for r in _events(tmp_path / "b", "blocked") if r["symbol"] == "BTCUSDT"]
    assert blocked and blocked[-1]["reason"] == "inputs: bar processed too late"
    for i in range(first_entry + 1, first_entry + 4):            # on time again: the entry goes through
        _advance(world, runtime, i)
    assert btc.posts


def test_stale_or_unknown_equity_sizes_to_zero():
    clock = {"t": 1000.0}
    equity = EquitySource(lambda: clock["t"])
    assert equity.current() == 0                                  # never refreshed
    equity.update(D("5000"))
    assert equity.current() == D("5000")
    clock["t"] += 120.0
    assert equity.current() == D("5000")
    clock["t"] += 0.1
    assert equity.current() == 0                                  # older than two minutes
    equity.update(D("-1"))
    assert equity.current() == 0
    sizing = dict(price=D("100"), current_qty=D(0), entry_pending=False, band=0.05, step=D("0.001"),
                  min_qty=D("0.001"), min_notional=D("5"))
    assert plan(0.5, equity=D(0), **sizing) == [] and len(plan(0.5, equity=D("5000"), **sizing)) == 1


def _order(reduce_only, qty="0.01"):
    return make_order(order_id="BTCUSDT:1:test:0", source="target.BTCUSDT", symbol="BTCUSDT", side="buy",
                      qty=D(qty), price=D("30000"), reduce_only=reduce_only, reason="test")


def test_the_entry_gate_as_the_runtime_assembles_it(tmp_path):
    world, runtime = _start(tmp_path)
    session = runtime.sessions["BTCUSDT"]
    session.entry_ready, session.entry_reason = True, ""
    gate = session.gate
    assert gate.check(_order(False)).allowed
    assert gate.check(_order(False, qty="10")).allowed is False              # 300,000 USDT on 10,000 of equity
    session.entry_ready, session.entry_reason = False, "bar processed too late"
    assert gate.check(_order(False)).reason == "inputs: bar processed too late"
    assert gate.check(_order(True)).allowed                                   # a reduction needs no fresh inputs
    session.entry_ready = True
    runtime.kill.write(active=True, reason="test", flatten=False)
    assert gate.check(_order(False)).reason.startswith("kill_switch")
    assert gate.check(_order(True)).allowed and gate.check(_order(False, qty="0")).allowed
    assert gate.check(_order("yes")).allowed is False                         # only the literal True reduces
    runtime.drawdown.invalidate("test")
    assert gate.check(_order(False)).reason.startswith("kill_switch")         # the kill switch is asked first
    runtime.kill.write(active=False, reason="", flatten=False)
    assert gate.check(_order(False)).reason.startswith("drawdown")
    assert gate.check(_order(True)).allowed
