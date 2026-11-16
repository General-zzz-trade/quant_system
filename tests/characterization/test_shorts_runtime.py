"""R6 of the runtime extension (docs/arch/runtime-lab-strategies-plan.md, owner 2026-11-16: shorts supported, the spec
decides): a short goes through every part of the execution path with its sign, and a spec that does not allow shorts
never opens one.

The world: BTC and ETH on fake venues, a strategy that wants -0.5 of each symbol's share at every close.
- ``allow_short`` true: each symbol sells to open a short; its protective stop is a BUY above the entry price
  (``entry x (1 + stop)``); a kill switch with flatten buys the whole position back, reduce-only; the account's gross
  exposure counts the short's size; a short whose mark is above its entry is a loss in the unrealized PnL.
- ``allow_short`` false: the same strategy opens nothing; a short already held is bought back to flat.
"""
from __future__ import annotations

import json
from decimal import Decimal as D

import numpy as np
import pandas as pd

from lab.strategy import StrategySpec
from runner.binance_target import TargetRuntime
from tests.support.state_db import state_dir_db
from tests.unit.lab.test_target_runtime import _advance, _world
from trading.application.risk.loss_stop import strategy_unrealized
from trading.domain.strategy.base import PerSymbol

SPEC = StrategySpec(lookbacks=(5, 10, 20), vol_lookback=10, target_vol=0.6, band=0.05, execution="market")


class Short(PerSymbol):
    name = "short"

    def __init__(self, weight=-0.5):
        self.weight = weight

    def window_hours(self):
        return SPEC.window_hours()

    def funding_window(self):
        return SPEC.funding_window()

    def symbol_history(self, window):
        times = window.hourly["available_at"].to_numpy(dtype=np.int64)
        return pd.DataFrame({"available_at": times, "target": self.weight, "ready": True})


def _run(tmp_path, *, allow_short, closes=3, strategy=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    world = _world(tmp_path, SPEC)
    world["spec_path"].write_text(json.dumps({
        "schema": 2, "kind": "target_strategy", "strategy": {"name": "overlay", "params": SPEC.record()},
        "execution": {"band": SPEC.band, "execution": SPEC.execution, "ttl_bars": SPEC.ttl_bars},
        "symbols": ["BTCUSDT", "ETHUSDT"], "allow_short": allow_short, "passed": False}))
    state_dir_db(tmp_path / "exec")
    runtime = TargetRuntime(spec_path=world["spec_path"], state_dir=tmp_path / "exec", mode="execution",
                            market=world["adapter"], account=world["adapter"],
                            clock=lambda: world["clock"]["now"], testnet=True)
    runtime.sessions["BTCUSDT"].gate.inner._limiter._monotonic = lambda: world["clock"]["now"]
    for session in runtime.sessions.values():
        session.strategy = strategy or Short()
    runtime.warmup()
    for i in range(world["warm"], world["warm"] + closes):
        _advance(world, runtime, i)
        for _ in range(2):                           # the next 5-second cycles send what the per-second limit held
            world["clock"]["now"] += 5
            runtime.cycle()
    return world, runtime


def test_a_short_is_opened_protected_counted_and_closed_with_its_sign(tmp_path):
    world, runtime = _run(tmp_path, allow_short=True)
    assert all(s.allow_short for s in runtime.sessions.values())
    for symbol, fake in world["fakes"].items():
        assert fake.trades and fake.trades[0]["side"] == "SELL"
        assert fake.positions[symbol] < 0
        stops = [a for a in fake.algo.values() if a["algoStatus"] == "NEW"]
        assert len(stops) == 1 and stops[0]["side"] == "BUY"
        entry = fake.entry[symbol]
        assert D(stops[0]["triggerPrice"]) > entry
        assert abs(D(stops[0]["triggerPrice"]) / entry - (1 + runtime.config.stop_pct)) < D("0.001")
    _, gross = runtime.sessions["BTCUSDT"].gate.inner._portfolio_reader.exposure()
    held = sum(abs(f.positions[s]) * f.mark(s) for s, f in world["fakes"].items())
    assert held > 0 and abs(gross - held) <= held * D("0.0001")
    btc = world["fakes"]["BTCUSDT"]
    btc.price["BTCUSDT"] = btc.entry["BTCUSDT"] * D("1.01")             # the price rose: the short loses
    ledger = [("BTCUSDT", btc.positions["BTCUSDT"], btc.entry["BTCUSDT"])]    # the short as the ledger holds it
    loss = strategy_unrealized(ledger, world["adapter"].get_position)
    assert btc.positions["BTCUSDT"] < 0 and loss < 0
    runtime.kill.write(active=True, reason="test", flatten=True)
    world["clock"]["now"] += 5
    runtime.cycle()
    for symbol, fake in world["fakes"].items():
        assert fake.positions[symbol] == 0
        closing = fake.posts[-1]
        assert closing["side"] == "BUY" and closing.get("reduceOnly") in ("true", True)


def test_a_spec_without_shorts_opens_none_and_closes_one_it_holds(tmp_path):
    world, runtime = _run(tmp_path, allow_short=False)
    assert all(not s.allow_short for s in runtime.sessions.values())
    assert all(not fake.trades for fake in world["fakes"].values())
    assert all(s.last_target == (0.0, True) for s in runtime.sessions.values())


def test_a_short_held_when_shorts_are_not_allowed_is_bought_back(tmp_path):
    world, runtime = _run(tmp_path / "short", allow_short=True, closes=2)
    held = {s: f.positions[s] for s, f in world["fakes"].items()}
    assert all(q < 0 for q in held.values())
    for session in runtime.sessions.values():
        session.allow_short = False                  # as a new spec without shorts would set it
    i = world["warm"] + 2
    _advance(world, runtime, i)
    for _ in range(2):
        world["clock"]["now"] += 5
        runtime.cycle()
    for symbol, fake in world["fakes"].items():
        assert fake.positions[symbol] == 0 and fake.trades[-1]["side"] == "BUY"
