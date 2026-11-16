"""R4 in the running runtime: each symbol's funding interval comes from the exchange at warm-up; a symbol the exchange
does not list takes no entries, and the other symbol trades exactly as when both are listed."""
from __future__ import annotations

from lab.strategy import StrategySpec
from runner.binance_target import TargetRuntime
from tests.support.state_db import state_dir_db
from tests.unit.lab.test_target_runtime import _advance, _world

SPEC = StrategySpec(lookbacks=(5, 10, 20), vol_lookback=10, target_vol=0.6, band=0.05, execution="market")


def _run(tmp_path, hours):
    tmp_path.mkdir(parents=True, exist_ok=True)
    world = _world(tmp_path, SPEC)
    for fake in world["fakes"].values():
        fake.funding_hours = dict(hours)
    state_dir_db(tmp_path / "exec")
    runtime = TargetRuntime(spec_path=world["spec_path"], state_dir=tmp_path / "exec", mode="execution",
                            market=world["adapter"], account=world["adapter"],
                            clock=lambda: world["clock"]["now"], testnet=True)
    # The orders-per-second limit on the world's clock, not the real one (see test_decide_together_runtime.py).
    runtime.sessions["BTCUSDT"].gate.inner._limiter._monotonic = lambda: world["clock"]["now"]
    runtime.warmup()
    runtime.warm_intervals = {s: session.funding_interval for s, session in runtime.sessions.items()}
    for i in range(world["warm"], world["warm"] + 40):
        _advance(world, runtime, i)
    trades = {s: [(t["side"], t["qty"], t["price"]) for t in f.trades] for s, f in world["fakes"].items()}
    return runtime, trades


def test_a_symbol_without_a_funding_interval_takes_no_entries(tmp_path):
    runtime, both = _run(tmp_path / "both", {"BTCUSDT": 8, "ETHUSDT": 8})
    assert both["BTCUSDT"] and both["ETHUSDT"]
    assert runtime.reporter.status()["symbols"]["ETHUSDT"]["funding_interval_hours"] == 8
    assert runtime.warm_intervals == {"BTCUSDT": 8 * 3_600_000, "ETHUSDT": 8 * 3_600_000}   # known before any bar
    runtime, one = _run(tmp_path / "one", {"BTCUSDT": 8})
    assert one["ETHUSDT"] == [] and one["BTCUSDT"] == both["BTCUSDT"]
    eth = runtime.reporter.status()["symbols"]["ETHUSDT"]
    assert eth["funding_interval_hours"] is None and eth["entry_reason"] == "funding interval not known"
    assert runtime.sessions["BTCUSDT"].funding_interval == 8 * 3_600_000
