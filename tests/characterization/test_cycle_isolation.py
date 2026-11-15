"""S3c (ADR-0013 point 4, design section 10, tech lead 2026-11-15): the market, exit and status blocks of a cycle are
isolated. If one raises, the other two still run in that cycle; the failure blocks every entry until that block works
again, and it is written to the event log. One symbol's failing decision blocks only that symbol."""
from __future__ import annotations

import json
from decimal import Decimal as D

from tests.characterization.test_exit_paths import _execution, _until
from tests.unit.lab.test_target_runtime import _advance


def _issues(tmp_path, kind):
    rows = [json.loads(line) for line in (tmp_path / "exec/target_events.jsonl").read_text().splitlines()]
    return [r for r in rows if r["kind"] == "issue" and r.get("issue_kind") == kind]


def _posts(world):
    return [len(f.posts) for f in world["fakes"].values()]


def _status(tmp_path):
    return json.loads((tmp_path / "exec/target_status.json").read_text())


def _first_entry_bar(tmp_path):
    """The bar index at which the strategy first enters BTC, found on an untouched copy of the world."""
    (tmp_path / "probe").mkdir()
    world, runtime = _execution(tmp_path / "probe", execution="market")
    btc = world["fakes"]["BTCUSDT"]
    return _until(world, runtime, lambda: btc.positions.get("BTCUSDT", D(0)) != 0) - 1


def test_a_failing_market_side_still_runs_the_exits_and_the_status(tmp_path, monkeypatch):
    world, runtime = _execution(tmp_path, execution="market")
    btc = world["fakes"]["BTCUSDT"]
    i = _until(world, runtime, lambda: btc.positions.get("BTCUSDT", D(0)) != 0)

    def broken(*args, **kwargs):
        raise RuntimeError("an unexpected market-side error")
    # A kline error is one symbol's failure since the S3 audit (finding 4); this one fails the side as a whole.
    monkeypatch.setattr(runtime.sessions["BTCUSDT"], "retry_deferred", broken)
    runtime.kill.write(active=True, reason="test", flatten=True)
    _advance(world, runtime, i)                                       # must not raise
    assert btc.positions["BTCUSDT"] == 0                              # the exit side ran in the same cycle
    assert _status(tmp_path)["at"] == world["clock"]["now"]          # and so did the status
    assert _issues(tmp_path, "cycle_failed")
    assert not any(s.executor.ready for s in runtime.sessions.values())


def test_a_failing_exit_side_still_processes_the_bar_and_the_status_but_enters_nothing(tmp_path, monkeypatch):
    entry = _first_entry_bar(tmp_path)
    world, runtime = _execution(tmp_path, execution="market")
    for i in range(world["warm"], entry):
        _advance(world, runtime, i)
    cursor, posts = runtime.sessions["BTCUSDT"].cursor, _posts(world)

    def unreadable():
        raise RuntimeError("kill state unreadable")
    monkeypatch.setattr(runtime.kill, "state", unreadable)
    _advance(world, runtime, entry)                                   # must not raise
    assert runtime.sessions["BTCUSDT"].cursor > cursor               # the market side processed the bar
    assert _posts(world) == posts                                     # but entered nothing
    assert _status(tmp_path)["at"] == world["clock"]["now"]
    assert _issues(tmp_path, "cycle_failed")


def test_a_failing_status_keeps_trading_safe_and_blocks_entries_until_it_works_again(tmp_path):
    world, runtime = _execution(tmp_path, execution="market")
    btc = world["fakes"]["BTCUSDT"]
    i = _until(world, runtime, lambda: btc.positions.get("BTCUSDT", D(0)) != 0)
    status = tmp_path / "exec/target_status.json"
    status.unlink()
    status.mkdir()                                                    # the status file cannot be written
    runtime.kill.write(active=True, reason="test", flatten=True)
    _advance(world, runtime, i)                                       # must not raise
    assert btc.positions["BTCUSDT"] == 0                              # exits and market ran
    assert _issues(tmp_path, "cycle_failed")
    runtime.kill.write(active=False, reason="cleared")
    _advance(world, runtime, i + 1)
    assert not any(s.executor.ready for s in runtime.sessions.values())   # still failing: still blocked
    status.rmdir()
    _advance(world, runtime, i + 2)                                   # works again: written ...
    _advance(world, runtime, i + 3)                                   # ... and the next cycle unblocks
    assert status.is_file() and all(s.executor.ready for s in runtime.sessions.values())


def test_one_symbol_s_failing_decision_blocks_only_that_symbol(tmp_path, monkeypatch):
    world, runtime = _execution(tmp_path, execution="market")
    eth = runtime.sessions["ETHUSDT"]

    def broken():
        raise RuntimeError("ETH decision failed")
    monkeypatch.setattr(eth, "_target", broken)
    btc = world["fakes"]["BTCUSDT"]
    i = _until(world, runtime, lambda: btc.positions.get("BTCUSDT", D(0)) != 0)   # BTC still trades
    assert world["fakes"]["ETHUSDT"].posts == []
    assert eth.cursor == runtime.sessions["BTCUSDT"].cursor           # ETH bars are still taken in
    assert {r["symbol"] for r in _issues(tmp_path, "decision_failed")} == {"ETHUSDT"}
    _advance(world, runtime, i)
    assert eth.executor.ready is False and runtime.sessions["BTCUSDT"].executor.ready is True


def test_an_exit_side_that_failed_blocks_entries_on_the_next_cycle_s_market_side(tmp_path, monkeypatch):
    # The market side runs before the exit side, so the block must be in place before the next decision.
    entry = _first_entry_bar(tmp_path)
    world, runtime = _execution(tmp_path, execution="market")
    for i in range(world["warm"], entry - 1):
        _advance(world, runtime, i)

    def broken():
        raise RuntimeError("exit side broken")
    monkeypatch.setattr(runtime.exits, "risk_actions", broken)       # the gate itself reads nothing broken
    _advance(world, runtime, entry - 1)
    posts = _posts(world)
    _advance(world, runtime, entry)                                   # the bar the strategy would enter on
    assert _posts(world) == posts


def test_a_symbol_whose_decisions_work_again_may_enter_again(tmp_path, monkeypatch):
    world, runtime = _execution(tmp_path, execution="market")
    eth = runtime.sessions["ETHUSDT"]
    working = eth._target

    def broken():
        raise RuntimeError("ETH decision failed")
    monkeypatch.setattr(eth, "_target", broken)
    i = world["warm"]
    for i in range(world["warm"], world["warm"] + 3):
        _advance(world, runtime, i)
    assert eth.executor.ready is False
    monkeypatch.setattr(eth, "_target", working)
    _advance(world, runtime, i + 1)                                   # a bar succeeds ...
    _advance(world, runtime, i + 2)                                   # ... and the next reconciliation unblocks
    assert eth.executor.ready is True


def test_paper_mode_raises_a_failing_decision_instead_of_skipping_it(tmp_path, monkeypatch):
    import pytest
    from runner.binance_target import TargetRuntime
    from tests.unit.lab.test_target_runtime import _world
    from trading.domain.strategy.overlay import StrategySpec
    world = _world(tmp_path, StrategySpec(lookbacks=(5, 10, 20), vol_lookback=10, target_vol=0.6, band=0.05))
    runtime = TargetRuntime(spec_path=world["spec_path"], state_dir=tmp_path / "paper", mode="paper",
                            market=world["adapter"], clock=lambda: world["clock"]["now"])
    runtime.warmup()

    def broken():
        raise RuntimeError("paper decision failed")
    monkeypatch.setattr(runtime.sessions["BTCUSDT"], "_target", broken)
    with pytest.raises(RuntimeError, match="paper decision failed"):
        _advance(world, runtime, world["warm"])
