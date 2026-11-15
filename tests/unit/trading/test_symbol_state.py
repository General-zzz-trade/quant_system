"""ADR-0014 point 3 (S3d): every symbol is in one state, derived each cycle from that cycle's issues and the kill
switch, drawdown and pause; the allowed transitions are a fixed table; only READY lets an entry through, every
state lets a reduction through; every change is an event and the status shows the state."""
from __future__ import annotations

import json
from decimal import Decimal as D
from types import SimpleNamespace

import pytest

from tests.characterization.test_exit_paths import _execution, _tick, _until
from trading.application.risk.readiness import EntryReadinessGate
from trading.domain.symbol_state import ALLOWED, StateInputs, SymbolState as S, classify

ALL = list(S)


@pytest.fixture(autouse=True)
def _offline_env(monkeypatch):
    for name in ("QUANT_MODEL_SIGN_KEY", "BINANCE_TESTNET", "BINANCE_EXCHANGE_LEVERAGE", "BINANCE_MARGIN_TYPE",
                 "MAX_DAILY_DRAWDOWN_PCT", "MAX_WEEKLY_DRAWDOWN_PCT", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("QUANT_ALLOW_UNSIGNED_MODELS", "1")
    monkeypatch.setenv("RISK_SCOPE", "binance-testnet-ut")


def _inputs(**over):
    base = dict(warmed=True, flattening=False, halted=False, paused=False, blocked=False)
    base.update(over)
    return StateInputs(**base)


@pytest.mark.parametrize("over,state", [
    ({}, S.READY),
    ({"blocked": True}, S.BLOCKED),
    ({"paused": True, "blocked": True}, S.PAUSED),
    ({"halted": True, "paused": True, "blocked": True}, S.HALTED),
    ({"flattening": True, "halted": True, "paused": True}, S.FLATTENING),
    ({"warmed": False, "flattening": True, "halted": True}, S.WARMING),
])
def test_the_state_follows_a_fixed_priority(over, state):
    assert classify(_inputs(**over)) is state


def test_the_allowed_transitions_are_exactly_the_decided_ones():
    # Anything may follow anything, except that nothing goes back to WARMING once warmed up: the bars of a running
    # process are never thrown away. Adding or removing a row fails here.
    expected = {(a, b) for a in ALL for b in ALL if b is not S.WARMING or a is S.WARMING}
    assert ALLOWED == frozenset(expected)
    assert len(ALLOWED) == 31


@pytest.mark.parametrize("state", ALL)
def test_only_ready_lets_an_entry_through_and_every_state_lets_a_reduction_through(state):
    inner = SimpleNamespace(check=lambda event: SimpleNamespace(allowed=True, reason="ok"))
    gate = EntryReadinessGate(SimpleNamespace(entry_ready=True, entry_reason="", state=state), inner)
    entry = SimpleNamespace(qty=D("1"), reduce_only=False)
    reduction = SimpleNamespace(qty=D("1"), reduce_only=True)
    assert gate.check(entry).allowed is (state is S.READY)
    assert gate.check(reduction).allowed is True
    if state is not S.READY:
        assert gate.check(entry).reason == f"state: {state.value}"


# -- in the runtime -----------------------------------------------------------------------------------------------
def _events(tmp_path, kind):
    rows = [json.loads(line) for line in (tmp_path / "exec/target_events.jsonl").read_text().splitlines()]
    return [r for r in rows if r["kind"] == kind]


def _status(tmp_path):
    return json.loads((tmp_path / "exec/target_status.json").read_text())


def test_a_warmed_up_runtime_without_problems_is_ready_and_says_so(tmp_path):
    world, runtime = _execution(tmp_path, execution="market")
    _tick(world, runtime)
    assert {s: runtime.sessions[s].state for s in runtime.sessions} == {"BTCUSDT": S.READY, "ETHUSDT": S.READY}
    assert {s: v["state"] for s, v in _status(tmp_path)["symbols"].items()} == {"BTCUSDT": "ready", "ETHUSDT": "ready"}
    assert {(r["symbol"], r["previous"], r["state"]) for r in _events(tmp_path, "symbol_state")} == {
        ("BTCUSDT", "warming", "ready"), ("ETHUSDT", "warming", "ready")}


def test_the_kill_switch_flattens_then_halts_and_a_cleared_kill_makes_it_ready_again(tmp_path):
    world, runtime = _execution(tmp_path, execution="market")
    btc = world["fakes"]["BTCUSDT"]
    _until(world, runtime, lambda: btc.positions.get("BTCUSDT", D(0)) != 0)
    session = runtime.sessions["BTCUSDT"]
    runtime.kill.write(active=True, reason="test", flatten=True)
    _tick(world, runtime)
    assert btc.positions["BTCUSDT"] == 0
    _tick(world, runtime)
    assert session.state is S.HALTED and runtime.sessions["ETHUSDT"].state is S.HALTED
    runtime.kill.write(active=False, reason="cleared")
    _tick(world, runtime)
    assert session.state is S.READY
    trail = [r["state"] for r in _events(tmp_path, "symbol_state") if r["symbol"] == "BTCUSDT"]
    assert "flattening" in trail and trail[-2:] == ["halted", "ready"]


def test_a_blocking_problem_makes_the_symbol_blocked(tmp_path):
    world, runtime = _execution(tmp_path, execution="market")
    world["fakes"]["ETHUSDT"].positions["ETHUSDT"] = D("0.5")         # a position the ledger does not hold
    _tick(world, runtime)
    assert runtime.sessions["ETHUSDT"].state is S.BLOCKED and runtime.sessions["BTCUSDT"].state is S.READY


def test_a_forbidden_transition_blocks_the_symbol_and_is_alerted(tmp_path):
    world, runtime = _execution(tmp_path, execution="market")
    _tick(world, runtime)
    session = runtime.sessions["BTCUSDT"]
    session.bars.clear()                                              # "not warmed up" while running: never allowed
    _tick(world, runtime)
    assert session.state is S.BLOCKED
    assert any("ready -> warming" in r["reason"] for r in _events(tmp_path, "alert"))


def test_a_latched_drawdown_halts_every_symbol(tmp_path):
    from tests.unit.lab.test_target_runtime import _advance
    world, runtime = _execution(tmp_path, execution="market")
    btc = world["fakes"]["BTCUSDT"]
    i = _until(world, runtime, lambda: btc.positions.get("BTCUSDT", D(0)) != 0)
    for fake in world["fakes"].values():
        fake.equity = D("7000")                                       # -30%: past the daily and the weekly limit
    for k in range(3):
        _advance(world, runtime, i + k)
    assert runtime.drawdown.latched()
    assert {s.state for s in runtime.sessions.values()} == {S.HALTED}


def test_a_failing_side_makes_every_symbol_blocked(tmp_path):
    world, runtime = _execution(tmp_path, execution="market")
    _tick(world, runtime)
    status = tmp_path / "exec/target_status.json"
    status.unlink()
    status.mkdir()                                                    # the status side fails from now on
    _tick(world, runtime)
    _tick(world, runtime)
    assert {s.state for s in runtime.sessions.values()} == {S.BLOCKED}


def test_a_position_that_cannot_be_read_still_counts_as_held_while_flattening():
    from trading.application.exit_supervisor import ExitSupervisor

    def unreadable():
        raise OSError("database lost")
    assert ExitSupervisor._holds(SimpleNamespace(position_qty=unreadable)) is True
    assert ExitSupervisor._holds(SimpleNamespace(position_qty=lambda: D(0))) is False
