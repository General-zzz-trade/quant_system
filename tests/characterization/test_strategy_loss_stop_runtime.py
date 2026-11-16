"""The strategy loss stop in the running runtime (owner, 2026-11-16): a 15 % loss of the strategy capital closes
every position through the kill switch and stops entries until a person clears it; clearing alone trips it again.
The fake account holds 10,000 USDT, the strategy capital is 300, so the stop is at -45 USDT."""
from __future__ import annotations

import json
from decimal import Decimal as D

import pytest

from tests.characterization.test_exit_paths import _tick, _until
from tests.characterization.test_runtime_decisions import _events
from tests.characterization.test_strategy_capital_sizing import _execution
from tests.unit.lab.test_target_runtime import _advance


@pytest.fixture(autouse=True)
def _capital(monkeypatch):
    monkeypatch.setenv("STRATEGY_CAPITAL_USDT", "300")
    monkeypatch.delenv("STRATEGY_LOSS_STOP_PCT", raising=False)


def _holding(tmp_path):
    world, runtime = _execution(tmp_path)
    btc = world["fakes"]["BTCUSDT"]
    i = _until(world, runtime, lambda: btc.positions.get("BTCUSDT", D(0)) != 0)
    return world, runtime, btc, i


def _kill(tmp_path):
    """The kill file; the test directory starts without one, which reads as inactive there (no manifest)."""
    path = tmp_path / "exec" / "kill_switch.json"
    return json.loads(path.read_text()) if path.exists() else {"active": False}


def _flat(world):
    return all(fake.positions.get(symbol, D(0)) == 0 for symbol, fake in world["fakes"].items())


def test_a_loss_of_15_per_cent_closes_everything_and_stops_entries_until_a_person_clears_it(tmp_path):
    world, runtime, btc, i = _holding(tmp_path)
    btc.add_income("BTCUSDT", "-50", btc.now_ms() + 1_000)          # funding paid: more than 45 with the fees
    _tick(world, runtime, 601)                                      # the funding refresh and the loss check are due
    kill = _kill(tmp_path)
    assert kill["active"] is True and kill["flatten"] is True and "strategy loss stop" in kill["reason"]
    _tick(world, runtime)
    assert _flat(world)
    alerts = [r for r in _events(tmp_path, "alert") if "strategy loss stop" in r.get("reason", "")]
    assert len(alerts) == 1 and "15" in alerts[0]["reason"]
    for k in range(1, 4):                                           # later bars: nothing is entered
        _advance(world, runtime, i + k)
        assert _flat(world)
    assert len([r for r in _events(tmp_path, "alert") if "strategy loss stop" in r.get("reason", "")]) == 1
    (tmp_path / "exec" / "kill_switch.json").write_text(json.dumps({"active": False, "reason": "cleared"}))
    _tick(world, runtime, 61)                                       # the loss is still in the ledger
    assert _kill(tmp_path)["active"] is True and _kill(tmp_path)["flatten"] is True
    _advance(world, runtime, i + 4)
    assert _flat(world)


def test_a_loss_short_of_15_per_cent_leaves_the_runtime_trading(tmp_path):
    world, runtime, btc, _ = _holding(tmp_path)
    btc.add_income("BTCUSDT", "-30", btc.now_ms() + 1_000)
    _tick(world, runtime, 601)
    assert _kill(tmp_path)["active"] is False
    assert btc.positions["BTCUSDT"] != 0


def test_a_tighter_limit_set_by_a_person_is_used(tmp_path, monkeypatch):
    monkeypatch.setenv("STRATEGY_LOSS_STOP_PCT", "8")               # 24 USDT
    world, runtime, btc, _ = _holding(tmp_path)
    assert runtime.config.strategy_loss_stop == D("0.08")
    btc.add_income("BTCUSDT", "-30", btc.now_ms() + 1_000)
    _tick(world, runtime, 601)
    assert _kill(tmp_path)["active"] is True


def test_the_open_position_s_unrealized_loss_counts(tmp_path):
    world, runtime, btc, _ = _holding(tmp_path)
    qty, entry = btc.positions["BTCUSDT"], btc.entry["BTCUSDT"]
    btc.price["BTCUSDT"] = entry - D("50") / qty                    # 50 USDT under water at the mark
    _tick(world, runtime, 61)
    assert _kill(tmp_path)["active"] is True and "strategy loss stop" in _kill(tmp_path)["reason"]


def test_a_loss_that_cannot_be_known_closes_nothing_and_blocks_entries(tmp_path):
    world, runtime, btc, i = _holding(tmp_path)
    held = btc.positions["BTCUSDT"]
    for fake in world["fakes"].values():
        fake.income_error = True
    for step in range(3):                                           # after two hours the funding total is unknown
        _advance(world, runtime, i + step)
    assert _kill(tmp_path)["active"] is False
    assert btc.positions["BTCUSDT"] == held                         # nothing was closed
    issues = [r for r in _events(tmp_path, "issue") if "strategy loss" in r.get("issue", "")]
    assert issues and issues[0]["issue_kind"] == "strategy_loss_stop"


def test_without_a_strategy_capital_there_is_no_loss_stop(tmp_path, monkeypatch):
    monkeypatch.delenv("STRATEGY_CAPITAL_USDT", raising=False)
    world, runtime, btc, _ = _holding(tmp_path)
    btc.add_income("BTCUSDT", "-5000", btc.now_ms() + 1_000)
    _tick(world, runtime, 601)
    assert _kill(tmp_path)["active"] is False and runtime.exits.loss_stop is None


def test_while_the_loss_is_unknown_no_entry_is_made_although_the_sizing_is_known(tmp_path):
    """The unknown-loss test above also has a zero sizing base. Here only the loss stop reports a finding: entries
    are blocked at once and stay blocked through the reconciliations, until the finding is gone."""
    from types import SimpleNamespace
    from trading.application.issues import Issue
    from trading.domain.issues import IssueKind
    control, runtime_c, _, first = _holding(tmp_path / "control")          # the bar after which BTC is held
    world, runtime = _execution(tmp_path / "blocked")
    btc = world["fakes"]["BTCUSDT"]
    _tick(world, runtime)
    assert all(s.executor.ready for s in runtime.sessions.values())
    stub = SimpleNamespace(findings=[Issue(IssueKind.STRATEGY_LOSS_STOP, None, "strategy loss unknown (test)")],
                           take_trip=lambda: None)
    stub.check = lambda: list(stub.findings)
    runtime.exits.loss_stop = stub
    runtime.exits.check_loss_stop()
    assert not any(s.executor.ready for s in runtime.sessions.values())    # at once, before any reconciliation
    for i in range(world["warm"], first + 3):
        _advance(world, runtime, i)
        _tick(world, runtime, 31)                                          # past a reconciliation
        assert not any(s.executor.ready for s in runtime.sessions.values())
    assert _flat(world)                                                     # the control holds BTC by now
    assert control["fakes"]["BTCUSDT"].positions["BTCUSDT"] != 0
    stub.findings = []
    for i in range(first + 3, first + 40):                                 # known again: entries come back
        if btc.positions.get("BTCUSDT", D(0)) != 0:
            break
        _advance(world, runtime, i)
    assert btc.positions.get("BTCUSDT", D(0)) != 0
