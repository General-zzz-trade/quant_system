"""After a stop: pause 4 hours, then wait for the strategy's target to return to 0 (tech lead 2026-11-17).

The lab found that waiting for the signal to reset after a stop beats the 4-hour pause alone for the trend ideas
(research/lab/runs/2026-11-17/judge.md). The tech lead chose the stricter form: a spec may say
``"after_stop": "pause_then_reset"``. Then, once a stop's pause has ended by itself, the symbol takes no entry while
it is flat until the strategy has decided a target of exactly 0 (on a ready decision) since the stop. Without the
field nothing changes (``"pause"``): every existing spec keeps its behaviour.

- A pause a person released, or a pause still running, carries no extra wait (the person decided; the pause blocks).
- A position held means an entry already happened after the reset: no wait.
- A target of 0 seen before a restart is not remembered: after a restart the symbol waits for the next one (fail
  closed). A store that cannot be read blocks the symbol's entries.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import pytest

from lab.strategy import StrategySpec
from runner.binance_target import TargetRuntime
from tests.characterization.test_exit_paths import _tick
from tests.support.state_db import state_db, state_dir_db
from tests.unit.lab.test_target_runtime import _advance, _world
from trading.adapters.sqlite.entry_pause import SqliteEntryPauses
from trading.adapters.sqlite.schema import STATE_DB
from trading.domain.issues import STOP_PAUSE_SECONDS
from trading.domain.strategy.base import PerSymbol
from trading.domain.strategy.spec import parse_spec

SPEC = StrategySpec(lookbacks=(5, 10, 20), vol_lookback=10, target_vol=0.6, band=0.05, execution="market")
WAITING = "waiting for the strategy's target to return to 0 after a stop"


@pytest.fixture(autouse=True)
def _offline_env(monkeypatch):
    for name in ("QUANT_MODEL_SIGN_KEY", "BINANCE_TESTNET", "BINANCE_EXCHANGE_LEVERAGE", "BINANCE_MARGIN_TYPE",
                 "MAX_DAILY_DRAWDOWN_PCT", "MAX_WEEKLY_DRAWDOWN_PCT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("QUANT_ALLOW_UNSIGNED_MODELS", "1")
    monkeypatch.setenv("RISK_SCOPE", "binance-testnet-ut")


def _doc(**over):
    doc = {"schema": 2, "kind": "target_strategy", "strategy": {"name": "overlay", "params": SPEC.record()},
           "execution": {"band": SPEC.band, "execution": SPEC.execution, "ttl_bars": SPEC.ttl_bars},
           "symbols": ["BTCUSDT", "ETHUSDT"], "allow_short": False, "passed": False}
    doc.update(over)
    return doc


# -- the spec ----------------------------------------------------------------------------------------------------
def test_a_spec_without_the_field_keeps_the_pause_alone():
    spec = parse_spec(_doc())
    assert spec.after_stop == "pause"
    assert "after_stop" not in spec.identity()                       # existing records stay identical


def test_a_spec_may_ask_to_wait_for_the_reset_and_says_so_in_its_identity():
    spec = parse_spec(_doc(after_stop="pause_then_reset"))
    assert spec.after_stop == "pause_then_reset"
    assert spec.identity()["after_stop"] == "pause_then_reset"
    assert spec.record()["after_stop"] == "pause_then_reset"


@pytest.mark.parametrize("value", ["signal_reset", "reset", "", None, 1, True])
def test_any_other_rule_is_refused(value):
    with pytest.raises(ValueError, match="after_stop"):
        parse_spec(_doc(after_stop=value))


def test_a_schema_1_spec_keeps_the_pause_alone():
    doc = {"schema": 1, "kind": "btc_eth_multitimeframe_target", "symbols": ["BTCUSDT", "ETHUSDT"],
           "spec": SPEC.record(), "passed": False}
    assert parse_spec(doc).after_stop == "pause"


# -- the pause store ---------------------------------------------------------------------------------------------
def _store(tmp_path):
    clock = {"t": 1000.0}
    return SqliteEntryPauses(state_db(tmp_path / STATE_DB), clock=lambda: clock["t"]), clock


def test_the_store_says_when_the_latest_pause_ended_by_itself(tmp_path):
    pauses, clock = _store(tmp_path)
    assert pauses.ended_by_itself("BTCUSDT") is None                  # never paused
    pauses.pause("BTCUSDT", "protective stop triggered", ["7"], ends_at=1000.0 + STOP_PAUSE_SECONDS)
    assert pauses.ended_by_itself("BTCUSDT") is None                  # still running
    clock["t"] = 1000.0 + STOP_PAUSE_SECONDS
    assert pauses.ended_by_itself("BTCUSDT") == 1000.0                # ended by itself: when it began
    assert pauses.ended_by_itself("ETHUSDT") is None


def test_a_pause_a_person_released_or_a_later_pause_for_a_person_is_not_one_that_ended_by_itself(tmp_path):
    pauses, clock = _store(tmp_path)
    pauses.pause("BTCUSDT", "protective stop triggered", ends_at=1000.0 + STOP_PAUSE_SECONDS)
    clock["t"] = 2000.0
    pauses.release("BTCUSDT", "checked the venue")
    clock["t"] = 1000.0 + 2 * STOP_PAUSE_SECONDS
    assert pauses.ended_by_itself("BTCUSDT") is None                  # a person decided
    pauses.pause("BTCUSDT", "protective stop triggered", ends_at=clock["t"] + STOP_PAUSE_SECONDS)
    clock["t"] += STOP_PAUSE_SECONDS
    assert pauses.ended_by_itself("BTCUSDT") == clock["t"] - STOP_PAUSE_SECONDS    # the latest one counts
    pauses.pause("BTCUSDT", "restored from a backup")                 # a pause for a person, later
    clock["t"] += 10
    pauses.release("BTCUSDT", "checked")
    assert pauses.ended_by_itself("BTCUSDT") is None


# -- the runtime -------------------------------------------------------------------------------------------------
class Scripted(PerSymbol):
    """Wants ``weight`` of each symbol's share at every close; a test changes it between closes."""
    name = "scripted"

    def __init__(self, weight=0.5):
        self.weight, self.ready = weight, True

    def window_hours(self):
        return SPEC.window_hours()

    def funding_window(self):
        return SPEC.funding_window()

    def symbol_history(self, window):
        times = window.hourly["available_at"].to_numpy(dtype=np.int64)
        return pd.DataFrame({"available_at": times, "target": self.weight, "ready": self.ready})


def _runtime(world, tmp_path, strategy):
    runtime = TargetRuntime(spec_path=world["spec_path"], state_dir=tmp_path / "exec", mode="execution",
                            market=world["adapter"], account=world["adapter"],
                            clock=lambda: world["clock"]["now"], testnet=True)
    for session in runtime.sessions.values():
        session.gate.inner._limiter._monotonic = lambda: world["clock"]["now"]
        session.strategy = strategy
    return runtime


def _start(tmp_path, after_stop):
    world = _world(tmp_path, SPEC)
    world["spec_path"].write_text(json.dumps(_doc(**({"after_stop": after_stop} if after_stop else {}))))
    state_dir_db(tmp_path / "exec")
    strategy = Scripted()
    runtime = _runtime(world, tmp_path, strategy)
    runtime.warmup()
    return world, runtime, strategy


def _close(world, runtime, i):
    _advance(world, runtime, i)
    for _ in range(2):                    # the next 5-second cycles send what the per-second limit held
        _tick(world, runtime, 5)


def _stopped(tmp_path, after_stop):
    """Enter, trigger BTC's stop, and run the closes of the 4-hour pause; returns where the next close is."""
    world, runtime, strategy = _start(tmp_path, after_stop)
    i = world["warm"]
    btc = world["fakes"]["BTCUSDT"]
    while btc.positions.get("BTCUSDT", 0) == 0:
        _close(world, runtime, i)
        i += 1
    (client_id,) = [k for k, v in btc.algo.items() if v["algoStatus"] == "NEW"]
    btc.trigger_stop(client_id, btc.mark("BTCUSDT"))
    _tick(world, runtime)
    assert btc.positions.get("BTCUSDT", 0) == 0
    stopped_at = world["clock"]["now"]
    while world["clock"]["now"] < stopped_at + STOP_PAUSE_SECONDS + 3600:
        _close(world, runtime, i)
        i += 1
    return world, runtime, strategy, i


def _status(runtime, symbol="BTCUSDT"):
    return runtime.status()["symbols"][symbol]["entry_pause"]


def test_without_the_field_entries_resume_after_the_4_hours(tmp_path):
    world, runtime, strategy, i = _stopped(tmp_path, None)
    for k in range(i, i + 3):
        _close(world, runtime, k)
        time.sleep(1.1)
    assert world["fakes"]["BTCUSDT"].positions.get("BTCUSDT", 0) != 0 and _status(runtime) is None


def test_after_the_4_hours_the_symbol_waits_until_the_target_was_0_then_enters_again(tmp_path):
    world, runtime, strategy, i = _stopped(tmp_path, "pause_then_reset")
    btc = world["fakes"]["BTCUSDT"]
    for k in range(i, i + 4):                                        # the signal is still on: no entry
        _close(world, runtime, k)
        assert btc.positions.get("BTCUSDT", 0) == 0
    assert _status(runtime) == WAITING and runtime.sessions["BTCUSDT"].state.value == "paused"
    assert world["fakes"]["ETHUSDT"].positions.get("ETHUSDT", 0) != 0      # the other symbol never stopped: unaffected
    strategy.weight = 0.0                                            # the signal resets
    _close(world, runtime, i + 4)
    assert _status(runtime) is None and btc.positions.get("BTCUSDT", 0) == 0
    strategy.weight = 0.5                                            # and comes back: a new entry
    for k in range(i + 5, i + 8):
        _close(world, runtime, k)
        time.sleep(1.1)
    assert btc.positions.get("BTCUSDT", 0) != 0


def test_a_target_of_0_during_the_pause_counts(tmp_path):
    world, runtime, strategy = _start(tmp_path, "pause_then_reset")
    i = world["warm"]
    btc = world["fakes"]["BTCUSDT"]
    while btc.positions.get("BTCUSDT", 0) == 0:
        _close(world, runtime, i)
        i += 1
    (client_id,) = [k for k, v in btc.algo.items() if v["algoStatus"] == "NEW"]
    btc.trigger_stop(client_id, btc.mark("BTCUSDT"))
    _tick(world, runtime)
    stopped_at = world["clock"]["now"]
    strategy.weight = 0.0
    _close(world, runtime, i)                                        # a 0 decision inside the 4 hours
    strategy.weight = 0.5
    i += 1
    while world["clock"]["now"] < stopped_at + STOP_PAUSE_SECONDS + 3600:
        _close(world, runtime, i)
        i += 1
    for k in range(i, i + 3):
        _close(world, runtime, k)
        time.sleep(1.1)
    assert btc.positions.get("BTCUSDT", 0) != 0 and _status(runtime) is None


def test_a_person_s_release_ends_the_wait(tmp_path):
    world, runtime, strategy, i = _stopped(tmp_path, "pause_then_reset")
    assert _status(runtime) == WAITING
    # the 4-hour pause has ended already; a person who wants entries back releases the symbol: here the scheduled
    # row is the latest, so the person pauses and releases it (the CLI's release ends active pauses only)
    pauses = SqliteEntryPauses(tmp_path / "exec" / STATE_DB, clock=lambda: world["clock"]["now"])
    pauses.pause("BTCUSDT", "held by a person")
    pauses.release("BTCUSDT", "checked; entries may resume")
    for k in range(i, i + 3):
        _close(world, runtime, k)
        time.sleep(1.1)
    assert world["fakes"]["BTCUSDT"].positions.get("BTCUSDT", 0) != 0


def test_after_a_restart_the_symbol_waits_for_a_new_0(tmp_path):
    world, runtime, strategy, i = _stopped(tmp_path, "pause_then_reset")
    strategy.weight = 0.0
    _close(world, runtime, i)
    assert _status(runtime) is None                                  # reset seen in this process
    strategy.weight = 0.5
    again = _runtime(world, tmp_path, strategy)                      # a new process over the same state
    again.warmup()
    for k in range(i + 1, i + 3):
        _close(world, again, k)
    assert _status(again) == WAITING                                 # not remembered: waits (fail closed)
    assert world["fakes"]["BTCUSDT"].positions.get("BTCUSDT", 0) == 0


def test_an_unreadable_store_blocks_the_symbol(tmp_path, monkeypatch):
    world, runtime, strategy, i = _stopped(tmp_path, "pause_then_reset")
    strategy.weight = 0.0
    _close(world, runtime, i)
    strategy.weight = 0.5

    def broken(symbol):
        raise OSError("disk")
    monkeypatch.setattr(runtime.exits.pauses, "ended_by_itself", broken)
    for k in range(i + 1, i + 3):
        _close(world, runtime, k)
    assert _status(runtime) == "unreadable" and world["fakes"]["BTCUSDT"].positions.get("BTCUSDT", 0) == 0


def test_the_stop_alert_says_that_entries_also_wait_for_the_reset(tmp_path):
    world, runtime, strategy, i = _stopped(tmp_path, "pause_then_reset")
    rows = [json.loads(x) for x in (tmp_path / "exec" / "target_events.jsonl").read_text().splitlines()]
    alerts = [r for r in rows if r["kind"] == "alert" and "entries paused" in r.get("reason", "")]
    assert alerts and "only once the strategy's target has returned to 0" in alerts[0]["reason"]


def test_a_0_from_a_decision_that_is_not_ready_does_not_count(tmp_path):
    world, runtime, strategy, i = _stopped(tmp_path, "pause_then_reset")
    strategy.weight, strategy.ready = 0.0, False
    _close(world, runtime, i)
    assert _status(runtime) == WAITING
    strategy.ready = True
    _close(world, runtime, i + 1)
    assert _status(runtime) is None


def test_after_a_restart_a_position_held_does_not_wait(tmp_path):
    world, runtime, strategy, i = _stopped(tmp_path, "pause_then_reset")
    strategy.weight = 0.0
    _close(world, runtime, i)
    strategy.weight = 0.5
    for k in range(i + 1, i + 4):
        _close(world, runtime, k)
        time.sleep(1.1)
    assert world["fakes"]["BTCUSDT"].positions.get("BTCUSDT", 0) != 0       # entered again after the reset
    again = _runtime(world, tmp_path, strategy)
    again.warmup()
    _close(world, again, i + 4)
    assert _status(again) is None                     # an entry happened after the stop: no wait


def test_both_modes_take_the_rule_from_the_spec(tmp_path):
    world = _world(tmp_path, SPEC)
    world["spec_path"].write_text(json.dumps(_doc(after_stop="pause_then_reset")))
    paper = TargetRuntime(spec_path=world["spec_path"], state_dir=tmp_path / "paper", mode="paper",
                          market=world["adapter"], clock=lambda: world["clock"]["now"])
    assert {s.after_stop for s in paper.sessions.values()} == {"pause_then_reset"}
    state_dir_db(tmp_path / "exec")
    execution = _runtime(world, tmp_path, Scripted())
    assert {s.after_stop for s in execution.sessions.values()} == {"pause_then_reset"}


def test_paper_s_pauses_pass_the_query_on(tmp_path):
    from trading.infrastructure.paper import DivergencePauses
    store, clock = _store(tmp_path)
    pauses = DivergencePauses(store, type("Log", (), {"paused": lambda self, symbol, cause: None})())
    pauses.pause("BTCUSDT", "protective stop triggered", ends_at=1000.0 + STOP_PAUSE_SECONDS)
    clock["t"] = 1000.0 + STOP_PAUSE_SECONDS
    assert pauses.ended_by_itself("BTCUSDT") == store.ended_by_itself("BTCUSDT") == 1000.0
