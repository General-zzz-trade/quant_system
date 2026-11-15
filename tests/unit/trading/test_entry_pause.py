"""After a protective stop triggers or is breached, the symbol's entries pause until a person releases them.

Decisions of the tech lead: breached stop 2026-11-14, normally triggered stop 2026-11-15 (migration step 12).
The runtime only ever sets a pause; releasing one is a command a person runs, with a reason and a typed
confirmation.
"""
from __future__ import annotations

import io
import json
import sqlite3
import time
from decimal import Decimal as D

import pytest

from tests.characterization.test_exit_paths import _tick
from tests.characterization.test_runtime_decisions import _events, _order, _runtime, _start, _until_position
from tests.unit.lab.test_target_runtime import _advance
from trading.adapters.sqlite.entry_pause import SqliteEntryPauses
from trading.infrastructure import entry_pause as cli
from tests.support.state_db import state_db
from trading.adapters.sqlite.schema import STATE_DB as PAUSES_FILE


@pytest.fixture(autouse=True)
def _offline_env(monkeypatch):
    for name in ("QUANT_MODEL_SIGN_KEY", "BINANCE_TESTNET", "BINANCE_EXCHANGE_LEVERAGE", "BINANCE_MARGIN_TYPE",
                 "MAX_DAILY_DRAWDOWN_PCT", "MAX_WEEKLY_DRAWDOWN_PCT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("QUANT_ALLOW_UNSIGNED_MODELS", "1")
    monkeypatch.setenv("RISK_SCOPE", "binance-testnet-ut")


def _release(state_dir, symbol, typed, reason="checked the venue and the ledger"):
    out = io.StringIO()
    code = cli.main(["--state-dir", str(state_dir), "release", "--symbol", symbol, "--reason", reason],
                    stdin=io.StringIO(typed + "\n"), stdout=out)
    return code, out.getvalue()


# -- the store ------------------------------------------------------------------------------------------------
def test_a_pause_is_kept_until_released_and_its_history_stays(tmp_path):
    clock = {"t": 100.0}
    pauses = SqliteEntryPauses(state_db(tmp_path / PAUSES_FILE), clock=lambda: clock["t"])
    assert pauses.reason("BTCUSDT") is None
    assert pauses.pause("BTCUSDT", "protective stop triggered") is True
    assert pauses.pause("BTCUSDT", "protective stop breached") is False          # the first cause is kept
    assert pauses.reason("BTCUSDT") == "protective stop triggered" and pauses.reason("ETHUSDT") is None
    assert SqliteEntryPauses(tmp_path / PAUSES_FILE).reason("BTCUSDT") == "protective stop triggered"   # restart
    with pytest.raises(ValueError, match="reason"):
        pauses.release("BTCUSDT", "  ")
    clock["t"] = 200.0
    assert pauses.release("BTCUSDT", "checked") == 1 and pauses.reason("BTCUSDT") is None
    assert pauses.release("BTCUSDT", "checked") == 0
    with sqlite3.connect(tmp_path / PAUSES_FILE) as db:
        rows = db.execute("SELECT symbol, paused_at, cause, released_at, release_reason FROM entry_pauses").fetchall()
    assert rows == [("BTCUSDT", 100.0, "protective stop triggered", 200.0, "checked")]


def test_an_unreadable_store_raises_instead_of_reporting_no_pause(tmp_path):
    path = tmp_path / PAUSES_FILE
    path.write_bytes(b"not a database")
    with pytest.raises(sqlite3.DatabaseError):
        SqliteEntryPauses(path).reason("BTCUSDT")


# -- the command a person runs ---------------------------------------------------------------------------------
def test_releasing_needs_the_symbol_typed_back_and_a_reason(tmp_path):
    pauses = SqliteEntryPauses(state_db(tmp_path / PAUSES_FILE))
    pauses.pause("BTCUSDT", "protective stop breached; position flattened")
    code, out = _release(tmp_path, "BTCUSDT", typed="yes")
    assert code == 1 and "not released" in out and pauses.reason("BTCUSDT") is not None
    code, out = _release(tmp_path, "BTCUSDT", typed="BTCUSDT", reason="")
    assert code == 2 and pauses.reason("BTCUSDT") is not None
    code, out = _release(tmp_path, "BTCUSDT", typed="BTCUSDT")
    assert code == 0 and "released" in out and "protective stop breached" in out
    assert pauses.reason("BTCUSDT") is None
    code, out = _release(tmp_path, "BTCUSDT", typed="BTCUSDT")
    assert code == 0 and "no pause" in out


def test_status_lists_the_active_pauses(tmp_path):
    SqliteEntryPauses(state_db(tmp_path / PAUSES_FILE)).pause("ETHUSDT", "protective stop triggered")
    out = io.StringIO()
    assert cli.main(["--state-dir", str(tmp_path), "status"], stdout=out) == 0
    assert [(r["symbol"], r["cause"]) for r in json.loads(out.getvalue())] == [("ETHUSDT", "protective stop triggered")]


def test_the_command_never_creates_a_state_directory(tmp_path):
    missing = tmp_path / "nowhere"
    assert cli.main(["--state-dir", str(missing), "status"], stdout=io.StringIO()) == 2
    assert not missing.exists()


# -- the runtime -----------------------------------------------------------------------------------------------
def _breach(world, runtime, i):
    btc, session = world["fakes"]["BTCUSDT"], runtime.sessions["BTCUSDT"]
    qty, entry = session.consumer.position.qty, session.consumer.position.entry_price
    for stop in btc.algo.values():
        stop["algoStatus"] = "CANCELED"
    btc.price["BTCUSDT"] = entry * (D("0.85") if qty > 0 else D("1.15"))
    _advance(world, runtime, i)
    return qty


def test_a_breached_stop_pauses_entries_until_a_person_releases_them(tmp_path):
    world, runtime = _start(tmp_path)
    i = _until_position(world, runtime)
    _advance(world, runtime, i)
    btc = world["fakes"]["BTCUSDT"]
    qty = _breach(world, runtime, i + 1)
    assert btc.positions["BTCUSDT"] == 0
    cause = "protective stop breached; position flattened"
    assert runtime.status()["symbols"]["BTCUSDT"]["entry_pause"] == cause
    (alert,) = [r for r in _events(tmp_path, "alert") if r["symbol"] == "BTCUSDT"]
    assert "entries paused" in alert["reason"] and "trading.infrastructure.entry_pause" in alert["reason"]
    btc.price.pop("BTCUSDT")
    for k in range(2, 5):
        _advance(world, runtime, i + k)
    assert btc.positions["BTCUSDT"] == 0                                      # no re-entry while paused
    assert runtime.status()["symbols"]["ETHUSDT"]["entry_pause"] is None       # only this symbol
    def paused_refusals():
        return sum(1 for r in _events(tmp_path, "blocked")
                   if r["symbol"] == "BTCUSDT" and r["reason"] == "runner paused")

    assert paused_refusals() == 3
    assert _release(tmp_path / "exec", "BTCUSDT", typed="BTCUSDT")[0] == 0
    _advance(world, runtime, i + 5)
    assert paused_refusals() == 3                                             # the pause no longer refuses
    time.sleep(1.1)                  # both symbols enter on this bar: the second waits one second (R23 retry)
    _tick(world, runtime)
    assert btc.positions["BTCUSDT"] == qty                                    # entries again once released


def test_a_normally_triggered_stop_pauses_entries_and_the_pause_survives_a_restart(tmp_path):
    world, runtime = _start(tmp_path)
    _until_position(world, runtime)
    btc = world["fakes"]["BTCUSDT"]
    (client_id,) = [k for k, v in btc.algo.items() if v["algoStatus"] == "NEW"]
    btc.trigger_stop(client_id, btc.mark("BTCUSDT"))
    _tick(world, runtime)
    assert btc.positions["BTCUSDT"] == 0
    assert runtime.status()["symbols"]["BTCUSDT"]["entry_pause"] == "protective stop triggered"
    runtime = _runtime(tmp_path, world)                                       # restart
    runtime.warmup()
    session = runtime.sessions["BTCUSDT"]
    session.entry_ready, session.entry_reason = True, ""
    refused = session.gate.check(_order(False))
    assert refused.allowed is False and refused.reason == "runner paused"
    assert session.gate.check(_order(True)).allowed                           # reductions still pass


def test_a_pause_that_cannot_be_saved_blocks_entries_and_is_saved_later(tmp_path):
    world, runtime = _start(tmp_path)
    _until_position(world, runtime)
    btc, session = world["fakes"]["BTCUSDT"], runtime.sessions["BTCUSDT"]
    pauses = runtime.pauses
    save = pauses.pause

    def disk_full(symbol, cause):
        raise sqlite3.OperationalError("disk full")

    pauses.pause = disk_full
    (client_id,) = [k for k, v in btc.algo.items() if v["algoStatus"] == "NEW"]
    btc.trigger_stop(client_id, btc.mark("BTCUSDT"))
    _tick(world, runtime)
    assert session.executor.ready is False
    _tick(world, runtime)                    # the stop's fill is in the ledger now: only the unsaved pause blocks
    assert session.position_qty() == 0 and session.stops.unprotected() is False
    assert session.executor.ready is False
    assert sum("entry pause not saved" in r["issue"] for r in _events(tmp_path, "issue")) == 2
    pauses.pause = save
    _tick(world, runtime)
    assert pauses.reason("BTCUSDT") == "protective stop triggered"


def test_an_unreadable_pause_store_refuses_entries(tmp_path):
    world, runtime = _start(tmp_path)
    session = runtime.sessions["BTCUSDT"]
    session.entry_ready, session.entry_reason = True, ""

    def unreadable(symbol):
        raise sqlite3.DatabaseError("file is not a database")

    runtime.pauses.reason = unreadable
    refused = session.gate.check(_order(False))
    assert refused.allowed is False and refused.reason == "risk check failed: DatabaseError"
    assert runtime.status()["symbols"]["BTCUSDT"]["entry_pause"] == "unreadable"
