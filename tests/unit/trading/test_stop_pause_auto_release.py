"""A stop's entry pause ends by itself after 4 hours (tech lead 2026-11-16).

The tech lead chose that the pause set after an exchange stop triggers, or after a breached stop is flattened,
ends by itself 4 hours later, in live and in the backtest alike (lab/um_lab.py ``LiveTerms.pause_hours`` uses the
same constant). A person can still release it earlier. A pause set by a restore from backup still waits for a person,
also when a stop's pause is active at the same time. The store keeps every row: a scheduled pause is written with
its end, so the history shows when it ended and why.
"""
from __future__ import annotations

import sqlite3
from decimal import Decimal as D

import pytest

from lab import um_lab
from tests.characterization.test_exit_paths import _tick
from tests.characterization.test_runtime_decisions import _events, _start, _until_position
from tests.support.state_db import state_db
from tests.unit.lab.test_target_runtime import _advance
from trading.adapters.sqlite.entry_pause import SCHEDULED_END, SqliteEntryPauses
from trading.adapters.sqlite.schema import STATE_DB
from trading.domain.issues import STOP_PAUSE_SECONDS


@pytest.fixture(autouse=True)
def _offline_env(monkeypatch):
    for name in ("QUANT_MODEL_SIGN_KEY", "BINANCE_TESTNET", "BINANCE_EXCHANGE_LEVERAGE", "BINANCE_MARGIN_TYPE",
                 "MAX_DAILY_DRAWDOWN_PCT", "MAX_WEEKLY_DRAWDOWN_PCT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("QUANT_ALLOW_UNSIGNED_MODELS", "1")
    monkeypatch.setenv("RISK_SCOPE", "binance-testnet-ut")


def _store(tmp_path):
    clock = {"t": 1000.0}
    return SqliteEntryPauses(state_db(tmp_path / STATE_DB), clock=lambda: clock["t"]), clock


def _rows(tmp_path):
    with sqlite3.connect(tmp_path / STATE_DB) as db:
        return db.execute("SELECT symbol, paused_at, cause, released_at, release_reason FROM entry_pauses "
                          "ORDER BY rowid").fetchall()


def test_the_pause_lasts_4_hours_the_same_in_live_and_in_the_backtest():
    assert STOP_PAUSE_SECONDS == 4 * 3600
    assert um_lab.LIVE.pause_hours * 3600 == STOP_PAUSE_SECONDS


def test_a_scheduled_pause_ends_by_itself_and_its_row_records_the_end(tmp_path):
    pauses, clock = _store(tmp_path)
    assert pauses.pause("BTCUSDT", "protective stop triggered", triggers=["7"], ends_at=1000.0 + 4 * 3600) is True
    clock["t"] = 1000.0 + 4 * 3600 - 1
    assert pauses.reason("BTCUSDT") == "protective stop triggered"
    assert [(r["symbol"], r["ends_at"]) for r in pauses.active()] == [("BTCUSDT", 1000.0 + 4 * 3600)]
    clock["t"] = 1000.0 + 4 * 3600
    assert pauses.reason("BTCUSDT") is None and pauses.active() == []
    assert pauses.covered("BTCUSDT") == {"7"}                       # the trigger stays covered: no second pause
    assert _rows(tmp_path) == [("BTCUSDT", 1000.0, "protective stop triggered", 1000.0 + 4 * 3600, SCHEDULED_END)]
    assert pauses.release("BTCUSDT", "checked") == 0                 # nothing left to release


def test_a_person_can_end_a_scheduled_pause_earlier(tmp_path):
    pauses, clock = _store(tmp_path)
    pauses.pause("BTCUSDT", "protective stop triggered", ends_at=1000.0 + 4 * 3600)
    clock["t"] = 2000.0
    assert pauses.release("BTCUSDT", "checked the venue") == 1
    assert pauses.reason("BTCUSDT") is None
    assert _rows(tmp_path) == [("BTCUSDT", 1000.0, "protective stop triggered", 2000.0, "checked the venue")]


def test_a_pause_for_a_person_outlasts_a_scheduled_one(tmp_path):
    pauses, clock = _store(tmp_path)
    pauses.pause("BTCUSDT", "protective stop triggered", ends_at=1000.0 + 4 * 3600)
    assert pauses.pause("BTCUSDT", "restored from a backup") is True             # longer: a person must release
    clock["t"] = 1000.0 + 5 * 3600
    assert pauses.reason("BTCUSDT") == "restored from a backup"                 # the stop's part has ended
    assert pauses.active()[0]["ends_at"] is None
    assert pauses.pause("BTCUSDT", "protective stop breached", ends_at=clock["t"] + 4 * 3600) is False
    assert pauses.release("BTCUSDT", "checked") == 1 and pauses.reason("BTCUSDT") is None


def test_a_later_stop_extends_a_scheduled_pause_and_an_earlier_end_does_not(tmp_path):
    pauses, clock = _store(tmp_path)
    pauses.pause("BTCUSDT", "protective stop triggered", ends_at=1000.0 + 4 * 3600)
    assert pauses.pause("BTCUSDT", "protective stop breached", ends_at=1000.0 + 3600) is False
    clock["t"] = 3000.0
    assert pauses.pause("BTCUSDT", "protective stop breached", ends_at=3000.0 + 4 * 3600) is True
    clock["t"] = 1000.0 + 4 * 3600
    assert pauses.reason("BTCUSDT") == "protective stop breached"               # the first one has ended
    clock["t"] = 3000.0 + 4 * 3600
    assert pauses.reason("BTCUSDT") is None


def test_after_a_pause_ended_a_new_stop_pauses_again(tmp_path):
    pauses, clock = _store(tmp_path)
    pauses.pause("BTCUSDT", "protective stop triggered", ends_at=1000.0 + 4 * 3600)
    clock["t"] = 1000.0 + 5 * 3600
    assert pauses.pause("BTCUSDT", "protective stop triggered", ends_at=clock["t"] + 4 * 3600) is True
    assert pauses.reason("BTCUSDT") == "protective stop triggered" and len(_rows(tmp_path)) == 2


def test_the_runtime_resumes_entries_4_hours_after_a_stop(tmp_path):
    world, runtime = _start(tmp_path)
    i = _until_position(world, runtime)
    btc = world["fakes"]["BTCUSDT"]
    qty = btc.positions["BTCUSDT"]
    (client_id,) = [k for k, v in btc.algo.items() if v["algoStatus"] == "NEW"]
    btc.trigger_stop(client_id, btc.mark("BTCUSDT"))
    _tick(world, runtime)
    assert btc.positions["BTCUSDT"] == 0
    stopped_at = world["clock"]["now"]
    (alert,) = [r for r in _events(tmp_path, "alert") if r["symbol"] == "BTCUSDT"]
    assert "resume by themselves" in alert["reason"] and "trading.infrastructure.entry_pause" in alert["reason"]
    k = i
    while world["clock"]["now"] < stopped_at + STOP_PAUSE_SECONDS - 3600:      # within the 4 hours: no entry
        k += 1
        _advance(world, runtime, k)
        assert btc.positions["BTCUSDT"] == 0
        assert runtime.status()["symbols"]["BTCUSDT"]["entry_pause"] == "protective stop triggered"
    for _ in range(3):                                                          # past the 4 hours: entries again
        k += 1
        _advance(world, runtime, k)
        _tick(world, runtime, seconds=1.1)
    assert runtime.status()["symbols"]["BTCUSDT"]["entry_pause"] is None
    assert btc.positions["BTCUSDT"] != D(0) and qty != 0                    # entered again


@pytest.mark.parametrize("ends_at", [1000.0, 999.0, float("nan"), float("inf")])
def test_a_scheduled_end_that_is_not_in_the_future_is_refused(tmp_path, ends_at):
    pauses, _ = _store(tmp_path)
    with pytest.raises(ValueError, match="future"):          # the runtime then blocks entries (PAUSE_NOT_SAVED)
        pauses.pause("BTCUSDT", "protective stop triggered", ends_at=ends_at)
    assert _rows(tmp_path) == []
