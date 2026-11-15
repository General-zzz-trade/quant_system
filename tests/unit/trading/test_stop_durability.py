"""Stops and pauses that survive restarts and odd venue answers (audit after step 12b, findings 2, 4, 5, 8).

A pause is derived from the stop rows the order journal keeps - triggered, awaiting its order, breached -
and every such row must be covered by a pause the store has recorded. So a pause that could not be written,
a restart in between, or a deleted pause store all end in the symbol being paused again.
"""
from __future__ import annotations

import sqlite3
from decimal import Decimal as D

import pytest

from trading.adapters.binance.rest import BinanceNonRetryableError, BinanceRetryableError
from tests.characterization.test_exit_paths import _tick
from tests.characterization.test_runtime_decisions import _events, _runtime, _start, _until_position
from tests.characterization.test_stops_and_fills import OWNER, _with_position
from tests.unit.lab.test_target_runtime import _advance
from trading.adapters.sqlite.entry_pause import SqliteEntryPauses
from trading.infrastructure.target import PAUSES_FILE


@pytest.fixture(autouse=True)
def _offline_env(monkeypatch):
    for name in ("QUANT_MODEL_SIGN_KEY", "BINANCE_TESTNET", "BINANCE_EXCHANGE_LEVERAGE", "BINANCE_MARGIN_TYPE",
                 "MAX_DAILY_DRAWDOWN_PCT", "MAX_WEEKLY_DRAWDOWN_PCT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("QUANT_ALLOW_UNSIGNED_MODELS", "1")
    monkeypatch.setenv("RISK_SCOPE", "binance-testnet-ut")


def _trigger(world, runtime):
    btc = world["fakes"]["BTCUSDT"]
    (client_id,) = [k for k, v in btc.algo.items() if v["algoStatus"] == "NEW"]
    btc.trigger_stop(client_id, btc.mark("BTCUSDT"))
    return client_id


# -- findings 2 and 8: the pause comes back -------------------------------------------------------------------
def test_a_pause_that_was_never_written_is_written_after_a_restart(tmp_path):
    world, runtime = _start(tmp_path)
    _until_position(world, runtime)

    def disk_full(symbol, cause, triggers=()):
        raise sqlite3.OperationalError("disk full")

    runtime.pauses.pause = disk_full
    _trigger(world, runtime)
    _tick(world, runtime)
    _tick(world, runtime)
    assert runtime.sessions["BTCUSDT"].executor.ready is False
    runtime = _runtime(tmp_path, world)                                       # restart; the store works again
    runtime.warmup()
    _tick(world, runtime)
    assert runtime.pauses.reason("BTCUSDT") == "protective stop triggered"
    assert [r for r in _events(tmp_path, "alert") if r["symbol"] == "BTCUSDT"]


def test_a_deleted_pause_store_pauses_again_every_symbol_whose_stop_ever_fired(tmp_path):
    world, runtime = _start(tmp_path)
    _until_position(world, runtime)
    _trigger(world, runtime)
    _tick(world, runtime)
    pauses = SqliteEntryPauses(tmp_path / "exec" / PAUSES_FILE)
    assert pauses.release("BTCUSDT", "checked") == 1
    _tick(world, runtime)
    assert pauses.reason("BTCUSDT") is None                                   # a release holds ...
    (tmp_path / "exec" / PAUSES_FILE).unlink()
    _tick(world, runtime)
    assert pauses.reason("BTCUSDT") == "protective stop triggered"            # ... a lost store does not release
    assert runtime.pauses.reason("ETHUSDT") is None


def test_a_breached_stop_is_remembered_as_breached(tmp_path):
    world, runtime = _start(tmp_path)
    i = _until_position(world, runtime)
    btc, session = world["fakes"]["BTCUSDT"], runtime.sessions["BTCUSDT"]
    qty, entry = session.consumer.position.qty, session.consumer.position.entry_price
    for stop in btc.algo.values():
        stop["algoStatus"] = "CANCELED"
    btc.price["BTCUSDT"] = entry * (D("0.85") if qty > 0 else D("1.15"))
    _advance(world, runtime, i)
    rows = session.executor.journal.stop_rows(OWNER, "BTCUSDT", live_only=False)
    assert "breached" in [r["status"] for r in rows]
    assert runtime.pauses.reason("BTCUSDT") == "protective stop breached; position flattened"


# -- finding 5: a stop that triggered but produced no order ----------------------------------------------------
def test_a_triggered_stop_that_ends_without_an_order_is_replaced_and_still_pauses(tmp_path):
    fake, executor, consumer, stops = _with_position(tmp_path)
    stops.sync()
    (client_id,) = list(fake.algo)
    fake.algo[client_id]["algoStatus"] = "TRIGGERED"
    stops.sync()
    fake.algo[client_id]["algoStatus"] = "EXPIRED"                            # the venue gave up on it
    issues = stops.sync()
    assert issues == ["BTCUSDT: protective stop triggered but ended without an order; a new stop is placed"]
    assert len(fake.algo) == 2 and stops.has_active_stop()                    # the position is protected again
    assert client_id in stops.triggers()                                      # and the trigger still pauses


# -- finding 4: no stop for a position the venue does not hold -------------------------------------------------
def test_no_stop_is_placed_for_a_position_the_venue_does_not_hold(tmp_path):
    fake, executor, consumer, stops = _with_position(tmp_path)
    fake.positions["BTCUSDT"] = D(0)                                          # the ledger says 0.1
    assert stops.sync() == ["BTCUSDT: venue holds 0, ledger 0.1; protective stop not placed"]
    assert fake.algo == {}
    fake.positions["BTCUSDT"] = D("-0.1")
    assert stops.sync() == ["BTCUSDT: venue holds -0.1, ledger 0.1; protective stop not placed"]
    fake.positions["BTCUSDT"] = D("0.05")                                     # same side: the stop closes it all
    assert stops.sync() == [] and len(fake.algo) == 1


def test_no_stop_is_placed_when_the_venue_position_cannot_be_read(tmp_path):
    fake, executor, consumer, stops = _with_position(tmp_path)
    fake.position_error = True
    assert stops.sync() == ["BTCUSDT: venue position unavailable; protective stop not placed"]
    assert fake.algo == {}


# -- the auditor's surviving mutants S24 and ST1 ---------------------------------------------------------------
def test_a_stop_placement_with_an_unknown_outcome_does_not_count_as_protection(tmp_path):
    world, runtime = _start(tmp_path)
    _until_position(world, runtime)
    btc, session = world["fakes"]["BTCUSDT"], runtime.sessions["BTCUSDT"]
    original = btc.request_signed

    def unknown(*, method, path, params=None):
        if path == "/fapi/v1/algoOrder" and method == "POST":
            raise BinanceRetryableError("HTTP 503: unknown")
        return original(method=method, path=path, params=params)

    for stop in btc.algo.values():
        stop["algoStatus"] = "CANCELED"
    btc.request_signed = unknown
    _tick(world, runtime)
    assert session.stops.has_active_stop() is False and session.executor.ready is False
    assert "BTCUSDT: position without an active protective stop; entries blocked" in [
        r["issue"] for r in _events(tmp_path, "issue")]


def test_a_named_order_the_venue_cannot_find_is_not_taken_as_adopted(tmp_path):
    fake, executor, consumer, stops = _with_position(tmp_path)
    stops.sync()
    (client_id,) = list(fake.algo)
    fake.trigger_stop(client_id, D("95.9"))
    original = stops.adapter.get_order
    stops.adapter.get_order = lambda symbol, **kw: None
    assert stops.sync() == ["BTCUSDT: protective stop triggered; the order it became is not known yet"]
    assert not [r for r in executor.journal.orders(OWNER) if r["intent_id"].startswith("stop-")]
    stops.adapter.get_order = original
    assert stops.sync() == ["BTCUSDT: protective stop triggered"]


def test_a_refused_placement_that_is_not_a_breach_is_not_a_trigger(tmp_path):
    fake, executor, consumer, stops = _with_position(tmp_path)
    original = fake.request_signed

    def refuse(*, method, path, params=None):
        if path == "/fapi/v1/algoOrder" and method == "POST":
            raise BinanceNonRetryableError('HTTP 400: {"code":-4120}', http_status=400, code=-4120)
        return original(method=method, path=path, params=params)

    fake.request_signed = refuse
    stops.sync()
    assert stops.triggers() == {}


# -- audit after M3: finding 9 -------------------------------------------------------------------------------------
def test_a_breach_is_written_in_one_step(tmp_path):
    fake, executor, consumer, stops = _with_position(tmp_path)
    fake.price["BTCUSDT"] = D("80")                                           # far below the stop of a long
    written = []
    original = stops.journal.set_stop_status

    def record(client_algo_id, status):
        written.append(status)
        original(client_algo_id, status)

    stops.journal.set_stop_status = record
    issues = stops.sync()
    assert any("stop_breached" in issue for issue in issues)
    assert written == ["breached"]                       # never "rejected" first: a crash in between lost the trigger
