"""Reliable flattening (migration step 11, ADR-0005 sub-decisions 1b, 2b, 3b and 7b, approved 2026-11-14)."""
from __future__ import annotations

import json
import threading
from decimal import Decimal as D

import pytest

from tests.characterization.test_exit_paths import _execution, _tick
from tests.characterization.test_runtime_decisions import _events, _start, _until_position
from tests.unit.binance.fake_venue import HOUR
from tests.unit.lab.test_target_runtime import _advance
from tests.unit.trading.test_execution_service import OWNER, SCOPE, Venue
from trading.adapters.binance.rest import BinanceRetryableError
from trading.adapters.sqlite.order_journal import SqliteOrderJournal
from trading.application.execution.executor import OwnerExecutor
from trading.application.execution.fills import TargetFillConsumer
from trading.application.execution.flatten import MAX_FAILURES, RETRY_SECONDS
from trading.application.execution.lifecycle import OrderLifecycle
from trading.application.execution.service import ExecutionService
from trading.domain.orders import REDUCTION_OVERLAP, OrderIntent
from trading.domain.rebalance import Intent
from tests.support.state_db import state_db
from trading.domain.symbol_state import SymbolState


@pytest.fixture(autouse=True)
def _offline_env(monkeypatch):
    for name in ("QUANT_MODEL_SIGN_KEY", "BINANCE_TESTNET", "BINANCE_EXCHANGE_LEVERAGE", "BINANCE_MARGIN_TYPE",
                 "MAX_DAILY_DRAWDOWN_PCT", "MAX_WEEKLY_DRAWDOWN_PCT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("QUANT_ALLOW_UNSIGNED_MODELS", "1")
    monkeypatch.setenv("RISK_SCOPE", "binance-testnet-ut")


def _attempts(tmp_path, reason="kill_switch", symbol="BTCUSDT"):
    return [r["order_id"] for r in _events(tmp_path, "intent") if r["symbol"] == symbol and r["reason"] == reason]


def _alerts(tmp_path, symbol="BTCUSDT"):
    path = tmp_path / "exec/target_events.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    return [r for r in rows if r["kind"] == "alert" and r["symbol"] == symbol]


def _refuse_reductions(fake):
    """The venue rejects every reduce-only order until the returned function is called."""
    original = fake.request_signed

    def refuse(*, method, path, params=None):
        if path == "/fapi/v1/order" and method == "POST" and (params or {}).get("reduceOnly") in (True, "true"):
            fake.posts.append(dict(params))
            fake._error(-1008, "server busy")
        return original(method=method, path=path, params=params)

    fake.request_signed = refuse
    return lambda: setattr(fake, "request_signed", original)


def _with_position(tmp_path):
    world, runtime = _start(tmp_path)
    i = _until_position(world, runtime)
    _advance(world, runtime, i)
    return world, runtime, i, world["fakes"]["BTCUSDT"], runtime.sessions["BTCUSDT"]


def test_the_retry_settings_are_the_approved_ones():
    assert (RETRY_SECONDS, MAX_FAILURES) == (30.0, 10)


# -- 1b: attempts, interval, cap ------------------------------------------------------------------------------
def test_ten_failed_attempts_stop_the_retries_with_an_alert_and_leave_one_attempt_per_bar(tmp_path):
    world, runtime, i, btc, session = _with_position(tmp_path)
    restore = _refuse_reductions(btc)
    runtime.kill.write(active=True, reason="test", flatten=True)
    for _ in range(15):
        _tick(world, runtime, 31)
    attempts = _attempts(tmp_path)
    assert len(attempts) == 10 and len(set(attempts)) == 10                  # ten intents, each sent once
    (alert,) = _alerts(tmp_path)
    assert alert["failures"] == 10 and "once per bar" in alert["reason"]
    assert runtime.status()["symbols"]["BTCUSDT"]["flatten_failures"] == 10
    _advance(world, runtime, i + 1)                                           # a new bar: exactly one attempt
    for _ in range(3):
        _tick(world, runtime, 31)
    assert len(_attempts(tmp_path)) == 11 and len(_alerts(tmp_path)) == 2
    restore()
    _advance(world, runtime, i + 2)
    assert btc.positions["BTCUSDT"] == 0 and len(_attempts(tmp_path)) == 12
    _tick(world, runtime, 31)
    assert runtime.status()["symbols"]["BTCUSDT"]["flatten_failures"] == 0


def test_once_flat_the_next_flatten_starts_afresh(tmp_path):
    world, runtime, i, btc, session = _with_position(tmp_path)
    position = btc.positions["BTCUSDT"]
    restore = _refuse_reductions(btc)
    runtime.kill.write(active=True, reason="test", flatten=True)
    _tick(world, runtime, 31)
    _tick(world, runtime, 31)
    assert len(_attempts(tmp_path)) == 2
    btc.positions["BTCUSDT"] = D(0)                   # closed at the venue by other means (an exchange stop)
    _tick(world, runtime)
    assert runtime.status()["symbols"]["BTCUSDT"]["flatten_failures"] == 0
    restore()
    btc.positions["BTCUSDT"] = position               # exposure again: no 30 s wait left over from before
    _tick(world, runtime)
    assert len(_attempts(tmp_path)) == 3 and btc.positions["BTCUSDT"] == 0


def test_an_attempt_whose_outcome_is_unknown_is_never_followed_by_another(tmp_path):
    world, runtime, i, btc, session = _with_position(tmp_path)
    position = btc.positions["BTCUSDT"]
    btc.fault = "error_5xx"                          # the first reduce-only order: outcome unknown, not placed
    runtime.kill.write(active=True, reason="test", flatten=True)
    # 62 s: past the 30 s retry interval, inside the 90 s "not found" grace. Since step 13a the grace runs on
    # the simulated clock; this test ran 124 s before and passed only because the grace read the real clock.
    for _ in range(2):
        _tick(world, runtime, 31)
    (attempt,) = _attempts(tmp_path)
    assert btc.positions["BTCUSDT"] == position
    row = next(r for r in session.executor.journal.orders(OWNER) if r["intent_id"] == attempt)
    assert (row["settled"], row["status"]) == (0, "unknown")                 # looked up, never resent


def test_a_flatten_is_retried_while_market_data_is_down(tmp_path):
    world, runtime, i, btc, session = _with_position(tmp_path)
    for fake in world["fakes"].values():
        def down(*, method, path, params=None):
            raise BinanceRetryableError("Network error: market data down")
        fake.request_public = down
    btc.fault, btc.fault_code = "reject", -1008
    runtime.kill.write(active=True, reason="test", flatten=True)
    cursor = session.cursor
    world["clock"]["now"] = (world["rows"]["BTCUSDT"][i + 1]["start"] + HOUR + 5_000) / 1000   # the next bar is due
    runtime.cycle()
    assert runtime.status()["market_failures"] == 1 and session.cursor == cursor           # ... and cannot arrive
    assert len(_attempts(tmp_path)) == 1 and btc.positions["BTCUSDT"] != 0
    _tick(world, runtime, 31)                         # the retry does not wait for a bar
    assert len(_attempts(tmp_path)) == 2 and btc.positions["BTCUSDT"] == 0
    assert runtime.status()["market_failures"] == 2 and session.cursor == cursor


# -- 2b: cancel first ---------------------------------------------------------------------------------------
def test_a_cancellation_with_an_unknown_outcome_does_not_stop_the_flatten(tmp_path):
    world, runtime = _execution(tmp_path, execution="maker", ttl_bars=3)
    btc = world["fakes"]["BTCUSDT"]
    i = world["warm"]
    while not btc.orders:
        _advance(world, runtime, i)
        i += 1
    order_id = next(iter(btc.orders))
    btc.fill_limit(order_id, D("0.001"))
    _tick(world, runtime)
    original = btc.request_signed

    def cancel_fails(*, method, path, params=None):
        if path == "/fapi/v1/order" and method == "DELETE":
            raise BinanceRetryableError("HTTP 503: unknown")
        return original(method=method, path=path, params=params)

    btc.request_signed = cancel_fails
    runtime.kill.write(active=True, reason="test", flatten=True)
    _tick(world, runtime)
    assert btc.orders[order_id]["status"] == "PARTIALLY_FILLED"               # the cancellation failed ...
    assert btc.positions["BTCUSDT"] == 0                                      # ... and the position is closed
    btc.request_signed = original
    _tick(world, runtime)
    assert btc.orders[order_id]["status"] == "CANCELED"                       # cancellation is asked again


def test_a_kill_switch_cancels_a_resting_entry_even_before_anything_has_filled(tmp_path):
    world, runtime = _execution(tmp_path, execution="maker", ttl_bars=3)
    btc = world["fakes"]["BTCUSDT"]
    i = world["warm"]
    while not btc.orders:
        _advance(world, runtime, i)
        i += 1
    order_id = next(iter(btc.orders))
    assert btc.positions.get("BTCUSDT", D(0)) == 0 and btc.orders[order_id]["status"] == "NEW"
    runtime.kill.write(active=True, reason="test", flatten=True)
    _tick(world, runtime)
    assert btc.orders[order_id]["status"] == "CANCELED"                       # it can no longer fill (K2)
    assert _attempts(tmp_path) == []                                          # and there was nothing to close


# -- 3b: what a kill switch closes -----------------------------------------------------------------------------
@pytest.mark.parametrize("strategy_position", [True, False])
def test_a_kill_switch_closes_the_whole_venue_position_and_books_the_excess_as_foreign(tmp_path, strategy_position):
    # Until S5b the excess went into the ledger as a false opposite position that reconciliation kept reporting.
    # ADR-0012 point 5, extended by the tech lead on 2026-11-15 to every reduction: it is a foreign fill now.
    if strategy_position:
        world, runtime, i, btc, session = _with_position(tmp_path)
    else:
        world, runtime = _start(tmp_path)                                     # no bar has closed yet: flat
        btc, session = world["fakes"]["BTCUSDT"], runtime.sessions["BTCUSDT"]
    owned = session.position_qty()
    extra = D("0.005") if owned >= 0 else D("-0.005")
    btc.entry["BTCUSDT"] = btc.mark("BTCUSDT")
    btc.positions["BTCUSDT"] = btc.positions.get("BTCUSDT", D(0)) + extra       # a trade the ledger does not know
    runtime.kill.write(active=True, reason="test", flatten=True)
    _tick(world, runtime)
    assert btc.positions["BTCUSDT"] == 0
    assert session.position_qty() == 0                                        # the ledger closes only its part
    _tick(world, runtime)
    (foreign,) = session.executor.journal.fills("foreign:BTCUSDT")
    assert D(foreign["qty"]) == abs(extra)                                    # the rest is a foreign fill
    assert any(r.get("issue_kind") == "foreign_fill" for r in _events(tmp_path, "issue"))
    assert session.state is SymbolState.HALTED                                # and entries stay blocked


def test_a_breached_stop_closes_the_whole_venue_position(tmp_path):
    # Until S5b a breached stop closed only what the strategy owned; ADR-0012 point 6 makes it close the venue
    # position, and the excess is a foreign fill (point 5).
    world, runtime, i, btc, session = _with_position(tmp_path)
    owned = session.position_qty()
    extra = D("0.005") if owned > 0 else D("-0.005")
    btc.positions["BTCUSDT"] += extra
    session.flatten("stop_breached")
    assert btc.positions["BTCUSDT"] == 0 and session.position_qty() == 0


# -- 7b: a refusal by the order journal is visible ---------------------------------------------------------
def test_a_reduction_refused_by_the_order_journal_is_logged_with_its_reason(tmp_path):
    world, runtime, i, btc, session = _with_position(tmp_path)
    side = "sell" if session.position_qty() > 0 else "buy"
    btc.fault = "error_5xx"
    session.submit([Intent(side, D("0.001"), True)], session.bars[-1], "first")      # stays unresolved
    session.submit([Intent(side, D("0.001"), True)], session.bars[-1], "second")
    blocked = [r for r in _events(tmp_path, "blocked") if r["symbol"] == "BTCUSDT"]
    assert [r["reason"] for r in blocked] == [f"journal: {REDUCTION_OVERLAP}"]
    assert blocked[0]["order_id"].endswith(":second:0")


# -- the service: closing the venue position -------------------------------------------------------------------
def _executor(tmp_path, venue):
    journal = SqliteOrderJournal(state_db(tmp_path / "orders.sqlite"), SCOPE)
    orders = OrderLifecycle(venue, journal, SCOPE, clock=lambda: 1.0,
                            sleeper=lambda s: None, lock=threading.RLock(), wait_seconds=0, owner=OWNER,
                            entry_order_type="market", not_found_grace_seconds=90)
    service = ExecutionService(venue, orders, owner=OWNER, entry_mode="market", limit_offset=D("0.0002"),
                               lock=threading.RLock(), clock=lambda: 1.0, monotonic=lambda: 1.0)
    executor = OwnerExecutor(service, scope=SCOPE)
    assert executor.reconcile_account("BTCUSDT") == []
    return executor


def _close(n, *, venue_position, qty="0"):
    return OrderIntent(f"BTCUSDT:{n}:kill_switch:0", OWNER, "BTCUSDT", "sell", D(qty), D("100"), True,
                       "kill_switch", close_venue_position=venue_position)


@pytest.mark.parametrize("venue_position,closed", [(True, D("3")), (False, D("1"))])
def test_closing_the_venue_position_ignores_the_owner_cap_and_a_normal_reduction_keeps_it(
        tmp_path, venue_position, closed):
    venue = Venue()
    executor = _executor(tmp_path, venue)
    consumer = TargetFillConsumer(executor)
    consumer.consume(executor.send_order(OrderIntent("BTCUSDT:0:rebalance:0", OWNER, "BTCUSDT", "buy", D("1"),
                                                     D("100"), False, "rebalance"))[0])
    venue.position += D("2")                                                  # the venue holds more than owned
    (fill,) = executor.send_order(_close(1, venue_position=venue_position))
    assert (fill.side, fill.qty) == ("sell", closed)


@pytest.mark.parametrize("qty,reduce_only", [("1", True), ("0", False)])
def test_closing_the_venue_position_needs_a_reduce_only_intent_without_a_quantity(tmp_path, qty, reduce_only):
    venue = Venue()
    executor = _executor(tmp_path, venue)
    venue.position = D("2")
    intent = OrderIntent("BTCUSDT:1:kill_switch:0", OWNER, "BTCUSDT", "sell", D(qty), D("100"), reduce_only,
                         "kill_switch", close_venue_position=True)
    assert executor.send_order(intent) == () and venue.calls == []
