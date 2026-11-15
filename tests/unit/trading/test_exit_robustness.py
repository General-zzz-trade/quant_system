"""Exits that keep working when something else fails (audit after step 12b, findings 1, 6, 7, 9 and two gaps).

ADR-0005 7b already said it: the emergency close sizes from the venue, so it does not depend on the ledger
checkpoint. A fill that cannot be consumed must therefore neither stop the close nor end the process.
"""
from __future__ import annotations

import json
import threading
import time
from decimal import Decimal as D

import pytest

from tests.characterization.test_exit_paths import _execution, _tick
from tests.characterization.test_runtime_decisions import _events, _start, _until_position
from tests.unit.lab.test_target_runtime import _advance
from trading.adapters.notify.telegram import TelegramNotifier
from trading.adapters.sqlite.order_journal import SqliteOrderJournal
from trading.domain.orders import REDUCTION_OVERLAP, STALE_CHECKPOINT, UNCONSUMED_OWN_FILL

OWNER, SCOPE = "target.BTCUSDT", "binance-testnet-ut"


@pytest.fixture(autouse=True)
def _offline_env(monkeypatch):
    for name in ("QUANT_MODEL_SIGN_KEY", "BINANCE_TESTNET", "BINANCE_EXCHANGE_LEVERAGE", "BINANCE_MARGIN_TYPE",
                 "MAX_DAILY_DRAWDOWN_PCT", "MAX_WEEKLY_DRAWDOWN_PCT", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("QUANT_ALLOW_UNSIGNED_MODELS", "1")
    monkeypatch.setenv("RISK_SCOPE", SCOPE)


def _with_position(tmp_path):
    world, runtime = _start(tmp_path)
    i = _until_position(world, runtime)
    _advance(world, runtime, i)
    return world, runtime, world["fakes"]["BTCUSDT"], runtime.sessions["BTCUSDT"]


def _poison(session):
    """An own fill the consumer can never take, as a mismatched or invalid exchange fact would be."""
    journal, owner = session.executor.journal, session.executor.owner
    with journal.connect() as db:
        row = db.execute("SELECT intent_id, payload FROM execution_fills WHERE owner=? LIMIT 1", (owner,)).fetchone()
    payload = dict(json.loads(row["payload"]), fill_id="poison-1")
    journal.insert_fill(fill_id="poison-1", intent_id=row["intent_id"], owner=owner, payload=payload)

    def refuse(event):
        raise RuntimeError("cannot consume")

    session.consumer.consume = refuse


# -- finding 1 ------------------------------------------------------------------------------------------------
def test_a_fill_that_cannot_be_consumed_neither_stops_the_kill_close_nor_ends_the_process(tmp_path):
    world, runtime, btc, session = _with_position(tmp_path)
    _poison(session)
    runtime.kill.write(active=True, reason="test", flatten=True)
    for _ in range(3):
        _tick(world, runtime, 31)                                         # never raises
    assert btc.positions["BTCUSDT"] == 0
    issues = [r["issue"] for r in _events(tmp_path, "issue") if r.get("symbol") == "BTCUSDT"]
    assert "flatten: fill consumption failed (RuntimeError: cannot consume)" in issues     # reported where it
    assert not [i for i in issues if "flatten failed" in i]                  # happens; the flatten itself completed


def test_a_flatten_that_raises_is_reported_and_the_cycle_goes_on(tmp_path):
    world, runtime, btc, session = _with_position(tmp_path)

    def broken(reason, **kw):
        raise OSError("journal unreadable")

    session.flattener.flatten = broken
    runtime.kill.write(active=True, reason="test", flatten=True)
    _tick(world, runtime)
    _tick(world, runtime)
    issues = [r["issue"] for r in _events(tmp_path, "issue") if r.get("symbol") == "BTCUSDT"]
    assert "BTCUSDT: flatten failed: OSError: journal unreadable" in issues
    assert runtime.sessions["ETHUSDT"].position_qty() == 0                    # the other symbol was still closed


def test_the_flatten_goes_ahead_when_cancelling_entries_raises(tmp_path):
    world, runtime, btc, session = _with_position(tmp_path)

    def broken(symbol):
        raise OSError("cancel failed")

    session.executor.cancel_entries = broken
    runtime.kill.write(active=True, reason="test", flatten=True)
    _tick(world, runtime)
    assert btc.positions["BTCUSDT"] == 0


def test_an_emergency_close_sends_nothing_when_the_venue_position_cannot_be_read(tmp_path):
    world, runtime, btc, session = _with_position(tmp_path)
    position, posts = btc.positions["BTCUSDT"], len(btc.posts)
    btc.position_error = True
    runtime.kill.write(active=True, reason="test", flatten=True)
    _tick(world, runtime)
    assert btc.positions["BTCUSDT"] == position and len(btc.posts) == posts
    assert any("venue position unavailable" in r["issue"] for r in _events(tmp_path, "issue"))
    assert not [r for r in _events(tmp_path, "intent") if r["symbol"] == "BTCUSDT" and r["reason"] == "kill_switch"]


def test_the_journal_lets_a_venue_close_past_the_ledger_rules_but_never_past_another_reduction(tmp_path):
    journal = SqliteOrderJournal(tmp_path / "orders.sqlite", SCOPE)
    common = dict(symbol="BTCUSDT", side="sell", qty="0", reducing=True, owner=OWNER, order_type="market",
                  clock=lambda: 1.0)
    journal.reserve(intent_id="e0", client_id="c0", parent=None, **{**common, "side": "buy", "qty": "1",
                                                                    "reducing": False})
    journal.update_status("e0", "filled", settle=True)
    journal.insert_fill(fill_id="f0", intent_id="e0", owner=OWNER, payload={"fill_id": "f0"})   # not consumed
    stale = {"consumer_revision": "old"}
    assert journal.reserve(intent_id="r1", client_id="c1", parent=stale, **common)[0] == UNCONSUMED_OWN_FILL
    assert journal.reserve(intent_id="v1", client_id="v1", parent=stale, venue_close=True, **common)[0] == ""
    assert journal.reserve(intent_id="v2", client_id="v2", parent=None, venue_close=True, **common)[0] == \
        REDUCTION_OVERLAP
    journal.update_status("v1", "filled", settle=True)
    journal.consume(OWNER, "f0", lambda: {"position": {"qty": "1", "entry_price": "1", "realized_pnl": "0",
                                                         "fees": {}}, "revision": "f0", "strategy": {}},
                    lambda: None, expected_revision=None)
    assert journal.reserve(intent_id="r2", client_id="c2", parent=stale, **common)[0] == STALE_CHECKPOINT


# -- finding 6 ------------------------------------------------------------------------------------------------
def test_a_kill_switch_without_flatten_cancels_resting_entries_and_keeps_the_position(tmp_path):
    world, runtime = _execution(tmp_path, execution="maker", ttl_bars=3)
    btc = world["fakes"]["BTCUSDT"]
    i = world["warm"]
    while not btc.orders:
        _advance(world, runtime, i)
        i += 1
    order_id = next(iter(btc.orders))
    btc.fill_limit(order_id, D("0.001"))
    _tick(world, runtime)
    position = btc.positions["BTCUSDT"]
    runtime.kill.write(active=True, reason="pause entries", flatten=False)
    _tick(world, runtime)
    assert btc.orders[order_id]["status"] == "CANCELED"                       # no new exposure can fill
    assert btc.positions["BTCUSDT"] == position                               # and nothing is closed


def test_a_paused_symbol_has_its_resting_entries_cancelled(tmp_path):
    world, runtime = _execution(tmp_path, execution="maker", ttl_bars=3)
    btc = world["fakes"]["BTCUSDT"]
    i = world["warm"]
    while not btc.orders:
        _advance(world, runtime, i)
        i += 1
    order_id = next(iter(btc.orders))
    runtime.pauses.pause("BTCUSDT", "protective stop triggered")
    _tick(world, runtime)
    assert btc.orders[order_id]["status"] == "CANCELED"


# -- finding 7 ------------------------------------------------------------------------------------------------
def test_the_runtime_takes_its_lease_with_the_same_scope_the_tools_use(tmp_path, monkeypatch):
    import runner.binance_target as target
    from lab.strategy import StrategySpec
    seen = {}

    class Stop(Exception):
        pass

    def capture(journal_path, scope):
        seen["scope"] = scope
        raise Stop

    monkeypatch.setattr(target, "ExecutionLease", capture)
    monkeypatch.setenv("RISK_SCOPE", " binance-testnet-ut ")
    monkeypatch.setenv("BINANCE_TESTNET", "1")
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"schema": 1, "kind": "btc_eth_multitimeframe_target", "symbols": ["BTCUSDT"],
                                "spec": StrategySpec().record(), "passed": False}))
    with pytest.raises(Stop):
        target.main(["--spec", str(spec), "--enable-execution", "--state-dir", str(tmp_path / "state")])
    assert seen["scope"] == "binance-testnet-ut"


# -- finding 9 ------------------------------------------------------------------------------------------------
def test_a_slow_telegram_never_holds_up_the_caller():
    release, sent = threading.Event(), []

    def slow(url, body):
        release.wait(5)
        sent.append(json.loads(body)["text"])

    notifier = TelegramNotifier("t", "42", post=slow, background=True)
    started = time.monotonic()
    for k in range(3):
        notifier.alert("critical", f"alert {k}")
    assert time.monotonic() - started < 0.5                                   # the loop is not held up
    release.set()
    notifier.flush(timeout=5)
    assert sent == ["[CRITICAL] alert 0", "[CRITICAL] alert 1", "[CRITICAL] alert 2"]


def test_a_full_telegram_queue_drops_new_alerts_and_logs_it(caplog):
    gate = threading.Event()
    notifier = TelegramNotifier("t", "42", post=lambda url, body: gate.wait(5), background=True, queue_size=2)
    for k in range(6):
        notifier.alert("warning", f"a{k}")
    assert "alert dropped" in caplog.text
    gate.set()
    notifier.flush(timeout=5)


def test_the_runtime_s_telegram_notifier_sends_in_the_background():
    from trading.infrastructure.settings import notifier_from_env
    assert notifier_from_env({"TELEGRAM_BOT_TOKEN": "t", "TELEGRAM_CHAT_ID": "42"}).background is True
