"""Rules the audit after M3 found untested (its surviving mutants R16, R07, R02, R08, R14, T04, P01, R11).

- A failing exit is sent to a person after two minutes, and its end is reported.
- A failed warm-up is retried only after its backoff; an error the exchange calls final waits the longest
  backoff every time and is alerted once.
- A stop that triggered and still waits for its order already pauses entries.
- A trigger that arrives while the symbol is already paused is recorded, so a person's release holds.
- When a breached stop and a plain trigger arrive together, the pause names the breach.
"""
from __future__ import annotations

import pytest

from tests.characterization.test_exit_paths import _tick
from tests.characterization.test_runtime_decisions import _events, _start, _until_position
from tests.characterization.test_runtime_decisions import _runtime as _restart
from tests.characterization.test_stops_and_fills import _with_position
from tests.unit.trading.test_alerts import Recorder
from tests.unit.trading.test_alerts import _runtime as _alerting_runtime
from tests.unit.trading.test_market_failures import _due, _market
from trading.adapters.binance.rest import BinanceNonRetryableError, BinanceRetryableError
from trading.adapters.sqlite.entry_pause import SqliteEntryPauses
from trading.application.runtime import MARKET_BACKOFF_MAX, POLL_SECONDS


@pytest.fixture(autouse=True)
def _offline_env(monkeypatch):
    for name in ("QUANT_MODEL_SIGN_KEY", "BINANCE_TESTNET", "BINANCE_EXCHANGE_LEVERAGE", "BINANCE_MARGIN_TYPE",
                 "MAX_DAILY_DRAWDOWN_PCT", "MAX_WEEKLY_DRAWDOWN_PCT", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("QUANT_ALLOW_UNSIGNED_MODELS", "1")
    monkeypatch.setenv("RISK_SCOPE", "binance-testnet-ut")


# -- R16, R07 -----------------------------------------------------------------------------------------------------
def test_a_failing_exit_is_sent_after_two_minutes_and_its_end_is_reported(tmp_path):
    notifier = Recorder()
    world, runtime = _alerting_runtime(tmp_path, notifier, execution="market")
    _until_position(world, runtime)
    session = runtime.sessions["BTCUSDT"]
    working = session.flattener.flatten

    def broken(reason, **kw):
        raise OSError("journal unreadable")

    session.flattener.flatten = broken
    runtime.kill.write(active=True, reason="test", flatten=True)
    for _ in range(25):                                                       # 150 s of failing closes
        _tick(world, runtime)
    warnings = [d for level, t, d in notifier.sent if level == "warning" and "BTCUSDT" in t]
    assert any("flatten failed: OSError: journal unreadable" in d for d in warnings)
    session.flattener.flatten = working
    for _ in range(2):
        _tick(world, runtime)
    assert any(level == "info" and d.startswith("resolved:") and "flatten failed" in d
               for level, _, d in notifier.sent)


# -- R02 ------------------------------------------------------------------------------------------------------------
def test_a_failed_warm_up_is_retried_only_after_its_backoff(tmp_path):
    world, runtime = _start(tmp_path)
    runtime = _restart(tmp_path, world)                                       # a restart while market data is down
    calls = []

    def down(*, method, path, params=None):
        if "klines" in path or "funding" in path.lower():
            calls.append(path)
        raise BinanceRetryableError("Network error: down")

    for fake in world["fakes"].values():
        fake.request_public = down
    assert runtime.warmup_or_hold() is False
    first = len(calls)
    assert first > 0
    assert runtime.warmup_or_hold() is False                                  # the same instant: still backing off
    assert len(calls) == first and runtime.status()["market_failures"] == 1
    world["clock"]["now"] += POLL_SECONDS
    assert runtime.warmup_or_hold() is False
    assert len(calls) > first and runtime.status()["market_failures"] == 2


# -- R08, R14 ---------------------------------------------------------------------------------------------------------
def test_a_final_error_waits_the_longest_backoff_every_time_and_is_alerted_once(tmp_path):
    world, runtime = _start(tmp_path)
    _market(world, BinanceNonRetryableError('HTTP 400: {"code":-1121,"msg":"Invalid symbol."}', http_status=400,
                                            code=-1121))
    _due(world, world["warm"])
    for _ in range(4):
        runtime.cycle()
        world["clock"]["now"] += MARKET_BACKOFF_MAX + 1
    failures = [r for r in _events(tmp_path, "issue") if r["issue"].startswith("market data unavailable")]
    assert len(failures) >= 3 and {r["retry_in"] for r in failures} == {MARKET_BACKOFF_MAX}
    assert len([r for r in _events(tmp_path, "alert") if "-1121" in r["reason"]]) == 1


# -- T04 --------------------------------------------------------------------------------------------------------------
def test_a_stop_that_triggered_and_awaits_its_order_already_pauses(tmp_path):
    fake, executor, consumer, stops = _with_position(tmp_path)
    stops.sync()
    (client_id,) = list(fake.algo)
    fake.algo[client_id]["algoStatus"] = "TRIGGERED"                          # no order id yet
    stops.sync()
    assert stops.triggers() == {client_id: "protective stop triggered"}


# -- P01 --------------------------------------------------------------------------------------------------------------
def test_a_trigger_that_arrives_during_a_pause_is_recorded_so_a_release_holds(tmp_path):
    pauses = SqliteEntryPauses(tmp_path / "pauses.sqlite")
    assert pauses.pause("BTCUSDT", "protective stop triggered", triggers=["stop-a"]) is True
    assert pauses.pause("BTCUSDT", "protective stop triggered", triggers=["stop-b"]) is False   # already paused
    assert pauses.covered("BTCUSDT") == {"stop-a", "stop-b"}
    assert pauses.release("BTCUSDT", "checked") == 1
    assert pauses.reason("BTCUSDT") is None and pauses.covered("BTCUSDT") == {"stop-a", "stop-b"}


# -- R11 --------------------------------------------------------------------------------------------------------------
def test_a_breach_names_the_pause_when_it_comes_with_a_plain_trigger(tmp_path):
    world, runtime = _start(tmp_path)
    session = runtime.sessions["BTCUSDT"]
    session.stops.triggers = lambda: {"stop-a": "protective stop triggered",
                                      "stop-b": "protective stop breached; position flattened"}
    _tick(world, runtime)
    assert runtime.pauses.reason("BTCUSDT") == "protective stop breached; position flattened"
