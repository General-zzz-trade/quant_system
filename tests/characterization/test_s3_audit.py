"""The independent audit of S3 (2026-11-15): one test group per finding, each fails on the audited head f0730f2.
The tech lead decided finding 3 (2026-11-15): an event log that cannot be written never stops trading, but
blocks every entry until it works again."""
from __future__ import annotations

import time
from decimal import Decimal as D

from tests.characterization.test_exit_paths import _execution, _tick, _until
from tests.characterization.test_runtime_decisions import _events
from tests.unit.lab.test_target_runtime import _advance, _world
from trading.application.issues import Issue
from trading.domain.issues import IssueKind
from trading.domain.strategy.overlay import StrategySpec
from trading.domain.symbol_state import SymbolState as S

SPEC = dict(lookbacks=(5, 10, 20), vol_lookback=10, target_vol=0.6, band=0.05, execution="market")


class _Notifier:
    channel = "test"

    def __init__(self):
        self.sent = []

    def alert(self, level, title, detail=""):
        self.sent.append((level, title, detail))


def _unwarmed(tmp_path, notifier=None):
    from runner.binance_target import TargetRuntime
    from tests.support.state_db import state_dir_db
    world = _world(tmp_path, StrategySpec(**SPEC))
    state_dir_db(tmp_path / "exec")
    runtime = TargetRuntime(spec_path=world["spec_path"], state_dir=tmp_path / "exec", mode="execution",
                            market=world["adapter"], account=world["adapter"], clock=lambda: world["clock"]["now"],
                            testnet=True, notifier=notifier)
    return world, runtime


def _blocked(tmp_path, symbol, reason_prefix):
    return [r for r in _events(tmp_path, "blocked") if r["symbol"] == symbol and r["reason"].startswith(reason_prefix)]


# -- finding 1: the state the gate reads must be this cycle's, not the last cycle's -----------------------------
def test_after_a_pause_is_released_the_next_bar_enters_and_is_not_refused_as_paused(tmp_path):
    from tests.unit.trading.test_entry_pause import _breach, _release
    world, runtime = _execution(tmp_path, execution="market")
    btc = world["fakes"]["BTCUSDT"]
    i = _until(world, runtime, lambda: btc.positions.get("BTCUSDT", D(0)) != 0)
    _advance(world, runtime, i)
    qty = _breach(world, runtime, i + 1)
    btc.price.pop("BTCUSDT")
    for k in range(2, 4):
        _advance(world, runtime, i + k)
    assert _release(tmp_path / "exec", "BTCUSDT", typed="BTCUSDT", clock=lambda: world["clock"]["now"])[0] == 0
    time.sleep(1.2)                                       # no per-second deferral: BTC decides first on this bar
    _advance(world, runtime, i + 4)
    assert _blocked(tmp_path, "BTCUSDT", "state:") == []
    time.sleep(1.1)
    _tick(world, runtime)
    assert btc.positions["BTCUSDT"] == qty


def test_after_the_kill_switch_is_cleared_the_next_bar_is_not_refused_as_halted(tmp_path):
    from tests.characterization.test_cycle_isolation import _first_entry_bar
    entry = _first_entry_bar(tmp_path)
    world, runtime = _execution(tmp_path, execution="market")
    for i in range(world["warm"], entry - 1):
        _advance(world, runtime, i)
    runtime.kill.write(active=True, reason="test")
    _advance(world, runtime, entry - 1)                                # HALTED for this bar
    runtime.kill.write(active=False, reason="cleared")
    time.sleep(1.2)                                                   # no per-second deferral on the entry bar
    _advance(world, runtime, entry)
    assert _blocked(tmp_path, "BTCUSDT", "state:") == []
    time.sleep(1.1)
    _tick(world, runtime)
    assert world["fakes"]["BTCUSDT"].positions.get("BTCUSDT", D(0)) != 0


# -- finding 2: a block-all problem on one symbol blocks the symbols reconciled after it --------------------------
def test_a_block_all_problem_raised_on_one_symbol_blocks_every_symbol(tmp_path, monkeypatch):
    world, runtime = _execution(tmp_path, execution="market")
    btc = runtime.sessions["BTCUSDT"]
    monkeypatch.setattr(btc.executor, "reconcile_account",
                        lambda symbol, positions=None: [Issue(IssueKind.MARKET_REFUSED, symbol, f"{symbol}: probe")])
    _tick(world, runtime)
    eth = runtime.sessions["ETHUSDT"]
    assert eth.executor.ready is False and eth.state is S.BLOCKED


# -- finding 3 (tech lead's decision): an unwritable event log never stops trading but blocks every entry -----
def test_an_unwritable_event_log_lets_exits_run_and_blocks_entries_until_it_works_again(tmp_path):
    note = _Notifier()
    world, runtime = _unwarmed(tmp_path, note)
    runtime.warmup()
    btc = world["fakes"]["BTCUSDT"]
    i = _until(world, runtime, lambda: btc.positions.get("BTCUSDT", D(0)) != 0)
    events = tmp_path / "exec/target_events.jsonl"
    events.rename(tmp_path / "events.kept")
    events.mkdir()                                                    # nothing can be written to the event log
    runtime.kill.write(active=True, reason="test", flatten=True)
    _advance(world, runtime, i)                                       # must not raise
    assert btc.positions["BTCUSDT"] == 0                              # the flatten went out without its event
    runtime.kill.write(active=False, reason="cleared")
    _advance(world, runtime, i + 1)
    assert not any(s.executor.ready for s in runtime.sessions.values())
    assert any("event log" in detail for _, _, detail in note.sent)
    events.rmdir()
    _advance(world, runtime, i + 2)
    _advance(world, runtime, i + 3)
    assert all(s.executor.ready for s in runtime.sessions.values())


def test_the_guarded_event_log_swallows_a_failure_and_reports_it_until_a_write_succeeds():
    from trading.application.event_guard import GuardedEventLog
    from trading.domain.events import EventKind

    class Inner:
        fail = True

        def write(self, kind, **fields):
            if self.fail:
                raise OSError("disk full")
    inner = Inner()
    log = GuardedEventLog(inner)
    log.write(EventKind.INTENT, symbol="BTCUSDT")                    # never raises
    issue = log.failure()
    assert issue is not None and issue.kind is IssueKind.EVENT_LOG_FAILED and "disk full" in issue
    inner.fail = False
    log.write(EventKind.INTENT, symbol="BTCUSDT")
    assert log.failure() is None


# -- finding 4: one symbol's malformed market data stops only that symbol ----------------------------------------
def test_one_symbol_s_malformed_market_data_stops_only_that_symbol(tmp_path, monkeypatch):
    world, runtime = _execution(tmp_path, execution="market")
    get_klines = world["adapter"].get_klines

    def klines(symbol, *args, **kwargs):
        if symbol == "BTCUSDT":
            raise ValueError("malformed kline row")
        return get_klines(symbol, *args, **kwargs)
    monkeypatch.setattr(world["adapter"], "get_klines", klines)
    eth = runtime.sessions["ETHUSDT"]
    cursor = eth.cursor
    for i in range(world["warm"], world["warm"] + 3):
        _advance(world, runtime, i)
    assert eth.cursor > cursor and eth.executor.ready is True
    assert runtime.sessions["BTCUSDT"].executor.ready is False


def test_a_market_error_raised_inside_a_bar_backs_off_like_any_market_error(tmp_path, monkeypatch):
    from trading.adapters.binance.rest import BinanceRetryableError
    world, runtime = _execution(tmp_path, execution="market")

    def busy(*args, **kwargs):
        raise BinanceRetryableError("HTTP 503: busy")
    monkeypatch.setattr(runtime.sessions["BTCUSDT"], "process_bar", busy)
    _advance(world, runtime, world["warm"])
    assert runtime.market_side.failures == 1 and runtime.market_side.retry_at > world["clock"]["now"]


# -- findings 5 and 6: the start-up hold is isolated, and nothing is READY before the whole warm-up --------------
def test_the_start_up_hold_survives_an_unwritable_status_and_keeps_every_symbol_warming(tmp_path, monkeypatch):
    from trading.adapters.binance.rest import BinanceRetryableError
    note = _Notifier()
    world, runtime = _unwarmed(tmp_path, note)
    get_funding = world["adapter"].get_funding_history

    def funding(symbol, *args, **kwargs):
        if symbol == "ETHUSDT":
            raise BinanceRetryableError("HTTP 503: busy")
        return get_funding(symbol, *args, **kwargs)
    monkeypatch.setattr(world["adapter"], "get_funding_history", funding)
    status = tmp_path / "exec/target_status.json"
    status.mkdir(parents=True)
    for _ in range(3):
        assert runtime.warmup_or_hold() is False                      # must not raise
        assert {s.state for s in runtime.sessions.values()} == {S.WARMING}
        world["clock"]["now"] += 400                                  # past the market backoff
    assert not any("is not allowed" in detail for _, _, detail in note.sent)
    status.rmdir()
    monkeypatch.setattr(world["adapter"], "get_funding_history", get_funding)
    assert runtime.warmup_or_hold() is True
    # The status side failed during the hold: entries stay blocked until it has written once and the next
    # reconciliation no longer sees the failure.
    _tick(world, runtime)
    _tick(world, runtime)
    assert {s.state for s in runtime.sessions.values()} == {S.READY}


# -- finding 8: every input of the symbol state is pinned -------------------------------------------------------
def test_the_kill_flatten_shows_flattening_and_halted_within_one_cycle(tmp_path):
    world, runtime = _execution(tmp_path, execution="market")
    btc = world["fakes"]["BTCUSDT"]
    _until(world, runtime, lambda: btc.positions.get("BTCUSDT", D(0)) != 0)
    runtime.kill.write(active=True, reason="test", flatten=True)
    _tick(world, runtime)
    assert btc.positions["BTCUSDT"] == 0 and runtime.sessions["BTCUSDT"].state is S.HALTED
    trail = [r["state"] for r in _events(tmp_path, "symbol_state") if r["symbol"] == "BTCUSDT"]
    assert trail[-2:] == ["flattening", "halted"]


def test_a_paused_symbol_shows_paused(tmp_path):
    world, runtime = _execution(tmp_path, execution="market")
    runtime.pauses.pause("BTCUSDT", "test pause", triggers=[])
    _tick(world, runtime)
    assert runtime.sessions["BTCUSDT"].state is S.PAUSED and runtime.sessions["ETHUSDT"].state is S.READY


def test_a_latched_drawdown_with_a_position_shows_flattening_first(tmp_path):
    world, runtime = _execution(tmp_path, execution="market")
    btc = world["fakes"]["BTCUSDT"]
    i = _until(world, runtime, lambda: btc.positions.get("BTCUSDT", D(0)) != 0)
    for fake in world["fakes"].values():
        fake.equity = D("7000")
    for k in range(3):
        _advance(world, runtime, i + k)
    trail = [r["state"] for r in _events(tmp_path, "symbol_state") if r["symbol"] == "BTCUSDT"]
    assert "flattening" in trail and trail[-1] == "halted"


def test_a_breached_stop_whose_flatten_is_rejected_shows_flattening(tmp_path):
    from tests.unit.trading.test_entry_pause import _breach
    world, runtime = _execution(tmp_path, execution="market")
    btc = world["fakes"]["BTCUSDT"]
    i = _until(world, runtime, lambda: btc.positions.get("BTCUSDT", D(0)) != 0)
    _advance(world, runtime, i)
    btc.fault, btc.fault_code = "reject", -1008                        # the flatten is rejected
    _breach(world, runtime, i + 1)
    assert btc.positions["BTCUSDT"] != 0 and runtime.sessions["BTCUSDT"].state is S.FLATTENING


def test_unknown_equity_shows_blocked(tmp_path, monkeypatch):
    world, runtime = _execution(tmp_path, execution="market")

    def unavailable():
        raise ConnectionError("equity unavailable")
    monkeypatch.setattr(world["adapter"], "get_account_equity", unavailable)
    world["clock"]["now"] += 61                                       # the equity refresh is due
    _tick(world, runtime)
    assert {s.state for s in runtime.sessions.values()} == {S.BLOCKED}
