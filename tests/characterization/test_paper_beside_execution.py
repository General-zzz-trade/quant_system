"""S10 part 3 audit, finding H1 (tech lead 2026-11-16: "paper makes its own directory, but refuses an execution
directory"): paper never runs in, resets or re-creates the database of an execution state directory, and execution
never starts on a paper record's database.

The three scenarios are the auditor's (run1, run5, run6): paper started on an execution directory that is trading;
on one whose state.sqlite was deleted; and with --reset-paper-state on one holding an entry pause and a tripped
drawdown latch. Each is refused and leaves the execution database exactly as it was.
"""
from __future__ import annotations

import io
import json
import os
import sqlite3
from decimal import Decimal as D
from pathlib import Path

import pytest

from lab.strategy import StrategySpec
from runner.binance_target import TargetRuntime
from tests.unit.lab.test_target_runtime import _advance, _world
from trading.adapters.sqlite import schema
from trading.adapters.sqlite.drawdown_store import SqliteDrawdownGuard
from trading.adapters.sqlite.entry_pause import SqliteEntryPauses
from trading.infrastructure import state_dir as managed

SCOPE = "binance-testnet-ut"
EXECUTION_DIR = "execution state directory"


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.delenv("QUANT_MODEL_SIGN_KEY", raising=False)
    monkeypatch.setenv("QUANT_ALLOW_UNSIGNED_MODELS", "1")
    monkeypatch.setenv("RISK_SCOPE", SCOPE)
    monkeypatch.setenv("BINANCE_TESTNET", "1")


def _init(path):
    assert managed.main(["--state-dir", str(path), "init"], stdout=io.StringIO(),
                        env={"RISK_SCOPE": SCOPE, "BINANCE_TESTNET": "1"}) == 0


def _spec():
    return StrategySpec(lookbacks=(5, 10, 20), vol_lookback=10, target_vol=0.6, band=0.05, execution="market")


def _paper(world, state_dir, **kw):
    return TargetRuntime(spec_path=world["spec_path"], state_dir=state_dir, mode="paper", market=world["adapter"],
                         clock=lambda: world["clock"]["now"], **kw)


def _snapshot(directory):
    """Every file of the directory (its inode, and its bytes unless it is the database) and every database row."""
    files = {p.name: (os.stat(p).st_ino, None if p.name == schema.STATE_DB else p.read_bytes())
             for p in sorted(Path(directory).iterdir()) if p.is_file() and not p.name.endswith(("-wal", "-shm"))}
    rows = None
    if (Path(directory) / schema.STATE_DB).exists():
        db = sqlite3.connect(Path(directory) / schema.STATE_DB)
        try:
            rows = {t: db.execute(f"SELECT * FROM {t} ORDER BY 1").fetchall() for t in schema.TABLES}
        finally:
            db.close()
    return files, rows


@pytest.mark.parametrize("reset", [False, True])
def test_paper_refuses_an_execution_directory_that_is_trading(tmp_path, reset):
    world = _world(tmp_path, _spec())
    state = tmp_path / "exec"
    _init(state)
    execution = TargetRuntime(spec_path=world["spec_path"], state_dir=state, mode="execution",
                              market=world["adapter"], account=world["adapter"],
                              clock=lambda: world["clock"]["now"], testnet=True)
    execution.warmup()
    btc, i = world["fakes"]["BTCUSDT"], world["warm"]
    while btc.positions.get("BTCUSDT", D(0)) == 0:
        assert i < world["warm"] + 200
        _advance(world, execution, i)
        i += 1
    before = _snapshot(state)
    with pytest.raises(RuntimeError, match=EXECUTION_DIR):
        _paper(world, state, reset_paper=reset)
    assert _snapshot(state) == before
    _advance(world, execution, i)                                      # execution goes on with its own state
    assert execution.sessions["BTCUSDT"].position_qty() == btc.positions["BTCUSDT"] != 0
    managed.verify(state, SCOPE)


def test_paper_does_not_recreate_the_deleted_database_of_an_execution_directory(tmp_path):
    world = _world(tmp_path, _spec())
    state = tmp_path / "exec"
    _init(state)
    (state / schema.STATE_DB).unlink()
    with pytest.raises(RuntimeError, match=EXECUTION_DIR):
        _paper(world, state)
    assert not (state / schema.STATE_DB).exists()
    with pytest.raises(RuntimeError, match="state.sqlite is missing"):
        managed.verify(state, SCOPE)


def test_a_paper_reset_never_releases_an_execution_pause_or_latch(tmp_path):
    world = _world(tmp_path, _spec())
    state = tmp_path / "exec"
    _init(state)
    now = world["clock"]["now"]
    pauses = SqliteEntryPauses(state / schema.STATE_DB, clock=lambda: now)
    pauses.pause("BTCUSDT", "stop_triggered", triggers=["qsstop-test"])
    drawdown = SqliteDrawdownGuard(state / schema.STATE_DB, SCOPE, daily_pct=5.0, weekly_pct=12.0, clock=lambda: now)
    drawdown.observe(10000.0)
    drawdown.observe(9000.0)
    assert drawdown.latched()
    before = _snapshot(state)
    with pytest.raises(RuntimeError, match=EXECUTION_DIR):
        _paper(world, state, reset_paper=True)
    assert _snapshot(state) == before
    assert SqliteEntryPauses(state / schema.STATE_DB, clock=lambda: now).reason("BTCUSDT") is not None
    assert SqliteDrawdownGuard(state / schema.STATE_DB, SCOPE, daily_pct=5.0, weekly_pct=12.0,
                               clock=lambda: now).latched()


def test_execution_refuses_a_paper_record_s_database(tmp_path):
    world = _world(tmp_path, _spec())
    paper = _paper(world, tmp_path / "paper")
    paper.warmup()                                                     # a paper record with a saved state
    state = tmp_path / "exec"
    _init(state)
    (state / schema.STATE_DB).unlink()
    source, copy = sqlite3.connect(tmp_path / "paper" / schema.STATE_DB), sqlite3.connect(state / schema.STATE_DB)
    source.backup(copy)
    source.close()
    copy.close()
    with pytest.raises(RuntimeError, match="paper"):
        managed.verify(state, SCOPE)


def test_a_database_without_a_paper_state_does_not_send_a_person_to_reset(tmp_path):
    from trading.adapters.sqlite.paper_state import SqlitePaperState
    schema.create(tmp_path / schema.STATE_DB)                         # an execution database's tables only
    with pytest.raises(RuntimeError) as refused:
        SqlitePaperState(tmp_path / schema.STATE_DB).load()
    assert "reset" not in str(refused.value) and "paper" in str(refused.value)


@pytest.mark.parametrize("execution,expected", [(False, "data/runtime/paper"), (True, "data/runtime/binance_target")])
def test_paper_and_execution_have_different_default_state_directories(tmp_path, monkeypatch, execution, expected):
    import runner.binance_target as entry
    from tests.unit.binance.fake_venue import FakeBinance, make_bars
    from trading.adapters.binance import adapter as binance
    from trading.infrastructure import bootstrap
    fake = FakeBinance(make_bars(10), clock=lambda: 0.0)

    class Adapter(binance.BinanceAdapter):
        def __init__(self, config, client=None):
            super().__init__(config, client=fake)
    seen = {}

    class Started(Exception):
        pass

    def stand_in(**kwargs):
        seen.update(kwargs)
        raise Started()
    monkeypatch.setattr(bootstrap, "BinanceAdapter", Adapter)
    monkeypatch.setattr(entry, "TargetRuntime", stand_in)
    monkeypatch.setenv("BINANCE_TESTNET_API_KEY", "k")
    monkeypatch.setenv("BINANCE_TESTNET_API_SECRET", "s")
    monkeypatch.chdir(tmp_path)
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"schema": 1, "kind": "btc_eth_multitimeframe_target", "symbols": ["BTCUSDT"],
                                "spec": _spec().record(), "passed": False}))
    if execution:
        with pytest.raises(RuntimeError, match=f"{expected} is not initialized"):
            entry.main(["--spec", str(spec), "--enable-execution"])
    else:
        with pytest.raises(Started):
            entry.main(["--spec", str(spec)])
        assert seen["state_dir"] == Path(expected)


def test_the_paper_entry_point_refuses_an_execution_directory_before_creating_anything_in_it(tmp_path, monkeypatch):
    import runner.binance_target as entry
    state = tmp_path / "exec"
    _init(state)
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"schema": 1, "kind": "btc_eth_multitimeframe_target", "symbols": ["BTCUSDT"],
                                "spec": _spec().record(), "passed": False}))
    before = _snapshot(state)
    with pytest.raises(RuntimeError, match=EXECUTION_DIR):
        entry.main(["--spec", str(spec), "--state-dir", str(state)])                # paper (--dry-run)
    assert _snapshot(state) == before                                               # not even the lease's lock


def test_a_journal_refusal_event_names_the_refusal_itself():
    from types import SimpleNamespace
    from trading.application.session import ExecutionParts, TargetSession
    from trading.domain.events import EventKind
    from trading.domain.orders import DUPLICATE, OrderIntent
    from trading.domain.rebalance import Intent, Lot
    events = []
    executor = SimpleNamespace(send_order=lambda order: (), take_refusal=lambda intent_id: DUPLICATE,
                               notify_status=lambda: None, reconcile_pending=lambda: ())
    session = TargetSession("BTCUSDT", _spec(), journal=SimpleNamespace(write=lambda k, **f: events.append((k, f))),
                            rules=Lot(D("0.001"), D("0.001"), D("5")), make_order=OrderIntent,
                            execution=ExecutionParts(executor, None, None, None, None))
    order = OrderIntent(order_id="BTCUSDT:1:rebalance:0", source="target.BTCUSDT", symbol="BTCUSDT", side="buy",
                        qty=D("0.1"), price=D("100"), reduce_only=False, reason="rebalance")
    session._send(order, Intent("buy", D("0.1"), False))
    assert [(k, f.get("refusal")) for k, f in events] == [(EventKind.BLOCKED, DUPLICATE)]
