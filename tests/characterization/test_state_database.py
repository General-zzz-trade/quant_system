"""ADR-0009: one versioned state database per execution state directory.

- `state_dir init` creates exactly one database, `state.sqlite`, at version 1 with all ten tables; nothing else
  creates tables or database files.
- The start-up check refuses a wrong version, a missing table and the old five-file layout, and creates nothing.
- Every store opens the database through one connection setting (WAL, synchronous=FULL, a busy timeout).
- A database deleted while running is never re-created: entries stay blocked and no order is sent without
  being registered first (ADR-0009 acceptance, corrected 2026-11-15).
"""
from __future__ import annotations

import io
import sqlite3
from decimal import Decimal as D
from pathlib import Path

import pytest

from trading.infrastructure import state_dir as managed

ROOT = Path(__file__).resolve().parents[2]
SCOPE = "binance-testnet-ut"
TABLES = {"submissions", "execution_fills", "execution_consumers", "manual_actions", "protective_stops",
          "drawdown_periods", "equity_observations", "order_limits", "entry_pauses", "pause_triggers"}
OLD_FILES = ("orders.sqlite", "drawdown.sqlite", "order_limits.sqlite", "entry_pauses.sqlite")


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for name in ("QUANT_MODEL_SIGN_KEY", "BINANCE_TESTNET", "MAX_DAILY_DRAWDOWN_PCT", "MAX_WEEKLY_DRAWDOWN_PCT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("QUANT_ALLOW_UNSIGNED_MODELS", "1")
    monkeypatch.setenv("RISK_SCOPE", SCOPE)


def _init(path):
    assert managed.main(["--state-dir", str(path), "init"], stdout=io.StringIO(),
                        env={"RISK_SCOPE": SCOPE, "BINANCE_TESTNET": "1"}) == 0
    return path / "state.sqlite"


def _files(directory):
    return sorted(p.name for p in directory.iterdir() if not p.name.endswith(("-wal", "-shm")))


# -- initialization --------------------------------------------------------------------------------------------
def test_init_creates_one_state_database_at_version_1_with_every_table(tmp_path):
    db_path = _init(tmp_path / "state")
    assert [n for n in _files(tmp_path / "state") if n.endswith(".sqlite")] == ["state.sqlite"]
    with sqlite3.connect(db_path) as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == 1
        assert db.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert tables == TABLES
    managed.verify(tmp_path / "state", SCOPE)


# -- the start-up check ----------------------------------------------------------------------------------------
@pytest.mark.parametrize("name,value", [("MAX_DAILY_DRAWDOWN_PCT", "10.5"), ("MAX_WEEKLY_DRAWDOWN_PCT", "26")])
def test_init_with_a_drawdown_limit_above_the_ceiling_refuses_before_anything_exists(tmp_path, name, value):
    # S2 audit finding 3 (mutants M2, M3): the ceiling check and its place before the directory is created.
    state = tmp_path / "state"
    with pytest.raises(RuntimeError, match="drawdown limit"):
        managed.initialize(state, scope=SCOPE, testnet=True, env={name: value})
    assert not state.exists()


def test_the_schema_refuses_an_existing_empty_file_and_leaves_it_empty(tmp_path):
    # S2 audit finding 3 (mutant M1): a 0-byte leftover is not silently turned into a database.
    from trading.adapters.sqlite import schema
    leftover = tmp_path / "state.sqlite"
    leftover.touch()
    with pytest.raises(FileExistsError, match="created once"):
        schema.create(leftover)
    assert leftover.stat().st_size == 0


def _set_version(path, version):
    with sqlite3.connect(path) as db:
        db.execute(f"PRAGMA user_version={version}")


@pytest.mark.parametrize("version", [0, 2])
def test_a_wrong_version_refuses_to_start_and_creates_nothing(tmp_path, version):
    db_path = _init(tmp_path / "state")
    _set_version(db_path, version)
    before = _files(tmp_path / "state")
    with pytest.raises(RuntimeError, match=f"state.sqlite has version {version}, expected 1"):
        managed.verify(tmp_path / "state", SCOPE)
    assert _files(tmp_path / "state") == before


@pytest.mark.parametrize("table", sorted(TABLES))
def test_a_missing_table_refuses_to_start(tmp_path, table):
    db_path = _init(tmp_path / "state")
    with sqlite3.connect(db_path) as db:
        db.execute(f"DROP TABLE {table}")
    with pytest.raises(RuntimeError, match=f"state.sqlite has lost its tables \\({table}\\)"):
        managed.verify(tmp_path / "state", SCOPE)


@pytest.mark.parametrize("old", OLD_FILES)
def test_the_old_five_file_layout_refuses_to_start_and_says_how_to_fix_it(tmp_path, old):
    _init(tmp_path / "state")
    (tmp_path / "state" / old).write_bytes(b"")
    with pytest.raises(RuntimeError, match="old state layout.*initialize a new directory"):
        managed.verify(tmp_path / "state", SCOPE)


def test_the_old_layout_hint_names_a_new_directory_not_the_old_one(tmp_path):
    # S2 audit finding 6: "init" on the old directory only says "already initialized; nothing changed".
    _init(tmp_path / "state")
    (tmp_path / "state" / "orders.sqlite").write_bytes(b"")
    with pytest.raises(RuntimeError) as refused:
        managed.verify(tmp_path / "state", SCOPE)
    assert "--state-dir <new directory> init" in str(refused.value)
    assert f"--state-dir {tmp_path / 'state'} init" not in str(refused.value)


def test_a_missing_database_refuses_to_start_and_is_not_re_created(tmp_path):
    db_path = _init(tmp_path / "state")
    db_path.unlink()
    with pytest.raises(RuntimeError, match="state.sqlite is missing"):
        managed.verify(tmp_path / "state", SCOPE)
    assert not db_path.exists()


# -- the stores ------------------------------------------------------------------------------------------------
def _stores(path):
    from trading.adapters.sqlite.drawdown_store import SqliteDrawdownGuard
    from trading.adapters.sqlite.entry_pause import SqliteEntryPauses
    from trading.adapters.sqlite.limit_store import SqliteLimitStore
    from trading.adapters.sqlite.order_journal import SqliteOrderJournal
    return {
        "journal": lambda: SqliteOrderJournal(path, SCOPE).orders(),
        "drawdown": lambda: SqliteDrawdownGuard(path, SCOPE).latched(),
        "limits": lambda: SqliteLimitStore(path, SCOPE).load("2026-11-15"),
        "pauses": lambda: SqliteEntryPauses(path).reason("BTCUSDT"),
    }


@pytest.mark.parametrize("store", ["journal", "drawdown", "limits", "pauses"])
def test_no_store_creates_the_database_or_a_table(tmp_path, store):
    missing = tmp_path / "missing" / "state.sqlite"
    try:
        result = _stores(missing)[store]()
    except Exception:
        result = None
    assert store == "drawdown" and result is True or result is None     # unreadable: latched, or an error
    assert not missing.parent.exists() and not missing.exists()
    empty = tmp_path / "empty.sqlite"
    empty.write_bytes(b"")
    try:
        _stores(empty)[store]()
    except Exception:
        pass
    with sqlite3.connect(empty) as db:
        assert db.execute("SELECT count(*) FROM sqlite_master").fetchone()[0] == 0


def test_only_the_schema_module_creates_tables_and_only_the_factory_connects():
    sqlite_dir = ROOT / "trading" / "adapters" / "sqlite"
    creators = sorted(p.name for p in sqlite_dir.glob("*.py") if "CREATE TABLE" in p.read_text())
    connectors = sorted(p.name for p in sqlite_dir.glob("*.py") if "sqlite3.connect(" in p.read_text())
    assert creators == ["schema.py"]                  # paper's own store went with S10; its tables are in schema.py
    # check.py deleted 2026-11-15; backup.py since S8: the online backup writes a new snapshot (ADR-0016 point 6)
    assert connectors == ["backup.py", "connection.py", "schema.py"]


def test_every_connection_uses_full_synchronous_writes_and_a_busy_timeout(tmp_path):
    from trading.adapters.sqlite.connection import connect
    db_path = _init(tmp_path / "state")
    with connect(db_path) as db:
        assert db.execute("PRAGMA synchronous").fetchone()[0] == 2           # FULL
        assert db.execute("PRAGMA busy_timeout").fetchone()[0] >= 5000


# -- while running ---------------------------------------------------------------------------------------------
def test_a_database_deleted_while_running_is_not_re_created_and_sends_no_unregistered_order(tmp_path):
    from runner.binance_target import TargetRuntime
    from tests.unit.lab.test_target_runtime import LIVE, _advance, _world
    from trading.domain.strategy.overlay import StrategySpec
    state = tmp_path / "state"
    db_path = _init(state)
    world = _world(tmp_path, StrategySpec(lookbacks=(5, 10, 20), vol_lookback=10, target_vol=0.6, band=0.05,
                                          execution="market"))          # the fake fills market orders
    runtime = TargetRuntime(spec_path=world["spec_path"], state_dir=state, mode="execution", market=world["adapter"],
                            account=world["adapter"], clock=lambda: world["clock"]["now"], testnet=True)
    runtime.warmup()
    btc, i = world["fakes"]["BTCUSDT"], world["warm"]
    while btc.positions.get("BTCUSDT", D(0)) == 0 and i < world["warm"] + LIVE - 40:
        _advance(world, runtime, i)
        i += 1
    assert btc.positions.get("BTCUSDT", D(0)) != 0
    for suffix in ("", "-wal", "-shm"):
        Path(f"{db_path}{suffix}").unlink(missing_ok=True)
    posts = [len(f.posts) for f in world["fakes"].values()]
    for k in range(1, 6):
        _advance(world, runtime, i + k)
    assert [len(f.posts) for f in world["fakes"].values()] == posts      # nothing sent without a journal row
    assert not db_path.exists()
    assert all(not s.executor.ready for s in runtime.sessions.values())   # entries blocked by reconciliation
    assert runtime.status()["symbols"]["BTCUSDT"]["protective_stop"].startswith("unreadable")


class _Notifier:
    channel = "test"

    def __init__(self):
        self.sent = []

    def alert(self, level, title, detail=""):
        self.sent.append((level, title, detail))


def _holding(tmp_path, notifier):
    """An execution runtime on a fresh state directory, advanced until BTC holds a position."""
    from runner.binance_target import TargetRuntime
    from tests.unit.lab.test_target_runtime import LIVE, _advance, _world
    from trading.domain.strategy.overlay import StrategySpec
    state = tmp_path / "state"
    db_path = _init(state)
    world = _world(tmp_path, StrategySpec(lookbacks=(5, 10, 20), vol_lookback=10, target_vol=0.6, band=0.05,
                                          execution="market"))
    runtime = TargetRuntime(spec_path=world["spec_path"], state_dir=state, mode="execution", market=world["adapter"],
                            account=world["adapter"], clock=lambda: world["clock"]["now"], testnet=True,
                            notifier=notifier)
    runtime.warmup()
    btc, i = world["fakes"]["BTCUSDT"], world["warm"]
    while btc.positions.get("BTCUSDT", D(0)) == 0 and i < world["warm"] + LIVE - 40:
        _advance(world, runtime, i)
        i += 1
    assert btc.positions.get("BTCUSDT", D(0)) != 0
    return world, runtime, db_path, i


def test_a_reduction_after_the_database_was_deleted_is_retried_and_alerted_and_the_cycle_goes_on(
        tmp_path, monkeypatch):
    # Audit 2026-11-15 of S2, finding 1: the failure handler read the lost journal again, and that error ended
    # the process (no more alerts, retries or entry cancellation; a restart is refused by the start-up check).
    # The handler was fixed in S2 (test_execution_service); the next read after the send escaped until S3c,
    # which contains a symbol's failing bar (DECISION_FAILED) instead of letting it end the cycle.
    from tests.unit.lab.test_target_runtime import _advance
    note = _Notifier()
    world, runtime, db_path, i = _holding(tmp_path, note)
    for suffix in ("", "-wal", "-shm"):
        Path(f"{db_path}{suffix}").unlink(missing_ok=True)
    session = runtime.sessions["BTCUSDT"]
    monkeypatch.setattr(session, "_target", lambda: (0.0, True))     # the strategy now wants flat: a reduction
    tried = []
    send = session.executor.send_order
    monkeypatch.setattr(session.executor, "send_order", lambda order: (tried.append(order), send(order))[1])
    posts, before = [len(f.posts) for f in world["fakes"].values()], len(note.sent)
    for k in range(1, 4):
        _advance(world, runtime, i + k)                               # no exception leaves the cycle
    assert [len(f.posts) for f in world["fakes"].values()] == posts  # never sent without a journal row
    assert not db_path.exists()
    reductions = {o.order_id for o in tried if o.reduce_only}
    assert len(reductions) == 3                                       # tried again on every bar
    assert note.sent[before:]                                         # and the failure is alerted
