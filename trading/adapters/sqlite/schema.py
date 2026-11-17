"""The state database of an execution state directory (ADR-0009): one file, a version number, every table.

Only ``create`` - called by ``state_dir init``, which a person runs - creates the file and its tables. The
stores open the existing file through ``trading.adapters.sqlite.connection`` and never create anything, so a
database that was deleted or emptied stays visibly missing instead of being silently re-created empty.
Changing a table means a new version and a migration a person runs with the runtime stopped (ADR-0009 5).
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

STATE_DB = "state.sqlite"
VERSION = 2          # 2 (tech lead 2026-11-17): the strategy's funding rows and how far each symbol's are known

# The columns are exactly those the stores used before ADR-0009 (the order table with all its added columns).
TABLES: dict[str, str] = {
    "submissions": """CREATE TABLE submissions (
        scope TEXT, intent_id TEXT, symbol TEXT, client_id TEXT, order_id TEXT,
        settled INTEGER NOT NULL DEFAULT 0, side TEXT, qty TEXT, reducing INTEGER,
        owner TEXT NOT NULL DEFAULT '', parent TEXT, status TEXT NOT NULL DEFAULT 'unknown', order_type TEXT,
        created_at REAL, PRIMARY KEY(scope, intent_id))""",
    "execution_fills": """CREATE TABLE execution_fills (
        scope TEXT NOT NULL, fill_id TEXT NOT NULL, intent_id TEXT NOT NULL,
        owner TEXT NOT NULL, payload TEXT NOT NULL, delivered INTEGER NOT NULL DEFAULT 0,
        PRIMARY KEY(scope, fill_id))""",
    "execution_consumers": """CREATE TABLE execution_consumers (
        scope TEXT NOT NULL, owner TEXT NOT NULL, checkpoint TEXT NOT NULL,
        PRIMARY KEY(scope, owner))""",
    "manual_actions": """CREATE TABLE manual_actions (
        scope TEXT NOT NULL, at REAL NOT NULL, kind TEXT NOT NULL, subject TEXT NOT NULL,
        before TEXT, after TEXT NOT NULL, venue TEXT NOT NULL, reason TEXT NOT NULL)""",
    "protective_stops": """CREATE TABLE protective_stops (
        scope TEXT NOT NULL, owner TEXT NOT NULL, client_algo_id TEXT NOT NULL,
        symbol TEXT NOT NULL, revision TEXT NOT NULL, attempt INTEGER NOT NULL,
        side TEXT NOT NULL, trigger_price TEXT NOT NULL, status TEXT NOT NULL,
        created_at REAL NOT NULL, PRIMARY KEY(scope, client_algo_id))""",
    "drawdown_periods": """CREATE TABLE drawdown_periods (
        scope TEXT, kind TEXT, period TEXT, baseline REAL NOT NULL,
        limit_pct REAL NOT NULL, tripped INTEGER NOT NULL DEFAULT 0,
        PRIMARY KEY(scope, kind, period))""",
    "equity_observations": """CREATE TABLE equity_observations (
        scope TEXT PRIMARY KEY, equity REAL NOT NULL, observed_at REAL NOT NULL)""",
    "order_limits": """CREATE TABLE order_limits (
        scope TEXT NOT NULL, day TEXT NOT NULL, orders INTEGER NOT NULL, notional TEXT NOT NULL,
        PRIMARY KEY(scope, day))""",
    "entry_pauses": """CREATE TABLE entry_pauses (
        symbol TEXT NOT NULL, paused_at REAL NOT NULL, cause TEXT NOT NULL,
        released_at REAL, release_reason TEXT)""",
    "pause_triggers": """CREATE TABLE pause_triggers (
        symbol TEXT NOT NULL, trigger TEXT NOT NULL, PRIMARY KEY(symbol, trigger))""",
    # Version 2: every funding row read from the venue (by its transaction id) and, per symbol, the span its rows are
    # known whole for (lab/README and CLAUDE.md: Binance keeps the income history for three months only).
    "funding_rows": """CREATE TABLE funding_rows (
        scope TEXT NOT NULL, symbol TEXT NOT NULL, tran_id TEXT NOT NULL, time_ms INTEGER NOT NULL,
        amount TEXT NOT NULL, PRIMARY KEY(scope, tran_id))""",
    "funding_coverage": """CREATE TABLE funding_coverage (
        scope TEXT NOT NULL, symbol TEXT NOT NULL, from_ms INTEGER NOT NULL, through_ms INTEGER NOT NULL,
        PRIMARY KEY(scope, symbol))""",
}
# A paper directory's database also holds the paper state, written with the order journal (S10 decision 4). Not part
# of the execution check below: an execution directory has no paper state.
PAPER_TABLES: dict[str, str] = {
    "paper_state": """CREATE TABLE paper_state (
        id INTEGER PRIMARY KEY CHECK (id = 1), saved_at REAL NOT NULL, document TEXT NOT NULL)""",
}


def create(path: Path, *, paper: bool = False) -> None:
    """Create the database at ``path`` with every table and the version, and with ``paper`` the paper state's table
    too; refuse if the file exists."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} already exists; a state database is created once")
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    try:
        db.execute("PRAGMA journal_mode=WAL")            # persistent in the file
        with db:
            for ddl in [*TABLES.values(), *(PAPER_TABLES.values() if paper else ())]:
                db.execute(ddl)
            db.execute(f"PRAGMA user_version={VERSION}")
    finally:
        db.close()


def problems(path: Path) -> list[str]:
    """Why the database at ``path`` cannot be used (empty when it can); read-only, creates nothing."""
    name = Path(path).name
    if not Path(path).is_file():
        return [f"{name} is missing"]
    try:
        db = sqlite3.connect(f"{Path(path).resolve().as_uri()}?mode=ro", uri=True)
        try:
            version = int(db.execute("PRAGMA user_version").fetchone()[0])
            tables = {str(r[0]) for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            # Every page, not only the schema page: a corrupt page reads as intact otherwise (S6-S8 audit, M4).
            check = [str(r[0]) for r in db.execute("PRAGMA quick_check(5)")]
        finally:
            db.close()
    except sqlite3.Error:
        return [f"{name} is unreadable"]
    found = [] if check == ["ok"] else [f"{name} is corrupt ({'; '.join(check)})"]
    if version != VERSION:
        hint = "; migrate it with the runtime stopped: python -m trading.infrastructure.state_dir --state-dir " \
            "<dir> migrate" if version in MIGRATIONS else ""
        found.append(f"{name} has version {version}, expected {VERSION}{hint}")
    later = {t for v in range(version, VERSION) for t in MIGRATIONS.get(v, ())}   # not yet in an older version
    if missing := [t for t in TABLES if t not in tables and t not in later]:
        found.append(f"{name} has lost its tables ({', '.join(missing)}); it was emptied or replaced")
    return found


def version(path: Path) -> int:
    """The database's version number; read-only."""
    db = sqlite3.connect(f"{Path(path).resolve().as_uri()}?mode=ro", uri=True)
    try:
        return int(db.execute("PRAGMA user_version").fetchone()[0])
    finally:
        db.close()


# The tables each migration adds, by the version it starts from (ADR-0009 5: a person runs it, the runtime stopped).
MIGRATIONS: dict[int, tuple[str, ...]] = {1: ("funding_rows", "funding_coverage")}


def migrate(path: Path) -> int:
    """Bring the database at ``path`` to ``VERSION`` by adding the tables of each step, in one transaction; returns
    the version it had. Nothing else changes. A version without a migration is refused."""
    db = sqlite3.connect(Path(path))
    try:
        had = int(db.execute("PRAGMA user_version").fetchone()[0])
        if had == VERSION:
            return had
        steps = list(range(had, VERSION))
        if not steps or not all(step in MIGRATIONS for step in steps):
            raise ValueError(f"{Path(path).name} has version {had}; there is no migration from it to {VERSION}")
        with db:
            for step in steps:
                for table in MIGRATIONS[step]:
                    db.execute(TABLES[table])
            db.execute(f"PRAGMA user_version={VERSION}")
        return had
    finally:
        db.close()


def holds_paper_state(path: Path) -> bool:
    """Whether the database at ``path`` is a paper record's (it has the paper state's table); read-only. Execution
    refuses such a database (S10 audit H1): a paper run's orders and positions are not an account's."""
    try:
        db = sqlite3.connect(f"{Path(path).resolve().as_uri()}?mode=ro", uri=True)
        try:
            return db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name IN ({})".format(
                ",".join("?" * len(PAPER_TABLES))), tuple(PAPER_TABLES)).fetchone() is not None
        finally:
            db.close()
    except sqlite3.Error:
        return False


__all__ = ["MIGRATIONS", "PAPER_TABLES", "STATE_DB", "TABLES", "VERSION", "create", "holds_paper_state",
           "migrate", "problems", "version"]
