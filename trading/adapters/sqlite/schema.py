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
VERSION = 1

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
        finally:
            db.close()
    except sqlite3.Error:
        return [f"{name} is unreadable"]
    found = []
    if version != VERSION:
        found.append(f"{name} has version {version}, expected {VERSION}")
    if missing := [t for t in TABLES if t not in tables]:
        found.append(f"{name} has lost its tables ({', '.join(missing)}); it was emptied or replaced")
    return found


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


__all__ = ["PAPER_TABLES", "STATE_DB", "TABLES", "VERSION", "create", "holds_paper_state", "problems"]
