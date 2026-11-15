"""Test helper: a state database as `state_dir init` creates it (ADR-0009).

Since ADR-0009 the stores never create a file or a table, so a test that builds a store or an execution
runtime directly creates the database first. This only changes the setup; the schema is the production one.
"""
from __future__ import annotations

from pathlib import Path

from trading.adapters.sqlite import schema


def state_db(path: Path) -> Path:
    """The database at ``path``, created with every table if it does not exist yet; returns ``path``."""
    path = Path(path)
    if not path.exists():
        schema.create(path)
    return path


def state_dir_db(directory: Path) -> Path:
    """The state database inside ``directory`` (``state.sqlite``), created if needed."""
    return state_db(Path(directory) / schema.STATE_DB)


__all__ = ["state_db", "state_dir_db"]


def database_tables(path: Path) -> set[str] | None:
    """The tables of the database at ``path``, opened read-only, or None if it cannot be read. Moved here from
    trading/adapters/sqlite/check.py, which no production code used any more (S2 audit, finding 6)."""
    import sqlite3
    try:
        db = sqlite3.connect(f"{Path(path).resolve().as_uri()}?mode=ro", uri=True)
        try:
            return {str(r[0]) for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        finally:
            db.close()
    except sqlite3.Error:
        return None
