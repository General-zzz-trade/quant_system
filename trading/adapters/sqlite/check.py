"""Whether a file is a readable SQLite database, opened read-only (for the state-directory check, step 13c)."""
from __future__ import annotations

import sqlite3
from pathlib import Path


def database_tables(path: Path) -> set[str] | None:
    """The tables of the database at ``path``, or None if it cannot be read. A file truncated to 0 bytes reads
    as a database with no tables (audit after M3, finding 1), so callers check the tables they need."""
    try:
        db = sqlite3.connect(f"{Path(path).resolve().as_uri()}?mode=ro", uri=True)
        try:
            return {str(r[0]) for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        finally:
            db.close()
    except sqlite3.Error:
        return None


__all__ = ["database_tables"]
