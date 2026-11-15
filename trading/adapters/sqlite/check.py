"""Whether a file is a readable SQLite database, opened read-only (for the state-directory check, step 13c)."""
from __future__ import annotations

import sqlite3
from pathlib import Path


def database_readable(path: Path) -> bool:
    try:
        with sqlite3.connect(f"file:{Path(path)}?mode=ro", uri=True) as db:
            db.execute("SELECT count(*) FROM sqlite_master").fetchone()
        return True
    except sqlite3.Error:
        return False


__all__ = ["database_readable"]
