"""A consistent copy of the state database while it is in use (ADR-0016 point 6, S8): SQLite's online backup.

The copy is one file in rollback-journal mode, with no WAL beside it; the restore command puts the live database
back in WAL. The source is opened read-only and is never changed.
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path


def copy_database(source: Path, target: Path) -> None:
    """Copy ``source`` to ``target`` (created) and sync it to disk; errors propagate."""
    src = sqlite3.connect(f"{Path(source).resolve().as_uri()}?mode=ro", uri=True)
    try:
        dst = sqlite3.connect(target)
        try:
            src.backup(dst)
            dst.execute("PRAGMA journal_mode=DELETE")
        finally:
            dst.close()
    finally:
        src.close()
    with open(target, "rb") as handle:
        os.fsync(handle.fileno())


__all__ = ["copy_database"]
