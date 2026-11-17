"""A consistent copy of the state database while it is in use (ADR-0016 point 6, S8): SQLite's online backup.

The copy is one file in rollback-journal mode, with no WAL beside it; the restore command puts the live database
back in WAL. The source is opened read-only and is never changed. ``copy_database`` is the one place besides the
connection factory and the schema that opens SQLite itself: the online backup writes a new file, which the
factory never creates (ADR-0009). Everything else here goes through the factory.
"""
from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

from trading.adapters.sqlite.connection import connect


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


def enable_wal(path: Path) -> None:
    """Put a restored database back in WAL mode, as ``schema.create`` sets it; the mode is kept in the file."""
    with connect(path) as db:
        if db.execute("PRAGMA journal_mode=WAL").fetchone()[0] != "wal":
            raise RuntimeError("the restored database did not switch to WAL")


def symbols_in(path: Path) -> set[str]:
    """Every symbol the database has an order, a protective stop or an entry pause for."""
    with connect(path) as db:
        return {str(r[0]) for table in ("submissions", "protective_stops", "entry_pauses")
                for r in db.execute(f"SELECT DISTINCT symbol FROM {table}")}


def record_restore(path: Path, *, scope: str, at: float, snapshot: str, after: dict[str, object],
                   reason: str) -> None:
    """The audit row of a restore, in the restored database's ``manual_actions`` (ADR-0016 point 6)."""
    with connect(path) as db:
        db.execute("INSERT INTO manual_actions VALUES (?, ?, 'restore', ?, NULL, ?, '', ?)",
                   (scope, float(at), snapshot, json.dumps(after, sort_keys=True), reason))


def record_migration(path: Path, *, scope: str, at: float, had: int, version: int, kept: str) -> None:
    """The audit row of a migration (``state_dir migrate``, ADR-0009 5), in the migrated database's
    ``manual_actions``."""
    with connect(path) as db:
        db.execute("INSERT INTO manual_actions VALUES (?, ?, 'migrate', ?, ?, ?, '', ?)",
                   (scope, float(at), f"state database version {had} to {version}", json.dumps({"version": had}),
                    json.dumps({"version": version}), f"the version-{had} copy is kept as {kept}"))


__all__ = ["copy_database", "enable_wal", "record_migration", "record_restore", "symbols_in"]
