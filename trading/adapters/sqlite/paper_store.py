"""The PaperStore port in SQLite: one current document, a version number, and an archive table."""
from __future__ import annotations

import json
import sqlite3
import time
from contextlib import closing
from pathlib import Path
from typing import Any, Callable

from trading.adapters.codec import decode, encode

VERSION = 1


class SqlitePaperStore:
    def __init__(self, path: Path, *, clock: Callable[[], float] = time.time) -> None:
        self.path, self.clock = Path(path), clock
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as db, db:
            db.execute("CREATE TABLE IF NOT EXISTS paper_state (id INTEGER PRIMARY KEY CHECK (id = 1), "
                       "version INTEGER NOT NULL, saved_at REAL NOT NULL, document TEXT NOT NULL, "
                       "mirror_digest TEXT)")
            db.execute("CREATE TABLE IF NOT EXISTS paper_state_archive (archived_at REAL NOT NULL, "
                       "version INTEGER NOT NULL, document TEXT NOT NULL)")

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=5)
        db.execute("PRAGMA synchronous=FULL")
        return db

    def load(self) -> dict[str, Any] | None:
        with closing(self._connect()) as db:
            row = db.execute("SELECT version, document FROM paper_state WHERE id = 1").fetchone()
        if row is None:
            return None
        if row[0] != VERSION:
            raise RuntimeError(f"paper state has version {row[0]}, this build reads version {VERSION}")
        document: dict[str, Any] = json.loads(row[1], object_hook=decode)
        return document

    def save(self, document: dict[str, Any], *, mirror_digest: str | None = None) -> None:
        """Replace the current document in one transaction; the mirror digest is kept unless given."""
        text = json.dumps(document, default=encode)
        with closing(self._connect()) as db, db:
            db.execute("INSERT INTO paper_state (id, version, saved_at, document, mirror_digest) VALUES (1,?,?,?,?) "
                       "ON CONFLICT(id) DO UPDATE SET version=excluded.version, saved_at=excluded.saved_at, "
                       "document=excluded.document, mirror_digest=COALESCE(excluded.mirror_digest, mirror_digest)",
                       (VERSION, float(self.clock()), text, mirror_digest))

    def mirror_digest(self) -> str | None:
        with closing(self._connect()) as db:
            row = db.execute("SELECT mirror_digest FROM paper_state WHERE id = 1").fetchone()
        return row[0] if row else None

    def set_mirror_digest(self, digest: str | None) -> None:
        with closing(self._connect()) as db, db:
            db.execute("UPDATE paper_state SET mirror_digest = ? WHERE id = 1", (digest,))

    def archive(self) -> str | None:
        with closing(self._connect()) as db, db:
            row = db.execute("SELECT version, document FROM paper_state WHERE id = 1").fetchone()
            if row is None:
                return None
            stamp = float(self.clock())
            db.execute("INSERT INTO paper_state_archive VALUES (?,?,?)", (stamp, row[0], row[1]))
            db.execute("DELETE FROM paper_state WHERE id = 1")
        return f"sqlite:{int(stamp * 1000)}"


__all__ = ["SqlitePaperStore", "VERSION"]
