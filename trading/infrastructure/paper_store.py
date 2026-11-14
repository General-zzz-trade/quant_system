"""Paper state: SQLite is the store, and a JSON file with the same content is written beside it.

The JSON mirror keeps a rollback to the code before migration step 10 possible (that code reads only the
file). If the file was changed from outside since the last save here - a rolled-back version ran, or
someone edited it - the file wins and is imported again; a mirror that is merely one save behind does not.
"""
from __future__ import annotations

from typing import Any

from trading.adapters.files.state_files import JsonPaperStore
from trading.adapters.sqlite.paper_store import SqlitePaperStore


class MirroredPaperStore:
    def __init__(self, primary: SqlitePaperStore, mirror: JsonPaperStore) -> None:
        self.primary, self.mirror = primary, mirror

    def load(self) -> dict[str, Any] | None:
        document = self.primary.load()
        digest = self.mirror.digest()
        if digest is not None and (document is None or digest != self.primary.mirror_digest()):
            mirrored = self.mirror.load()
            if mirrored is not None and mirrored != document:
                self.primary.save(mirrored, mirror_digest=digest)
                return mirrored
            self.primary.set_mirror_digest(digest)
        return document

    def save(self, document: dict[str, Any]) -> None:
        self.primary.save(document)
        self.mirror.save(document)
        self.primary.set_mirror_digest(self.mirror.digest())

    def archive(self) -> str | None:
        in_database = self.primary.archive()
        return self.mirror.archive() or in_database


__all__ = ["MirroredPaperStore"]
