"""Paper state and runtime status as files: atomic JSON writes that survive a crash or a power loss."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Callable

from trading.adapters.codec import decode as _decode
from trading.adapters.codec import encode as _encode


class JsonPaperStore:
    """The PaperStore port as one JSON file (``paper_state.json``), written atomically and fsynced."""

    def __init__(self, path: Path, *, clock: Callable[[], float] = time.time) -> None:
        self.path, self.clock = Path(path), clock

    def load(self) -> dict[str, Any] | None:
        if not self.path.exists():
            return None
        document: dict[str, Any] = json.loads(self.path.read_text(), object_hook=_decode)
        return document

    def save(self, document: dict[str, Any]) -> None:
        tmp = self.path.with_name(self.path.name + ".tmp")
        with tmp.open("w", encoding="utf-8") as stream:
            json.dump(document, stream, default=_encode)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, self.path)
        directory = os.open(self.path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)              # the rename itself survives a power loss
        finally:
            os.close(directory)

    def digest(self) -> str | None:
        """SHA-256 of the file as it is on disk, or None when there is no file."""
        return hashlib.sha256(self.path.read_bytes()).hexdigest() if self.path.exists() else None

    def archive(self) -> str | None:
        if not self.path.exists():
            return None
        stamp, n = int(self.clock() * 1000), 0
        archived = self.path.with_name(f"paper_state.{stamp}.json")
        while archived.exists():
            n += 1
            archived = self.path.with_name(f"paper_state.{stamp}.{n}.json")
        os.replace(self.path, archived)
        return archived.name


class JsonStatusFile:
    """The StatusSink port: the latest status as one JSON file, replaced atomically."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def write(self, status: dict[str, Any]) -> None:
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(status, default=str, indent=2))
        os.replace(tmp, self.path)


__all__ = ["JsonPaperStore", "JsonStatusFile"]
