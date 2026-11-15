"""The EventLog port as an append-only JSONL file (observability only, never state)."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from trading.domain.events import EventKind


class JsonlEventLog:
    """Append-only JSONL of signals, intents, fills and issues (observability only)."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, kind: EventKind, **fields: Any) -> None:
        if not isinstance(kind, EventKind):
            raise TypeError(f"an event needs an EventKind, not {kind!r}")
        row = {"ts": time.time(), "kind": kind.value, **fields}
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, default=str, ensure_ascii=False) + "\n")


__all__ = ["JsonlEventLog"]
