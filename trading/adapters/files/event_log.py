"""The EventLog port as an append-only JSONL file (observability only, never state)."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class JsonlEventLog:
    """Append-only JSONL of signals, intents, fills and issues (observability only)."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, kind: str, **fields: Any) -> None:
        row = {"ts": time.time(), "kind": kind, **fields}
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, default=str, ensure_ascii=False) + "\n")


__all__ = ["JsonlEventLog"]
