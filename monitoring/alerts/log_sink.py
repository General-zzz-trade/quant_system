"""Log-file alert sink — appends structured alerts as JSONL."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .base import Alert


@dataclass
class LogAlertSink:
    """Writes each alert as a JSON line to an append-only log file.

    Parameters
    ----------
    path : Path
        File path for the JSONL log.  Parent directories are created
        automatically.
    """
    path: Path

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, alert: Alert) -> None:
        row = alert.to_dict()
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
