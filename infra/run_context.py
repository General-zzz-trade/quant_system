from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict


@dataclass
class RunContext:
    """Standard run context.

    Provides a run_id, output directory, and common tags.
    """

    out_root: Path
    run_id: str = ""
    tags: Dict[str, str] = None  # type: ignore

    def __post_init__(self) -> None:
        if not self.run_id:
            self.run_id = uuid.uuid4().hex
        if self.tags is None:
            self.tags = {}
        self.out_root.mkdir(parents=True, exist_ok=True)

    @property
    def out_dir(self) -> Path:
        p = self.out_root / self.run_id
        p.mkdir(parents=True, exist_ok=True)
        return p

    def write_marker(self, name: str, content: str) -> Path:
        p = self.out_dir / name
        p.write_text(content, encoding="utf-8")
        return p
