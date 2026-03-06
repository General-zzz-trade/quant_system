from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


@dataclass
class RunContext:
    """Standard run context with a run_id and output directory."""

    run_id: str
    out_dir: Path

    @classmethod
    def create(cls, *, out_root: str | Path = "out", prefix: str = "run") -> "RunContext":
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_id = f"{prefix}_{ts}"
        out_dir = Path(out_root) / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        return cls(run_id=run_id, out_dir=out_dir)
