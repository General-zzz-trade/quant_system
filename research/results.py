from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class RunResult:
    """Result bundle for a single experiment run."""

    run_id: str
    experiment: str
    started_at: datetime
    finished_at: datetime
    metrics: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "experiment": self.experiment,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat(),
            "metrics": self.metrics,
            "artifacts": self.artifacts,
        }

    def save(self, out_dir: str | Path) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / "run_result.json"
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return path
