from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

from decision.persistence.serializers import dumps, loads


@dataclass
class DecisionStore:
    """Append-only JSONL store for DecisionOutput records."""
    path: Optional[str] = None

    def append(self, record: dict) -> None:
        if self.path is None:
            return  # in-memory mode: no-op
        p = Path(self.path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(dumps(record) + "\n")

    def iter_records(self) -> Iterator[dict]:
        if self.path is None:
            return iter(())  # type: ignore[return-value]
        p = Path(self.path)
        if not p.exists():
            return iter(())  # type: ignore[return-value]
        def gen():
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    line=line.strip()
                    if not line:
                        continue
                    yield loads(line)
        return gen()
