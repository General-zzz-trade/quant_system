from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

from decision.persistence.decision_store import DecisionStore


@dataclass
class DecisionReplayer:
    store: DecisionStore

    def iter_outputs(self) -> Iterator[dict[str, Any]]:
        return self.store.iter_records()
