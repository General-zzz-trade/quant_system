from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator, Optional, Sequence

from decision.persistence.decision_store import DecisionStore
from decision.types import DecisionOutput


@dataclass
class DecisionReplayer:
    store: DecisionStore

    def iter_outputs(self) -> Iterator[dict]:
        return self.store.iter_records()
