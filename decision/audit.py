from __future__ import annotations

from dataclasses import dataclass

from decision.persistence.decision_store import DecisionStore
from decision.types import DecisionOutput


@dataclass
class DecisionAuditor:
    store: DecisionStore

    def record(self, out: DecisionOutput) -> None:
        self.store.append(out.to_dict())
