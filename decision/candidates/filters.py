from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Sequence

from decision.types import Candidate


@dataclass(frozen=True, slots=True)
class CandidateFilter:
    min_abs_score: Decimal = Decimal("0")

    def apply(self, candidates: Sequence[Candidate]) -> Sequence[Candidate]:
        if self.min_abs_score <= 0:
            return list(candidates)
        return [c for c in candidates if abs(c.score) >= self.min_abs_score]
