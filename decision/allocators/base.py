from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Mapping, Protocol, Sequence

from decision.types import Candidate


class Allocator(Protocol):
    def allocate(self, candidates: Sequence[Candidate]) -> Mapping[str, Decimal]:
        ...


@dataclass(frozen=True, slots=True)
class EqualWeightAllocator:
    def allocate(self, candidates: Sequence[Candidate]) -> Mapping[str, Decimal]:
        if not candidates:
            return {}
        w = Decimal("1") / Decimal(str(len(candidates)))
        return {c.symbol: w for c in candidates}
