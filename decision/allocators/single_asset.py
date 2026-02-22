from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Mapping, Sequence

from decision.types import Candidate


@dataclass(frozen=True, slots=True)
class SingleAssetAllocator:
    """Pick the strongest candidate and allocate 100% decision budget."""
    def allocate(self, candidates: Sequence[Candidate]) -> Mapping[str, Decimal]:
        if not candidates:
            return {}
        best = max(candidates, key=lambda c: abs(c.score))
        return {best.symbol: Decimal("1")}
