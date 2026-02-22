from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Mapping


@dataclass(frozen=True, slots=True)
class AllocationConstraints:
    max_positions: int = 1

    def apply(self, weights: Mapping[str, Decimal]) -> Mapping[str, Decimal]:
        if not weights:
            return {}
        items = sorted(weights.items(), key=lambda kv: abs(kv[1]), reverse=True)
        items = items[: self.max_positions]
        total = sum([abs(w) for _, w in items])
        if total <= 0:
            return {k: Decimal("0") for k, _ in items}
        return {k: (w / total) for k, w in items}
