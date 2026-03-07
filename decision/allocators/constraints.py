from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Mapping

from _quant_hotpath import rust_apply_allocation_constraints as _rust_constraints


@dataclass(frozen=True, slots=True)
class AllocationConstraints:
    max_positions: int = 1

    def apply(self, weights: Mapping[str, Decimal]) -> Mapping[str, Decimal]:
        if not weights:
            return {}

        float_weights = {k: float(v) for k, v in weights.items()}
        result = _rust_constraints(float_weights, self.max_positions)
        return {k: Decimal(str(v)) for k, v in result.items()}
