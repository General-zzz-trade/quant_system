# portfolio/risk_model/input.py
"""Risk model input data."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence


@dataclass(frozen=True, slots=True)
class RiskModelInput:
    """风险模型标准化输入。"""
    symbols: tuple[str, ...]
    returns: Mapping[str, tuple[float, ...]]            # symbol → return series
    prices: Mapping[str, tuple[float, ...]] = field(default_factory=dict)
    volumes: Mapping[str, tuple[float, ...]] = field(default_factory=dict)
    timestamp_ms: int = 0

    @property
    def n_assets(self) -> int:
        return len(self.symbols)

    @property
    def n_observations(self) -> int:
        if not self.returns:
            return 0
        first = next(iter(self.returns.values()))
        return len(first)

    def returns_for(self, symbol: str) -> tuple[float, ...]:
        return self.returns.get(symbol, ())
