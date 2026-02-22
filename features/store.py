from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

from .types import FeatureName, FeatureSeries


@dataclass
class FeatureStore:
    """A simple in-memory feature store.

    Keys are (symbol, timeframe, feature_name).
    Values are feature series aligned with bars.

    This store is designed to be deterministic and easy to test.
    """

    _data: Dict[Tuple[str, str, FeatureName], FeatureSeries]

    def __init__(self) -> None:
        self._data = {}

    def put(self, *, symbol: str, timeframe: str, name: FeatureName, series: FeatureSeries) -> None:
        self._data[(symbol.upper(), timeframe, name)] = list(series)

    def get(self, *, symbol: str, timeframe: str, name: FeatureName) -> Optional[FeatureSeries]:
        return self._data.get((symbol.upper(), timeframe, name))

    def has(self, *, symbol: str, timeframe: str, name: FeatureName) -> bool:
        return (symbol.upper(), timeframe, name) in self._data

    def clear(self) -> None:
        self._data.clear()
