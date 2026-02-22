# features/online_store
"""Online feature store — real-time feature serving."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class OnlineFeatureStore:
    """在线特征存储 — 实时特征查询。"""
    _cache: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def update(self, symbol: str, name: str, value: float) -> None:
        if symbol not in self._cache:
            self._cache[symbol] = {}
        self._cache[symbol][name] = value

    def update_batch(self, symbol: str, features: Dict[str, float]) -> None:
        if symbol not in self._cache:
            self._cache[symbol] = {}
        self._cache[symbol].update(features)

    def get(self, symbol: str, name: str) -> Optional[float]:
        return self._cache.get(symbol, {}).get(name)

    def get_all(self, symbol: str) -> Dict[str, float]:
        return dict(self._cache.get(symbol, {}))

    def clear(self, symbol: str | None = None) -> None:
        if symbol is None:
            self._cache.clear()
        else:
            self._cache.pop(symbol, None)
