# features/offline_store
"""Offline feature store — batch feature computation and storage."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence


@dataclass
class OfflineFeatureStore:
    """离线特征存储 — 批量计算后的特征快照。"""
    _store: Dict[str, Dict[str, List[float]]] = field(default_factory=dict)

    def put(self, symbol: str, features: Dict[str, List[float]]) -> None:
        self._store[symbol] = features

    def get(self, symbol: str) -> Dict[str, List[float]]:
        return self._store.get(symbol, {})

    def get_feature(self, symbol: str, name: str) -> List[float]:
        return self._store.get(symbol, {}).get(name, [])

    def symbols(self) -> list[str]:
        return list(self._store.keys())

    def feature_names(self, symbol: str) -> list[str]:
        return list(self._store.get(symbol, {}).keys())
